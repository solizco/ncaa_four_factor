"""Score a simulated bracket against what actually happened.

Compares the Four Factors model with two simple baselines on a completed
tournament (default 2025):
  - game level: accuracy and log loss of P(winner) for every game played,
    against a logistic regression on seed difference alone
  - bracket level: ESPN-style score (10/20/40/80/160/320) of the model's
    bracket (results/<season>/bracket.csv) against an all-favorites bracket

Run simulate_tournament.py for the same season first.

Usage:
    python evaluate_tournament.py --season 2025
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from build_master_table import build
from ncaa_ff import data
from ncaa_ff.eda import tourney_seed_games
from ncaa_ff.features import matchup_features, season_profiles
from ncaa_ff.model import train
from ncaa_ff.simulate import slot_order

POINTS = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160, 6: 320}


def log_loss(p):
    return float(-np.mean(np.log(np.clip(p, 1e-6, 1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--data-dir", type=Path, default=data.DEFAULT_DATA_DIR)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    season, d = args.season, args.data_dir

    tourney = data.load("MNCAATourneyDetailedResults.csv", d)
    played = tourney[tourney["Season"] == season]
    if played.empty:
        raise SystemExit(f"No {season} tournament results in the data yet.")
    seeds = data.seeds_for(season, d)
    seed_num = dict(zip(seeds["TeamID"], seeds["SeedNum"]))
    names = data.team_names(d)

    # ── Game level ─────────────────────────────────────────────────────────
    model = train(build(season, d))
    profiles = season_profiles(data.load("MRegularSeasonDetailedResults.csv", d), season)
    X = np.array([matchup_features(profiles.loc[w], profiles.loc[l])
                  for w, l in zip(played["WTeamID"], played["LTeamID"])])
    p_model = model.win_probability(X)

    past = tourney_seed_games(tourney[tourney["Season"] < season],
                              data.load("MNCAATourneySeeds.csv", d))
    gap = (past["LoserSeed"] - past["WinnerSeed"]).to_numpy()
    seed_lr = LogisticRegression().fit(np.r_[gap, -gap].reshape(-1, 1),
                                       np.r_[np.ones(len(gap)), np.zeros(len(gap))])
    gap_now = np.array([seed_num[l] - seed_num[w]
                        for w, l in zip(played["WTeamID"], played["LTeamID"])])
    p_seed = seed_lr.predict_proba(gap_now.reshape(-1, 1))[:, 1]

    print(f"{season} tournament: {len(played)} games")
    print(f"  Four Factors model   correct {np.mean(p_model > 0.5):.1%}   log loss {log_loss(p_model):.3f}")
    print(f"  Seed-only baseline   correct {np.mean(p_seed > 0.5):.1%}   log loss {log_loss(p_seed):.3f}")

    # ── Bracket level ──────────────────────────────────────────────────────
    slots = data.load("MNCAATourneySlots.csv", d)
    slots = slot_order(slots[slots["Season"] == season])
    seed_to_team = dict(zip(seeds["Seed"], seeds["TeamID"]))
    results = set(zip(played["WTeamID"], played["LTeamID"]))

    actual, chalk = {}, {}
    chalk_score = 0
    for slot, strong, weak in slots:
        a = actual.get(strong, seed_to_team.get(strong))
        b = actual.get(weak, seed_to_team.get(weak))
        actual[slot] = a if (a, b) in results else b

        ca = chalk.get(strong, seed_to_team.get(strong))
        cb = chalk.get(weak, seed_to_team.get(weak))
        chalk[slot] = ca if seed_num[ca] <= seed_num[cb] else cb
        if slot.startswith("R"):
            chalk_score += POINTS[int(slot[1])] * (chalk[slot] == actual[slot])

    bracket_file = args.results / str(season) / "bracket.csv"
    if bracket_file.is_file():
        bracket = pd.read_csv(bracket_file)
        name_to_id = {v: k for k, v in names.items()}
        bracket = bracket[bracket["Round"] > 0]
        hit = [name_to_id[w] == actual[s] for s, w in zip(bracket["Slot"], bracket["Winner"])]
        bracket = bracket.assign(Correct=hit)
        score = int((bracket["Round"].map(POINTS) * bracket["Correct"]).sum())
        by_round = bracket.groupby("Round")["Correct"].agg(["sum", "count"])
        print(f"\n  Model bracket score   {score} / 1920   "
              + "  ".join(f"R{r}: {int(s)}/{c}" for r, (s, c) in by_round.iterrows()))
    else:
        print(f"\n  (no {bracket_file}; run simulate_tournament.py --season {season})")
    print(f"  All-favorites score   {chalk_score} / 1920")
    print(f"  Actual champion: {names[actual['R6CH']]}")


if __name__ == "__main__":
    main()
