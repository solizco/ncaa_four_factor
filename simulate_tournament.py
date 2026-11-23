"""Train the Four Factors model and simulate a full NCAA tournament.

For --season S:
  1. Train logistic regression on every game through S's regular season and
     every tournament before S (so S's tournament is never seen in training).
  2. Build each tournament team's season Four Factors from S's regular season.
  3. Cache P(A beats B) for every pair of the 68 teams.
  4. Simulate the bracket --sims times using Kaggle's seeds and slots files.

Outputs in results/<season>/:
  frequency_table.csv   probability each team reaches each round
  bracket.csv           most likely winner of every slot
  picks.md              champion, Cinderella, fade and contrarian picks
  champion_odds.png     top-15 championship probabilities

Usage:
    python simulate_tournament.py --season 2025
    python simulate_tournament.py --season 2026 --sims 20000
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from build_master_table import build
from ncaa_ff import data, simulate
from ncaa_ff.eda import tourney_seed_games
from ncaa_ff.features import season_profiles
from ncaa_ff.model import train

ROUND_NAMES = {0: "First Four", 1: "Round of 64", 2: "Round of 32", 3: "Sweet 16",
               4: "Elite 8", 5: "Final Four", 6: "Championship"}


def historical_favorite_rates(data_dir: Path, before: int) -> dict:
    """Round-of-64 win rate of the better seed, by seed line, in past tournaments."""
    tourney = data.load("MNCAATourneyDetailedResults.csv", data_dir)
    tourney = tourney[tourney["Season"] < before]
    games = tourney_seed_games(tourney, data.load("MNCAATourneySeeds.csv", data_dir))
    r64 = games[(games["WinnerSeed"] + games["LoserSeed"] == 17)]
    fav = r64[["WinnerSeed", "LoserSeed"]].min(axis=1)
    fav_won = r64["WinnerSeed"] < r64["LoserSeed"]
    return fav_won.groupby(fav).mean().to_dict()


def write_picks(season, freq, bracket, prob, seed_of, names, hist, metrics, coefs, out):
    champ = freq.iloc[0]
    cinderella = freq[(freq["SeedNum"] >= 10)].sort_values("Sweet16_Prob", ascending=False).iloc[0]
    fade = freq[freq["SeedNum"] <= 4].sort_values("R32_Prob").iloc[0]

    rows = []
    for _, g in bracket[bracket["Round"] == 1].iterrows():
        a, b = g["TeamA"], g["TeamB"]
        fav, dog = (a, b) if seed_of[a] < seed_of[b] else (b, a)
        model_p = prob[(fav, dog)]
        hist_p = hist.get(seed_of[fav], float("nan"))
        rows.append((hist_p - model_p, fav, dog, model_p, hist_p))
    gap, fav, dog, model_p, hist_p = max(rows)

    final_slot = bracket[bracket["Round"] == 6].iloc[0]
    lines = [
        f"# {season} NCAA Tournament — model picks",
        "",
        f"Model: logistic regression on Four Factor differentials, trained on "
        f"{metrics['train_games']:,} games (hold-out accuracy {metrics['accuracy']:.1%}, "
        f"log loss {metrics['log_loss']:.3f}, ROC AUC {metrics['roc_auc']:.3f}).",
        "",
        "Standardized coefficients (log-odds per 1 SD): " +
        ", ".join(f"{k} {v:+.2f}" for k, v in coefs.items()),
        "",
        f"**Most likely champion:** {champ['TeamName']} ({champ['Seed']}) — "
        f"wins the title in {champ['Champion_Prob']:.1%} of simulations.",
        "",
        f"**Bracket champion (slot-by-slot most likely winners):** {names[final_slot['Winner']]}",
        "",
        f"**Cinderella:** {cinderella['TeamName']} (seed {cinderella['SeedNum']}) — "
        f"reaches the Sweet 16 in {cinderella['Sweet16_Prob']:.1%} of simulations.",
        "",
        f"**Fade:** {fade['TeamName']} (seed {fade['SeedNum']}) — "
        f"wins its first game only {fade['R32_Prob']:.1%} of the time, the lowest of any 1–4 seed.",
        "",
        f"**Contrarian:** {names[fav]} (seed {seed_of[fav]}) vs {names[dog]} (seed {seed_of[dog]}). "
        f"Historically, {seed_of[fav]}-seeds win this game {hist_p:.0%} of the time; "
        f"the model gives {names[fav]} {model_p:.0%}.",
        "",
    ]
    (out / "picks.md").write_text("\n".join(lines))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--sims", type=int, default=10_000)
    parser.add_argument("--data-dir", type=Path, default=data.DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=None, help="default: results/<season>")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = args.out or Path("results") / str(args.season)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Training on games before the {args.season} tournament ...")
    model = train(build(args.season, args.data_dir), seed=args.seed)
    print("      " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v:,}"
                                 for k, v in model.metrics.items()))

    print(f"[2/4] Building {args.season} Four Factor profiles ...")
    seeds = data.seeds_for(args.season, args.data_dir)
    if seeds.empty:
        raise SystemExit(f"No seeds for {args.season} in MNCAATourneySeeds.csv")
    slots = data.load("MNCAATourneySlots.csv", args.data_dir)
    slots = simulate.slot_order(slots[slots["Season"] == args.season])
    reg = data.load("MRegularSeasonDetailedResults.csv", args.data_dir)
    profiles = season_profiles(reg, args.season)
    seed_to_team = dict(zip(seeds["Seed"], seeds["TeamID"]))
    teams = list(seed_to_team.values())
    missing = [t for t in teams if t not in profiles.index]
    if missing:
        raise SystemExit(f"No regular-season box scores for TeamIDs {missing}")

    print(f"[3/4] Simulating {args.sims:,} tournaments ...")
    prob = simulate.probability_cache(teams, profiles, model)
    t0 = time.time()
    reach, matchup_wins = simulate.run(slots, seed_to_team, prob, args.sims, args.seed)
    print(f"      {time.time() - t0:.1f}s")

    print("[4/4] Writing results ...")
    names = data.team_names(args.data_dir)
    seed_of = dict(zip(seeds["TeamID"], seeds["SeedNum"]))
    team_seed = dict(zip(seeds["TeamID"], seeds["Seed"]))

    freq = pd.DataFrame({f"{s}_Prob": [reach[t][i] / args.sims for t in teams]
                         for i, s in enumerate(simulate.STAGES)})
    freq.insert(0, "TeamName", [names[t] for t in teams])
    freq.insert(1, "Seed", [team_seed[t] for t in teams])
    freq.insert(2, "SeedNum", [seed_of[t] for t in teams])
    freq.insert(3, "TeamID", teams)
    freq = freq.sort_values(["Champion_Prob", "Final4_Prob"], ascending=False).reset_index(drop=True)
    freq.round(4).to_csv(out / "frequency_table.csv", index=False)
    print(f"      Championship probabilities sum to {freq['Champion_Prob'].sum():.4f}")

    bracket = simulate.most_likely_bracket(slots, seed_to_team, matchup_wins, prob)
    bracket_out = bracket.assign(
        RoundName=bracket["Round"].map(ROUND_NAMES),
        TeamA=bracket["TeamA"].map(names), TeamB=bracket["TeamB"].map(names),
        Winner=bracket["Winner"].map(names))
    bracket_out[["Round", "RoundName", "Slot", "TeamA", "TeamB", "Winner"]].to_csv(
        out / "bracket.csv", index=False)

    top = freq.head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["TeamName"] + " (" + top["SeedNum"].astype(str) + ")",
            top["Champion_Prob"] * 100, color="#3949ab")
    ax.invert_yaxis()
    ax.set_xlabel("Championship probability (%)")
    ax.set_title(f"{args.season}: top 15 title odds over {args.sims:,} simulations")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "champion_odds.png", dpi=150)

    hist = historical_favorite_rates(args.data_dir, args.season)
    print()
    print(write_picks(args.season, freq, bracket, prob, seed_of, names, hist,
                      model.metrics, model.coefficients, out))
    print(f"Saved results to {out}/")


if __name__ == "__main__":
    main()
