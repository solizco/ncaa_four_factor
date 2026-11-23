"""Monte Carlo simulation of a full NCAA tournament bracket."""

from collections import Counter

import numpy as np
import pandas as pd

from .features import matchup_features

STAGES = ["R32", "Sweet16", "Elite8", "Final4", "ChampGame", "Champion"]


def slot_order(slots: pd.DataFrame) -> list:
    """(slot, strong, weak) tuples: First Four play-ins first, then rounds 1-6."""
    def key(name):
        return (0, name) if not name.startswith("R") else (int(name[1]), name)

    ordered = slots.sort_values("Slot", key=lambda s: s.map(key))
    return list(zip(ordered["Slot"], ordered["StrongSeed"], ordered["WeakSeed"]))


def probability_cache(teams, profiles: pd.DataFrame, model) -> dict:
    """P(A beats B) for every ordered pair of tournament teams, computed once."""
    pairs = [(a, b) for a in teams for b in teams if a != b]
    X = np.array([matchup_features(profiles.loc[a], profiles.loc[b]) for a, b in pairs])
    probs = model.win_probability(X)
    return dict(zip(pairs, probs))


def run(slots: list, seed_to_team: dict, prob: dict, n_sims: int = 10_000, seed: int = 42):
    """Simulate the bracket n_sims times.

    Returns (reach, matchup_wins): reach[team][stage] counts how often a team
    reached at least that stage; matchup_wins counts (slot, team_a, team_b, winner).
    """
    rng = np.random.default_rng(seed)
    reach = {t: np.zeros(len(STAGES), dtype=np.int64) for t in seed_to_team.values()}
    matchup_wins = Counter()

    for _ in range(n_sims):
        winners = {}
        for slot, strong, weak in slots:
            a = winners.get(strong, seed_to_team.get(strong))
            b = winners.get(weak, seed_to_team.get(weak))
            w = a if rng.random() < prob[(a, b)] else b
            winners[slot] = w
            matchup_wins[(slot, min(a, b), max(a, b), w)] += 1
            if slot.startswith("R"):
                reach[w][int(slot[1]) - 1] += 1
    return reach, matchup_wins


def most_likely_bracket(slots: list, seed_to_team: dict, matchup_wins: Counter,
                        prob: dict) -> pd.DataFrame:
    """Fill the bracket slot by slot, picking whichever team won that matchup more often.

    If a matchup never came up in the simulations, fall back to the model probability.
    """
    winners, rows = {}, []
    for slot, strong, weak in slots:
        a = winners.get(strong, seed_to_team.get(strong))
        b = winners.get(weak, seed_to_team.get(weak))
        lo, hi = min(a, b), max(a, b)
        wa, wb = matchup_wins[(slot, lo, hi, a)], matchup_wins[(slot, lo, hi, b)]
        if wa + wb == 0:
            wa, wb = prob[(a, b)], prob[(b, a)]
        winners[slot] = a if wa >= wb else b
        rows.append({"Slot": slot, "TeamA": a, "TeamB": b, "Winner": winners[slot],
                     "Round": int(slot[1]) if slot.startswith("R") else 0})
    return pd.DataFrame(rows)
