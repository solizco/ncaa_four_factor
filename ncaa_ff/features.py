"""Dean Oliver's Four Factors, at the game level and as season profiles.

    eFG%      = (FGM + 0.5 * FGM3) / FGA
    TOV rate  = 100 * TO / Possessions,   Possessions = FGA - OR + TO + 0.475 * FTA
    ORB rate  = OR / (OR + opponent DR)
    FT rate   = FTA / FGA
"""

import numpy as np
import pandas as pd

from . import FEATURES


def _factors(fgm, fga, fgm3, fta, orb, to, opp_drb):
    possessions = fga - orb + to + 0.475 * fta
    return {
        "EFG": (fgm + 0.5 * fgm3) / fga,
        "TOV_rate": 100 * to / possessions,
        "ORB_rate": orb / (orb + opp_drb),
        "FTR": fta / fga,
    }


def _side(games: pd.DataFrame, me: str, opp: str) -> dict:
    """Four Factors for one side ('W' or 'L') of each game in a Kaggle results table."""
    return _factors(
        games[f"{me}FGM"], games[f"{me}FGA"], games[f"{me}FGM3"], games[f"{me}FTA"],
        games[f"{me}OR"], games[f"{me}TO"], games[f"{opp}DR"],
    )


def game_level_table(games: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """One row per game: Team A minus Team B Four Factor differentials and Win_A.

    The winner is randomly assigned to Team A or Team B (fixed seed), so Win_A is
    balanced and the differentials are centered on zero.
    """
    w = _side(games, "W", "L")
    l = _side(games, "L", "W")

    rng = np.random.default_rng(seed)
    winner_is_a = rng.random(len(games)) < 0.5
    sign = np.where(winner_is_a, 1.0, -1.0)

    out = pd.DataFrame({
        "Season": games["Season"].to_numpy(),
        "DayNum": games["DayNum"].to_numpy(),
        "GameType": games["GameType"].to_numpy(),
        "TeamA": np.where(winner_is_a, games["WTeamID"], games["LTeamID"]),
        "TeamB": np.where(winner_is_a, games["LTeamID"], games["WTeamID"]),
        "Win_A": winner_is_a.astype(int),
    })
    for key, col in zip(["EFG", "TOV_rate", "ORB_rate", "FTR"], FEATURES):
        out[col] = sign * (w[key] - l[key]).to_numpy()
    return out


def season_profiles(regular_season: pd.DataFrame, season: int) -> pd.DataFrame:
    """Season-total Four Factors for every team, indexed by TeamID."""
    games = regular_season[regular_season["Season"] == season]
    cols = ["FGM", "FGA", "FGM3", "FTA", "OR", "DR", "TO"]

    def side(me, opp):
        part = games[[f"{me}TeamID"] + [f"{me}{c}" for c in cols] + [f"{opp}DR"]].copy()
        part.columns = ["TeamID"] + cols + ["Opp_DR"]
        return part

    totals = pd.concat([side("W", "L"), side("L", "W")]).groupby("TeamID").sum()
    factors = _factors(totals["FGM"], totals["FGA"], totals["FGM3"], totals["FTA"],
                       totals["OR"], totals["TO"], totals["Opp_DR"])
    return pd.DataFrame(factors)


def matchup_features(profile_a: pd.Series, profile_b: pd.Series) -> list:
    """Differential feature vector (A - B) in the same order as FEATURES."""
    return [
        profile_a["EFG"] - profile_b["EFG"],
        profile_a["TOV_rate"] - profile_b["TOV_rate"],
        profile_a["ORB_rate"] - profile_b["ORB_rate"],
        profile_a["FTR"] - profile_b["FTR"],
    ]
