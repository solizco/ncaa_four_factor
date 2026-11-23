"""Exploratory data analysis figures for the Master Analytical Table."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import FEATURES

LABELS = {
    "DIFF_EFG": "Diff eFG%",
    "DIFF_TOV": "Diff TOV rate",
    "DIFF_ORB": "Diff ORB rate",
    "DIFF_FTR": "Diff FT rate",
    "Win_A": "Win_A",
}


def correlation_matrix(table: pd.DataFrame, out: Path = None):
    cols = FEATURES + ["Win_A"]
    corr = table[cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), [LABELS[c] for c in cols], rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), [LABELS[c] for c in cols])
    for i in range(len(cols)):
        for j in range(len(cols)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if abs(value) > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Correlation: Four Factor differentials and winning")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    return corr, fig


def separation_boxplots(table: pd.DataFrame, features=("DIFF_EFG", "DIFF_FTR"), out: Path = None):
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4.5))
    for ax, feat in zip(np.atleast_1d(axes), features):
        groups = [table.loc[table["Win_A"] == 0, feat], table.loc[table["Win_A"] == 1, feat]]
        ax.boxplot(groups, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#c6dbef"), medianprops=dict(color="#08306b", linewidth=2))
        ax.set_xticks([1, 2], ["Team A lost", "Team A won"])
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(LABELS[feat])
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Do the differentials separate winners from losers? (outliers hidden)")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    return fig


def tourney_seed_games(tourney_results: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    """Tournament games with each team's numeric seed attached."""
    seeds = seeds.copy()
    seeds["SeedNum"] = seeds["Seed"].str.extract(r"(\d+)")[0].astype(int)
    lookup = seeds.set_index(["Season", "TeamID"])["SeedNum"]
    games = tourney_results.copy()
    games["WinnerSeed"] = lookup.reindex(list(zip(games["Season"], games["WTeamID"]))).to_numpy()
    games["LoserSeed"] = lookup.reindex(list(zip(games["Season"], games["LTeamID"]))).to_numpy()
    games["SeedDiff"] = games["WinnerSeed"] - games["LoserSeed"]
    return games


def seed_heatmap(seed_games: pd.DataFrame, out: Path = None):
    counts = pd.crosstab(seed_games["LoserSeed"], seed_games["WinnerSeed"]).reindex(
        index=range(1, 17), columns=range(1, 17), fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.log1p(counts), cmap="magma", origin="upper")
    ax.set_xticks(range(16), range(1, 17))
    ax.set_yticks(range(16), range(1, 17))
    ax.set_xlabel("Winner seed")
    ax.set_ylabel("Loser seed")
    for i in range(16):
        for j in range(16):
            n = counts.iloc[i, j]
            if n:
                ax.text(j, i, n, ha="center", va="center", fontsize=7,
                        color="black" if n > 40 else "white")
    fig.colorbar(im, ax=ax, label="log(1 + games)")
    ax.set_title("Tournament results by seed (cells right of the diagonal are upsets)")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    return counts, fig


def biggest_upsets(seed_games: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Largest seed-differential upsets, with the winner-minus-loser Four Factors."""
    from .features import _side

    upsets = seed_games[seed_games["SeedDiff"] > 0].sort_values(
        ["SeedDiff", "Season"], ascending=[False, True]).head(n).copy()
    w = _side(upsets, "W", "L")
    l = _side(upsets, "L", "W")
    for key, col in zip(["EFG", "TOV_rate", "ORB_rate", "FTR"], FEATURES):
        upsets[col + " (W-L)"] = w[key] - l[key]
    return upsets
