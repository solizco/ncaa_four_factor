"""Reproduce the EDA report figures and numbers.

Usage:
    python eda.py            # reads data/Master_Analytical_Table.csv, writes figures/
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from ncaa_ff import FEATURES, data, eda


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=data.DEFAULT_DATA_DIR)
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.figures.mkdir(exist_ok=True)

    table = pd.read_csv(args.data_dir / "Master_Analytical_Table.csv")
    last = table["Season"].max()
    print(f"Games: {len(table):,}  |  Win_A = 1 in {table['Win_A'].mean():.1%}")
    print(table[FEATURES].describe().loc[["mean", "std"]].round(4).to_string(), "\n")

    corr, _ = eda.correlation_matrix(table, args.figures / "correlation_matrix.png")
    print("Correlation with Win_A:")
    print(corr["Win_A"].drop("Win_A").round(3).to_string(), "\n")

    eda.separation_boxplots(table, out=args.figures / "separation_boxplots.png")

    tourney = data.load("MNCAATourneyDetailedResults.csv", args.data_dir)
    tourney = tourney[tourney["Season"] < last]
    seeds = data.load("MNCAATourneySeeds.csv", args.data_dir)
    seed_games = eda.tourney_seed_games(tourney, seeds)
    eda.seed_heatmap(seed_games, out=args.figures / "seed_heatmap.png")
    print(f"Tournament games {seed_games['Season'].min()}-{seed_games['Season'].max()}: "
          f"{len(seed_games):,}, upsets (winner seed numerically higher): "
          f"{(seed_games['SeedDiff'] > 0).mean():.1%}\n")

    names = data.team_names(args.data_dir)
    upsets = eda.biggest_upsets(seed_games)
    upsets["Winner"] = upsets["WTeamID"].map(names)
    upsets["Loser"] = upsets["LTeamID"].map(names)
    cols = ["Season", "Winner", "WinnerSeed", "Loser", "LoserSeed"] + \
           [c for c in upsets.columns if c.endswith("(W-L)")]
    print("Biggest upsets (Four Factor differentials, winner minus loser):")
    print(upsets[cols].round(3).to_string(index=False))

    print(f"\nFigures saved to {args.figures}/")


if __name__ == "__main__":
    main()
