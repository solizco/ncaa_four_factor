"""Build Master_Analytical_Table.csv: one row per game with Four Factor differentials.

Includes every regular-season game from 2003 through --season, plus every
tournament game from seasons *before* --season. Stopping before that season's
tournament means a model trained on this table never sees the games it is
later asked to simulate. With the default --season 2025 this produces the
120,264-game table used in the EDA report.

Usage:
    python build_master_table.py                 # data/ -> data/Master_Analytical_Table.csv
    python build_master_table.py --season 2026
"""

import argparse
from pathlib import Path

import pandas as pd

from ncaa_ff import data
from ncaa_ff.features import game_level_table


def build(season: int, data_dir: Path) -> pd.DataFrame:
    reg = data.load("MRegularSeasonDetailedResults.csv", data_dir)
    tourney = data.load("MNCAATourneyDetailedResults.csv", data_dir)

    games = pd.concat([
        reg[reg["Season"] <= season].assign(GameType="Regular"),
        tourney[tourney["Season"] < season].assign(GameType="Tourney"),
    ], ignore_index=True)
    return game_level_table(games)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=2025,
                        help="last regular season to include; tournaments stop the year before")
    parser.add_argument("--data-dir", type=Path, default=data.DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=None,
                        help="output CSV (default: <data-dir>/Master_Analytical_Table.csv)")
    args = parser.parse_args()

    table = build(args.season, args.data_dir)
    out = args.out or args.data_dir / "Master_Analytical_Table.csv"
    table.to_csv(out, index=False)

    print(f"Games: {len(table):,}  (seasons {table['Season'].min()}-{table['Season'].max()}, "
          f"{(table['GameType'] == 'Tourney').sum():,} tournament)")
    print(f"Win_A balance: {table['Win_A'].mean():.1%} wins")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
