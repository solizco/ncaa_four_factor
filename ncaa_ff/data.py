"""Loading the Kaggle "March Machine Learning Mania" CSV files."""

from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

REQUIRED_FILES = [
    "MRegularSeasonDetailedResults.csv",
    "MNCAATourneyDetailedResults.csv",
    "MNCAATourneySeeds.csv",
    "MNCAATourneySlots.csv",
    "MTeams.csv",
]


def load(name: str, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Read one Kaggle CSV from the data directory with a helpful error."""
    path = Path(data_dir) / name
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Download the Kaggle March Machine Learning Mania "
            f"data and place the CSV files in {Path(data_dir).resolve()} "
            "(see data/README.md)."
        )
    return pd.read_csv(path)


def team_names(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """TeamID -> display name, from MTeams.csv."""
    teams = load("MTeams.csv", data_dir)
    return dict(zip(teams["TeamID"], teams["TeamName"]))


def seeds_for(season: int, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Tournament seeds for one season, with the numeric seed (W01 -> 1, Y11a -> 11)."""
    seeds = load("MNCAATourneySeeds.csv", data_dir)
    seeds = seeds[seeds["Season"] == season].copy()
    seeds["SeedNum"] = seeds["Seed"].str.extract(r"(\d+)")[0].astype(int)
    return seeds
