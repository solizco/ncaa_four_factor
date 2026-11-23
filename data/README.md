# Data

The raw data comes from Kaggle's **March Machine Learning Mania** competition and is not included in this repository. To get it:

1. Sign in to Kaggle, open the competition's data page (for example, [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data)) and accept the competition rules.
2. Download the files and put these five in this `data/` folder:

| File | Used for |
|---|---|
| `MRegularSeasonDetailedResults.csv` | box scores for every regular-season game (2003 onward) |
| `MNCAATourneyDetailedResults.csv` | box scores for every NCAA tournament game |
| `MNCAATourneySeeds.csv` | each season's 68 tournament teams and seeds |
| `MNCAATourneySlots.csv` | bracket structure (which winners meet next) |
| `MTeams.csv` | TeamID → team name |

Running `python build_master_table.py` then writes `Master_Analytical_Table.csv` into this folder.

Each year's release adds the new season. The 2026 release contains the full 2025 tournament and the 2026 bracket, so it reproduces every result in this repo. With an older release, results for seasons it doesn't cover won't be available.
