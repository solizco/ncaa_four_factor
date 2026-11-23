"""Logistic regression on the four differentials."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from . import FEATURES


@dataclass
class TrainedModel:
    pipeline: Pipeline
    metrics: dict
    coefficients: pd.Series

    def win_probability(self, features: np.ndarray) -> np.ndarray:
        """P(Team A wins) for rows of [DIFF_EFG, DIFF_TOV, DIFF_ORB, DIFF_FTR]."""
        return self.pipeline.predict_proba(np.atleast_2d(features))[:, 1]


def train(table: pd.DataFrame, seed: int = 42) -> TrainedModel:
    X = table[FEATURES].to_numpy()
    y = table["Win_A"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    pipeline.fit(X_tr, y_tr)

    p = pipeline.predict_proba(X_te)[:, 1]
    metrics = {
        "train_games": len(X_tr),
        "test_games": len(X_te),
        "accuracy": accuracy_score(y_te, p > 0.5),
        "log_loss": log_loss(y_te, p),
        "roc_auc": roc_auc_score(y_te, p),
    }
    # Standardized coefficients: change in log-odds per 1 SD of each differential
    coefficients = pd.Series(pipeline[-1].coef_[0], index=FEATURES)
    return TrainedModel(pipeline, metrics, coefficients)
