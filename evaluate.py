"""Safety-oriented evaluation metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

from .models import COST


def metrics(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    rec = recall_score(y, yhat, labels=[0, 1, 2, 3], average=None, zero_division=0)
    sev = y >= 2
    return {
        "accuracy": accuracy_score(y, yhat),
        "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "recall_low": rec[0], "recall_moderate": rec[1], "recall_high": rec[2], "recall_critical": rec[3],
        # P(decided < true | true >= Moderate): any late / insufficient intervention
        "under_rate": float(np.mean(yhat[y >= 1] < y[y >= 1])),
        # P(decided <= Moderate | true >= High): emergency needed but not commanded
        "severe_miss": float(np.mean(yhat[sev] <= 1)),
        # P(decided < Critical | true = Critical)
        "critical_miss": float(np.mean(yhat[y == 3] < 3)),
        # P(decided >= High | true = Low): unnecessary emergency intervention
        "false_emergency": float(np.mean(yhat[y == 0] >= 2)),
        "cost": float(np.mean(COST[y, yhat])),
    }


def conditional_coverage(y, yhat):
    """P(decision >= Y | Y = k) for each k."""
    y = np.asarray(y); yhat = np.asarray(yhat)
    return {k: float(np.mean(yhat[y == k] >= k)) for k in range(4)}
