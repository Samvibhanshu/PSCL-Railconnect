"""Class-conditional (Mondrian) split-conformal severity escalation.

For each severity class k with miscoverage budget alpha_k, the calibration
non-conformity scores s_i = 1 - p_hat(y_i | x_i) of the calibration points with
y_i = k give the threshold

    q_k = the ceil((n_k + 1)(1 - alpha_k))-th smallest score among class-k points.

The prediction set is C(x) = {k : 1 - p_hat(k|x) <= q_k}. The escalated decision
is the most severe tier in C(x) (or the arg-max class if C(x) is empty).

Guarantee (exchangeable calibration and test data): for every class k,
    P( decision(X) >= Y | Y = k ) >= P( Y in C(X) | Y = k ) >= 1 - alpha_k,
i.e. the per-class rate of *under-classification* is at most alpha_k.
"""
from __future__ import annotations

import numpy as np


class MondrianEscalation:
    def __init__(self, alphas: dict):
        self.alphas = alphas          # {class_index: alpha}; classes absent are unconstrained
        self.q = {}

    def fit(self, proba_cal: np.ndarray, y_cal: np.ndarray):
        y_cal = np.asarray(y_cal)
        for k, a in self.alphas.items():
            s = np.sort(1.0 - proba_cal[y_cal == k, k])
            n = len(s)
            r = int(np.ceil((n + 1) * (1 - a)))
            self.q[k] = np.inf if r > n else s[r - 1]
        return self

    def sets(self, proba: np.ndarray) -> np.ndarray:
        member = np.zeros_like(proba, dtype=bool)
        for k, q in self.q.items():
            member[:, k] = (1.0 - proba[:, k]) <= q
        member[np.arange(len(proba)), proba.argmax(1)] = True
        return member

    def decide(self, proba: np.ndarray) -> np.ndarray:
        m = self.sets(proba)
        K = proba.shape[1]
        return (K - 1) - np.argmax(m[:, ::-1], axis=1)   # most severe member
