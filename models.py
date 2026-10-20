"""Learned predictors, rule-based baselines and the physics shield."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from . import physics as ph
from .simulate import CATEGORICAL, MU_LOWER, NUMERIC_PHYSICS, NUMERIC_RAW, drcri, nominal_requirements

# ----------------------------------------------------------------- feature sets

def feature_frame(df: pd.DataFrame, distance: str = "track", physics: bool = True) -> pd.DataFrame:
    """Model inputs. Scenario names, identifiers and all hidden-truth columns are excluded."""
    X = df[CATEGORICAL + NUMERIC_RAW].copy()
    if distance == "euclid":
        X["track_distance_m"] = df["euclid_distance_m"].to_numpy()
        X = X.drop(columns=["occupancy_on_route", "route_conflict"])   # no route-table knowledge
    if physics:
        tmp = df.copy()
        if distance == "euclid":
            tmp["track_distance_m"] = df["euclid_distance_m"]
        d_avail, req = nominal_requirements(tmp)
        d_safe = req[2] + ph.MARGIN
        X["d_emg_nominal"] = req[2]
        X["d_svc_nominal"] = req[1]
        X["margin_emg_nominal"] = d_avail - d_safe
        idx, ttc = drcri(tmp, d_safe, d_avail)
        X["ttc_s"] = ttc
        X["drcri"] = idx
    return X


def _pre(X):
    cats = [c for c in CATEGORICAL if c in X.columns]
    nums = [c for c in X.columns if c not in cats]
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats),
                              ("num", StandardScaler(), nums)])


def make_model(name: str, X, seed: int):
    if name == "LR":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    elif name == "RF":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=seed)
    elif name == "HGB":
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                             l2_regularization=1.0, random_state=seed)
    elif name == "MLP":
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=400, early_stopping=True,
                            random_state=seed)
    else:
        raise ValueError(name)
    return Pipeline([("pre", _pre(X)), ("clf", clf)])


def fit(model, X, y):
    name = type(model.named_steps["clf"]).__name__
    if name in ("HistGradientBoostingClassifier", "MLPClassifier"):
        sw = compute_sample_weight("balanced", y)
        if name == "MLPClassifier":   # MLP has no sample_weight: rebalance by resampling
            rng = np.random.default_rng(0)
            p = sw / sw.sum(); idx = rng.choice(len(y), size=len(y), p=p)
            return model.fit(X.iloc[idx], np.asarray(y)[idx])
        return model.fit(X, y, clf__sample_weight=sw)
    return model.fit(X, y)


# ---------------------------------------------------------- non-learned baselines

def observed_conflict(df):
    return ((df["route_conflict"] == 1) | (df["occupancy_on_route"] == 1) | (df["obstruction_reported"] == 1)
            | df["signal_state"].isin(["red", "red_passed", "failed"])).to_numpy()


def fixed_threshold(df, bands=(2000.0, 1000.0, 400.0)):
    """Conventional fixed-separation warning rule (speed- and braking-agnostic)."""
    d = df["track_distance_m"].to_numpy()
    t = np.zeros(len(df), int)
    t[d < bands[0]] = 1; t[d < bands[1]] = 2; t[d < bands[2]] = 3
    t[~observed_conflict(df)] = 0
    return t


def nominal_physics(df):
    """Deterministic braking check on observed data with weather-nominal adhesion."""
    d_avail, req = nominal_requirements(df)
    return ph.required_tier(d_avail, req, conflict=observed_conflict(df))


def drcri_rule(df):
    idx = df["drcri"].to_numpy()
    return np.digitize(idx, [25, 50, 75])


def physics_shield(df, k_sigma=2.0):
    """Conservative (uncertainty-inflated) braking envelope used as a severity floor.

    * staleness correction:   d <- d - (v_close + k_sigma*sigma_v) * T_comm
    * position uncertainty:   d <- d - k_sigma * sigma_pos(reliability)
    * speed uncertainty:      v <- v + k_sigma * sigma_v(reliability)
    * adhesion:               weather-conditional lower bound mu_lo
    * braking performance:    rolling-stock factor lower bound K_rst = 0.90
    * moving leader:          absolute braking (leader treated as stationary)
    * multi-source conflict:  any route/occupancy/obstruction/signal indicator, or a
                              point-detection mismatch (distance to the adjacent occupied
                              section is then used), or written/invalid authority.
    """
    tmp = df.copy()
    mismatch = df["point_mismatch"].to_numpy() == 1
    tmp.loc[mismatch, "track_distance_m"] = np.minimum(df.loc[mismatch, "track_distance_m"],
                                                       df.loc[mismatch, "adjacent_distance_m"])
    d_avail, req = nominal_requirements(tmp, mu_table=MU_LOWER, stale_correct=True, k_sigma=k_sigma,
                                        k_rst=0.90, absolute_leader=True)
    possible = observed_conflict(df) | mismatch | df["authority"].isin(["written", "invalid"]).to_numpy()
    return ph.required_tier(d_avail, req, conflict=possible)


# -------------------------------------------------------------- cost-sensitive rule

COST = np.array([  # rows: true tier, cols: decided tier
    [0, 1, 3, 6],      # Low:      unnecessary interventions (service ~1, emergency ~3-6)
    [5, 0, 1, 3],      # Moderate
    [25, 8, 0, 1],     # High
    [100, 40, 10, 0],  # Critical: missed or late intervention
], float)


def bayes_cost_decision(proba, cost=COST):
    return np.argmin(proba @ cost, axis=1)


def shield_assumption_check(df, k_sigma=2.0):
    """Empirical check of Proposition 2 on simulated data (uses hidden truth; analysis only).

    Returns the number of encounters that satisfy the shield's stated assumptions and,
    among them, how many the shield under-classifies (should be zero).
    """
    rel = df["reliability"].to_numpy()
    sig_v = (1.0 + 3.0 * (1.0 - rel)); sig_pos = 5.0 + 120.0 * (1.0 - rel)
    mismatch = df["point_mismatch"].to_numpy() == 1
    d_used = np.where(mismatch, np.minimum(df["track_distance_m"], df["adjacent_distance_m"]), df["track_distance_m"])
    eps_eff = d_used - df["true_distance_m"].to_numpy() - df["true_closing_ms"].to_numpy() * df["comm_delay_s"].to_numpy()
    possible = observed_conflict(df) | mismatch | df["authority"].isin(["written", "invalid"]).to_numpy()
    ok = ((df["true_mu"].to_numpy() >= df["weather"].map(MU_LOWER).to_numpy())
          & (df["true_k_rst"].to_numpy() >= 0.90)
          & (eps_eff <= k_sigma * sig_pos)
          & (df["true_speed_a_kmh"].to_numpy() <= df["speed_a_kmh"].to_numpy() + k_sigma * sig_v)
          & (df["true_speed_b_kmh"].to_numpy() <= df["speed_b_kmh"].to_numpy() + k_sigma * sig_v)
          & (df["true_closing_ms"].to_numpy() * 3.6 <= df["closing_speed_kmh"].to_numpy() + k_sigma * sig_v)
          & (possible | (df["true_conflict"].to_numpy() == 0)))
    y = df["label"].to_numpy(); sh = physics_shield(df, k_sigma)
    return dict(n=int(len(df)), n_assumptions_hold=int(ok.sum()),
                under_when_hold=int((sh[ok] < y[ok]).sum()),
                under_when_violated=int((sh[~ok] < y[~ok]).sum()), n_violated=int((~ok).sum()))
