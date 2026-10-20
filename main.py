"""Reproduce every number, table and figure in the manuscript.

    python main.py                 # full run (10 seeds), ~10 min on 2 CPU cores
    python main.py --quick         # 2 seeds, smaller data; smoke test (~3 min)

Outputs go to ../results (Code Ocean: /results).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split

from railsafe import figures
from railsafe.conformal import MondrianEscalation
from railsafe.evaluate import conditional_coverage, metrics
from railsafe.models import (bayes_cost_decision, drcri_rule, feature_frame, fit, fixed_threshold, make_model,
                             nominal_physics, physics_shield, shield_assumption_check)
from railsafe.simulate import EncounterSimulator

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHAS = {1: 0.10, 2: 0.05, 3: 0.02}            # per-class under-classification budgets
CANONICAL_SEED = 2026


def split(y, seed):
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    ca, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    return tr, ca, te


def run_seed(seed, n, out, keep=None):
    rng_seed = 1000 + seed
    df = EncounterSimulator(rng_seed).generate(n)
    y = df["label"].to_numpy()
    tr, ca, te = split(y, seed)
    dte = df.iloc[te]
    rows, pred = [], {}

    def add(name, yhat, family):
        m = metrics(y[te], yhat); m.update(method=name, family=family, seed=seed); rows.append(m)
        pred[name] = yhat

    add("Fixed-threshold", fixed_threshold(dte), "baseline")
    add("DRCRI index", drcri_rule(dte), "baseline")
    add("Nominal physics", nominal_physics(dte), "baseline")
    shield = physics_shield(dte)
    add("Physics shield", shield, "baseline")
    check = shield_assumption_check(dte); check["seed"] = seed

    X = feature_frame(df)
    probs = {}
    for name in ["LR", "RF", "HGB", "MLP"]:
        t0 = time.time()
        model = fit(make_model(name, X, seed), X.iloc[tr], y[tr])
        pc, pt = model.predict_proba(X.iloc[ca]), model.predict_proba(X.iloc[te])
        probs[name] = (model, pc, pt)
        add(name, pt.argmax(1), "ml")
        cp = MondrianEscalation(ALPHAS).fit(pc, y[ca])
        add(f"{name}+CP", cp.decide(pt), "ml+cp")
        add(f"{name}+Shield", np.maximum(pt.argmax(1), shield), "ml+shield")
        add(f"{name}+CS", bayes_cost_decision(pt), "ml+cost")
        add(f"PSCL({name})", np.maximum(cp.decide(pt), shield), "proposed")
        print(f"  seed {seed} {name}: {time.time() - t0:.1f}s", flush=True)

    # ------------------------------------------------ ablations (HGB backbone)
    abl = []
    variants = {
        "Full (track distance + physics features)": feature_frame(df),
        "No physics-informed features": feature_frame(df, physics=False),
        "Euclidean distance (no route table)": feature_frame(df, distance="euclid"),
        "No reliability features": feature_frame(df).drop(columns=["comm_delay_s", "reliability"]),
        "No multi-source features": feature_frame(df).drop(columns=["point_mismatch", "adjacent_occupancy",
                                                                    "adjacent_distance_m"]),
    }
    for vname, Xv in variants.items():
        mdl = probs["HGB"][0] if vname.startswith("Full") else fit(make_model("HGB", Xv, seed), Xv.iloc[tr], y[tr])
        pc, pt = mdl.predict_proba(Xv.iloc[ca]), mdl.predict_proba(Xv.iloc[te])
        for tag, yhat in [("HGB", pt.argmax(1)),
                          ("PSCL", np.maximum(MondrianEscalation(ALPHAS).fit(pc, y[ca]).decide(pt), shield))]:
            m = metrics(y[te], yhat); m.update(variant=vname, model=tag, seed=seed); abl.append(m)
        if vname.startswith("Euclid"):
            pred["HGB-euclid"] = pt.argmax(1)

    # ------------------------------------------------ conformal alpha sweep (HGB)
    _, pc, pt = probs["HGB"]
    sweep = []
    for a in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
        cp = MondrianEscalation({1: a, 2: a, 3: a}).fit(pc, y[ca])
        d = cp.decide(pt); cov = conditional_coverage(y[te], d); m = metrics(y[te], d)
        sweep.append(dict(seed=seed, alpha=a, cov_moderate=cov[1], cov_high=cov[2], cov_critical=cov[3],
                          set_size=float(cp.sets(pt).sum(1).mean()), false_emergency=m["false_emergency"],
                          accuracy=m["accuracy"]))

    # ------------------------------------------------ distribution shift (HGB)
    shift = []
    dsh = EncounterSimulator(5000 + seed, low_adhesion_boost=0.25, degraded_boost=0.25).generate(n // 3)
    ysh = dsh["label"].to_numpy(); Xsh = feature_frame(dsh)
    model = probs["HGB"][0]; ps = model.predict_proba(Xsh)
    sh_shift = physics_shield(dsh)
    cp = MondrianEscalation(ALPHAS).fit(pc, y[ca])
    # re-calibration with a small labelled sample from the shifted regime (first 1,000 records)
    ncal = min(1000, len(dsh) // 4)
    cp_re = MondrianEscalation(ALPHAS).fit(ps[:ncal], ysh[:ncal])
    ev = slice(ncal, None)
    for name, yhat in [("Nominal physics", nominal_physics(dsh)), ("Physics shield", sh_shift),
                       ("HGB", ps.argmax(1)), ("HGB+CP", cp.decide(ps)),
                       ("HGB+CP (recalibrated)", cp_re.decide(ps)),
                       ("PSCL(HGB)", np.maximum(cp.decide(ps), sh_shift)),
                       ("PSCL(HGB) recalibrated", np.maximum(cp_re.decide(ps), sh_shift))]:
        m = metrics(ysh[ev], np.asarray(yhat)[ev]); m.update(method=name, seed=seed, n_eval=int(len(ysh) - ncal))
        cov = conditional_coverage(ysh[ev], np.asarray(yhat)[ev]); m.update(cov_critical=cov[3], cov_high=cov[2])
        shift.append(m)

    # ------------------------------------------------ scenario-wise (pooled later)
    scen = pd.DataFrame({"seed": seed, "scenario": dte["scenario"].to_numpy(), "y": y[te],
                         "Nominal physics": pred["Nominal physics"], "HGB": pred["HGB"],
                         "HGB-euclid": pred["HGB-euclid"], "PSCL(HGB)": pred["PSCL(HGB)"]})

    extra = {}
    if keep is not None:   # seed-0 artefacts for figures / importance / latency / Protocol A
        model = probs["HGB"][0]
        sub = te[:3000]
        imp = permutation_importance(model, X.iloc[sub], y[sub], n_repeats=5, random_state=0,
                                     scoring="f1_macro", n_jobs=1)
        extra["importance"] = pd.DataFrame({"feature": X.columns, "mean": imp.importances_mean,
                                            "std": imp.importances_std}).sort_values("mean", ascending=False)
        x1 = X.iloc[te[:1]]
        t0 = time.perf_counter(); [model.predict_proba(x1) for _ in range(200)]
        lat1 = (time.perf_counter() - t0) / 200 * 1e3
        t0 = time.perf_counter(); model.predict_proba(X.iloc[te]); latb = (time.perf_counter() - t0) / len(te) * 1e3
        t0 = time.perf_counter(); physics_shield(dte); lats = (time.perf_counter() - t0) / len(te) * 1e3
        extra["latency_ms"] = dict(single_record_hgb=lat1, batched_per_record_hgb=latb, shield_per_record=lats)
        extra["cm_hgb"] = confusion_matrix(y[te], pred["HGB"], labels=[0, 1, 2, 3])
        extra["cm_pscl"] = confusion_matrix(y[te], pred["PSCL(HGB)"], labels=[0, 1, 2, 3])
        # Protocol A: index-derived labels (the patent prototype's protocol)
        yA = drcri_rule(df)
        Xr = feature_frame(df, physics=False)
        rfA = fit(make_model("RF", Xr, seed), Xr.iloc[tr], yA[tr])
        mA = metrics(yA[te], rfA.predict(Xr.iloc[te]))
        extra["protocol_a"] = dict(
            rf_accuracy_vs_index_labels=mA["accuracy"], rf_macro_f1_vs_index_labels=mA["macro_f1"],
            index_label_accuracy_vs_physics=float(np.mean(yA == y)),
            index_label_kappa_vs_physics=float(cohen_kappa_score(yA, y)),
            index_label_critical_recall_vs_physics=float(np.mean(yA[y == 3] == 3)),
            index_class_counts=np.bincount(yA, minlength=4).tolist(),
            cm_index_vs_physics=confusion_matrix(y, yA, labels=[0, 1, 2, 3]).tolist())
        extra["df"] = df
    extra_check = check
    return pd.DataFrame(rows), pd.DataFrame(abl), pd.DataFrame(sweep), pd.DataFrame(shift), scen, extra, extra_check


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--results", default=os.path.normpath(os.path.join(HERE, "..", "results")))
    ap.add_argument("--data", default=os.path.normpath(os.path.join(HERE, "..", "data")))
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.n = 2, 9000
    os.makedirs(a.results, exist_ok=True)
    t_start = time.time()

    # -------------------------------------------- canonical released dataset
    canon = EncounterSimulator(CANONICAL_SEED).generate(a.n)
    canon_path = os.path.join(a.results, f"railconflict_encounters_seed{CANONICAL_SEED}_n{a.n}.csv")
    canon.to_csv(canon_path, index=False, float_format="%.6g")
    shipped = os.path.join(a.data, os.path.basename(canon_path))
    data_check = {"generated_sha256": sha256(canon_path)}
    if os.path.exists(shipped):
        data_check["shipped_sha256"] = sha256(shipped)
        data_check["identical"] = data_check["shipped_sha256"] == data_check["generated_sha256"]

    R, A, S, H, SC, extra, CHK = [], [], [], [], [], {}, []
    for s in range(a.seeds):
        print(f"seed {s}", flush=True)
        r, ab, sw, sh, sc, ex, chk = run_seed(s, a.n, a.results, keep=(s == 0) or None)
        R.append(r); A.append(ab); S.append(sw); H.append(sh); SC.append(sc); CHK.append(chk)
        if ex:
            extra = ex
    R, A, S, H, SC = map(pd.concat, (R, A, S, H, SC))
    for name, d in [("main_per_seed", R), ("ablation_per_seed", A), ("conformal_sweep_per_seed", S),
                    ("shift_per_seed", H), ("scenario_predictions", SC)]:
        d.to_csv(os.path.join(a.results, f"{name}.csv"), index=False)

    cols = ["accuracy", "macro_f1", "recall_high", "recall_critical", "under_rate", "severe_miss",
            "critical_miss", "false_emergency", "cost"]
    summ = R.groupby(["method", "family"], sort=False)[cols].agg(["mean", "std"])
    summ.to_csv(os.path.join(a.results, "table_main.csv"))
    A.groupby(["variant", "model"], sort=False)[cols].agg(["mean", "std"]).to_csv(os.path.join(a.results, "table_ablation.csv"))
    S.groupby("alpha").mean(numeric_only=True).to_csv(os.path.join(a.results, "table_conformal_sweep.csv"))
    H.groupby("method", sort=False)[cols + ["cov_critical", "cov_high"]].agg(["mean", "std"]).to_csv(
        os.path.join(a.results, "table_shift.csv"))

    rows = []
    for (sc_name), g in SC.groupby("scenario"):
        sev = g["y"] >= 2
        row = {"scenario": sc_name, "n_test_pooled": len(g), "share_high_or_critical": float(sev.mean())}
        for m in ["Nominal physics", "HGB", "HGB-euclid", "PSCL(HGB)"]:
            row[f"{m} accuracy"] = float((g[m] == g["y"]).mean())
            row[f"{m} severe_miss"] = float((g.loc[sev, m] <= 1).mean()) if sev.any() else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(a.results, "table_scenarios.csv"), index=False)

    df0 = extra["df"]
    pd.crosstab(df0["scenario"], df0["label"]).to_csv(os.path.join(a.results, "table_label_by_scenario_seed0.csv"))
    extra["importance"].to_csv(os.path.join(a.results, "table_importance.csv"), index=False)

    summary = dict(
        seeds=a.seeds, n_per_seed=a.n, alphas=ALPHAS, canonical_dataset=os.path.basename(canon_path),
        canonical_class_counts=np.bincount(canon["label"], minlength=4).tolist(),
        data_check=data_check,
        shield_proposition2_check={k: int(sum(c[k] for c in CHK)) for k in
                                   ("n", "n_assumptions_hold", "under_when_hold", "n_violated", "under_when_violated")},
        latency_ms=extra["latency_ms"], protocol_a=extra["protocol_a"],
        cm_hgb_seed0=extra["cm_hgb"].tolist(), cm_pscl_seed0=extra["cm_pscl"].tolist(),
        environment=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                         sklearn=sklearn.__version__, platform=platform.platform()),
        runtime_s=None,
    )
    figures.make_all(a.results, R, A, S, H, SC, extra)
    summary["runtime_s"] = round(time.time() - t_start, 1)
    with open(os.path.join(a.results, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(summ.xs("mean", axis=1, level=1).round(3).to_string())
    print(json.dumps(summary["protocol_a"], indent=1))
    print(f"done in {summary['runtime_s']} s")


if __name__ == "__main__":
    main()
