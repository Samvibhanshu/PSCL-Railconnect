"""Source-informed encounter simulator with hidden true state and degraded observations.

Every record is a *train-pair encounter* on the graph network. The simulator
keeps two views of each encounter:

  * the hidden TRUE state (true track distance, true wheel-rail adhesion,
    true route set by the points, true presence of conflicts), from which the
    ground-truth label -- the minimum intervention tier that physically avoids
    the conflict -- is computed with the train-dynamics model; and
  * the OBSERVED state available to a wayside/onboard safety layer (stale and
    noisy positions, weather-based adhesion estimate, interlocking-reported
    route, possibly failed track circuits), from which all predictors work.

Twelve scenario families are generated. Three reproduce failure mechanisms of
recent accidents: interlocking/route-state inconsistency (Bahanaga Bazar,
Balasore, 2023), degraded-mode written authority after signal failure
(Rangapani, Kanchanjunga Express, 2024) and low-adhesion overrun into a
single-line section (Talerddig, 2024).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import physics as ph
from .network import RouteGeometry, build_network, euclidean, route_table

HORIZON = 8000.0  # m: look-ahead horizon used when no conflict point is known

SCENARIOS = {
    "safe_parallel": 0.13, "rear_end": 0.12, "stationary_train": 0.08,
    "occupied_entry": 0.07, "head_on": 0.10, "junction": 0.11, "spad": 0.09,
    "obstruction": 0.07, "data_fault": 0.05, "interlocking_inconsistency": 0.06,
    "degraded_authority": 0.06, "low_adhesion_overrun": 0.06,
}

WEATHER = ["dry", "rain", "fog", "leaf_frost"]
MU_RANGE = {"dry": (0.15, 0.30), "rain": (0.08, 0.16), "fog": (0.10, 0.20), "leaf_frost": (0.03, 0.09)}
MU_NOMINAL = {"dry": 0.20, "rain": 0.11, "fog": 0.14, "leaf_frost": 0.06}
MU_LOWER = {"dry": 0.10, "rain": 0.07, "fog": 0.08, "leaf_frost": 0.03}

CATEGORICAL = ["route_type", "direction", "train_type", "weather", "signal_state", "authority"]
NUMERIC_RAW = [
    "speed_a_kmh", "speed_b_kmh", "accel_a", "closing_speed_kmh", "track_distance_m",
    "mass_t", "length_m", "a_emg_nominal", "a_serv_nominal", "brake_buildup_s",
    "gradient", "curvature", "adhesion_estimate", "occupancy_on_route", "route_conflict",
    "obstruction_reported", "point_mismatch", "adjacent_occupancy", "adjacent_distance_m",
    "comm_delay_s", "reliability",
]
NUMERIC_PHYSICS = ["d_emg_nominal", "d_svc_nominal", "margin_emg_nominal", "ttc_s", "drcri"]


class EncounterSimulator:
    def __init__(self, seed: int = 2026, low_adhesion_boost: float = 0.0, degraded_boost: float = 0.0):
        self.rng = np.random.default_rng(seed)
        self.G = build_network()
        self.routes = {k: RouteGeometry(self.G, r) for k, r in route_table().items()}
        self.low_adhesion_boost = low_adhesion_boost   # used for the distribution-shift study
        self.degraded_boost = degraded_boost

    # ------------------------------------------------------------------ helpers
    def _train(self):
        r = self.rng
        if r.random() < 0.7:
            return dict(train_type="passenger", mass_t=r.uniform(250, 750), length_m=r.uniform(100, 400),
                        a_emg_nominal=r.uniform(0.9, 1.3), brake_buildup_s=r.uniform(2.0, 3.5),
                        t_sb=2.5, vmax=130.0)
        return dict(train_type="freight", mass_t=r.uniform(1500, 5000), length_m=r.uniform(400, 750),
                    a_emg_nominal=r.uniform(0.40, 0.70), brake_buildup_s=r.uniform(4.0, 7.0),
                    t_sb=5.0, vmax=100.0)

    def _weather(self, forced=None):
        r = self.rng
        if forced is None:
            p = np.array([0.55, 0.25, 0.08, 0.12])
            p[3] += self.low_adhesion_boost; p /= p.sum()
            w = WEATHER[r.choice(4, p=p)]
        else:
            w = forced
        lo, hi = MU_RANGE[w]
        mu = r.uniform(lo, hi)
        if w == "dry" and r.random() < 0.03 + 0.5 * self.low_adhesion_boost:   # unseen contamination
            mu = r.uniform(0.03, 0.08)
        return w, mu

    def _comms(self, degraded):
        r = self.rng
        if degraded:
            return r.uniform(0.30, 0.70), r.uniform(2.0, 8.0)
        return r.uniform(0.85, 1.00), r.uniform(0.3, 1.5)

    def _gap(self, lo=60.0, hi=7000.0):
        """Mixture: log-uniform tail plus uniform near-boundary band (rare-event emphasis)."""
        if self.rng.random() < 0.35:
            return float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))
        return float(self.rng.uniform(max(lo, 100.0), min(hi, 2200.0)))

    # --------------------------------------------------------------- scenarios
    def _one(self, scen):
        r = self.rng
        A, B = self._train(), self._train()
        weather, mu = self._weather("leaf_frost" if (scen == "low_adhesion_overrun" and r.random() < 0.7) else
                                    ("rain" if scen == "low_adhesion_overrun" else None))
        if scen == "low_adhesion_overrun":
            mu = r.uniform(0.02, 0.06)
        degraded = (scen == "data_fault") or (r.random() < 0.10 + self.degraded_boost)
        rel, t_comm = self._comms(degraded)
        base = scen
        if scen == "data_fault":
            base = r.choice(["rear_end", "stationary_train", "junction", "safe_parallel"])

        sig, auth = "green", "valid"
        conflict = True; direction = "same"; vB = 0.0; d_b_stop_true = 0.0; d_b_stop_nom = 0.0
        opposing = False; route_conf_true = False; occ_on_route = 0; adj_occ = int(r.random() < 0.25)
        adj_dist = self._gap(200, 6000); mismatch = 0; obstr = 0
        believed_unknown = False

        if base == "safe_parallel":
            ra = self.routes["UP_MAIN"]; rb = self.routes["DN_MAIN"]
            sA = r.uniform(500, 19000); xA = ra.xy(sA)
            sB = np.clip(rb.length - sA + r.normal(0, 600), 100, rb.length - 100)
            xyB = rb.xy(sB); conflict = False; direction = "opposing_parallel"
            vA = r.uniform(0.3, 1.0) * A["vmax"]; vB = r.uniform(0.3, 1.0) * B["vmax"]
            d_true = HORIZON; sig = r.choice(["green", "yellow"], p=[0.8, 0.2])
        elif base in ("rear_end", "stationary_train", "degraded_authority"):
            ra = self.routes["UP_MAIN"]
            sBrear = r.uniform(2500, 19500); d_true = min(self._gap(), sBrear - 50)
            sA = sBrear - d_true; xA = ra.xy(sA); xyB = ra.xy(sBrear)
            vA = r.uniform(0.25, 1.0) * A["vmax"]
            vB = 0.0 if base != "rear_end" else r.uniform(0, 1.1) * vA
            if base == "degraded_authority":
                vA = r.uniform(10, 70); vB = r.choice([0.0, r.uniform(0, 15)])
                sig, auth = "failed", "written"
            occ_on_route = int(r.random() < (0.5 if base == "degraded_authority" else 0.98)) if d_true < 3000 else 0
            conflict = vA > vB + 0.5
            route_conf_true = conflict
        elif base == "occupied_entry":
            key = r.choice(["UP_LOOP1", "UP_LOOP2"]); ra = self.routes[key]
            loop_w = ra.node_chainage["L1w" if key == "UP_LOOP1" else "L2w"]
            sBrear = loop_w + r.uniform(50, 300); d_true = min(self._gap(), sBrear - 50)
            sA = sBrear - d_true; xA = ra.xy(sA); xyB = ra.xy(sBrear)
            vA = r.uniform(0.2, 0.8) * A["vmax"]; vB = 0.0
            occ_on_route = int(r.random() < 0.98); route_conf_true = True
        elif base in ("head_on", "low_adhesion_overrun"):
            ra = self.routes["UP_TO_BRANCH"]; opposing = True; direction = "opposing"
            pw = ra.node_chainage["Pw"]
            cpt = pw + r.uniform(0, 800) if base == "head_on" else pw + 50
            d_true = min(self._gap(80, 6000), cpt - 200); sA = cpt - d_true; xA = ra.xy(sA)
            vA = r.uniform(0.3, 1.0) * min(A["vmax"], 90)
            vB = r.choice([0.0, r.uniform(0.2, 1.0) * 90])
            xyB = ra.xy(min(cpt + 30, ra.length)); route_conf_true = True
            sig = "red" if r.random() < 0.6 else "yellow"; auth = "limited"
        elif base == "junction":
            ra = self.routes["UP_TO_BRANCH"]; direction = "crossing"
            dj = ra.node_chainage["DJ"]; d_true = min(self._gap(), dj - 100); sA = dj - d_true
            xA = ra.xy(sA); rb = self.routes["DN_MAIN"]; dB = self._gap(100, 6000)
            sB = rb.node_chainage["DJ"] - dB; xyB = rb.xy(max(sB, 0))
            vA = r.uniform(0.3, 1.0) * min(A["vmax"], 110); vB = r.uniform(0.3, 1.0) * B["vmax"]
            tA = d_true / (vA / 3.6); tB = dB / (vB / 3.6)
            clearA = (A["length_m"] + 100) / (vA / 3.6); clearB = (B["length_m"] + 100) / (vB / 3.6)
            conflict = (tA < tB + clearB + 30) and (tB < tA + clearA + 30)
            route_conf_true = conflict
            sig = r.choice(["red", "yellow"], p=[0.7, 0.3]) if conflict else r.choice(["green", "yellow", "red"], p=[0.5, 0.3, 0.2])
            auth = "limited" if sig != "green" else "valid"
        elif base == "spad":
            ra = self.routes["UP_MAIN"]; jj = ra.node_chainage["J"]
            d_true = self._gap(40, 2500); sA = jj - d_true; xA = ra.xy(sA); xyB = ra.xy(jj)
            vA = r.uniform(0.3, 1.0) * A["vmax"]; direction = "crossing"
            passed = d_true < 250
            sig = "red_passed" if passed else "red"; auth = "invalid" if passed else "limited"
            conflict = r.random() < 0.7; route_conf_true = conflict
        elif base == "obstruction":
            ra = self.routes[r.choice(["UP_MAIN", "UP_TO_BRANCH"])]
            d_true = self._gap(); sA = r.uniform(200, ra.length - d_true - 10) if ra.length - d_true > 300 else 100
            xA = ra.xy(sA); xyB = ra.xy(min(sA + d_true, ra.length)); vA = r.uniform(0.3, 1.0) * A["vmax"]
            obstr = 1; conflict = r.random() < 0.75; direction = "none"
        elif base == "interlocking_inconsistency":
            key = r.choice(["UP_LOOP1", "UP_LOOP2"]); ra = self.routes[key]   # TRUE route (points to loop)
            loop_w = ra.node_chainage["L1w" if key == "UP_LOOP1" else "L2w"]
            sBrear = loop_w + r.uniform(50, 300); d_true = min(self._gap(), sBrear - 50)
            sA = sBrear - d_true; xA = ra.xy(sA); xyB = ra.xy(sBrear)
            vA = r.uniform(0.5, 1.0) * A["vmax"]; vB = 0.0
            sig, auth = "green", "valid"                 # interlocking believes main line is set
            mismatch = int(r.random() < 0.65); adj_occ = 1; adj_dist = d_true
            route_conf_true = True; believed_unknown = True
        else:
            raise ValueError(base)

        # ------------------------------------------------------ hidden truth
        g = ra.attrs(sA)["gradient"]; curv = ra.attrs(sA)["curvature"]
        v = vA / 3.6; vb = vB / 3.6
        k_rst = r.uniform(0.90, 1.02)
        a_emg = ph.effective_decel(A["a_emg_nominal"], mu, g, k_rst)
        a_srv = ph.effective_decel(0.7 * A["a_emg_nominal"], mu, g, k_rst)
        req_a = ph.tier_distances(v, t_comm, A["t_sb"], A["brake_buildup_s"], a_srv, a_emg)
        aB_emg = ph.effective_decel(B["a_emg_nominal"], mu, -g, k_rst)
        aB_srv = ph.effective_decel(0.7 * B["a_emg_nominal"], mu, -g, k_rst)
        req_b = None; avail = d_true
        if opposing and vb > 0:
            req_b = ph.tier_distances(vb, t_comm, B["t_sb"], B["brake_buildup_s"], aB_srv, aB_emg)
        elif direction == "same" and vb > 0:
            d_b_stop_true = vb * vb / (2 * aB_emg); avail = d_true + d_b_stop_true
        label = int(ph.required_tier(np.array([avail]), [np.array([x]) for x in req_a],
                                     None if req_b is None else [np.array([x]) for x in req_b],
                                     np.array([conflict]))[0])

        # ---------------------------------------------------------- observation
        closing = v + vb if opposing else (max(v - vb, 0.0) if direction == "same" else v)
        sigma_pos = 5.0 + 120.0 * (1.0 - rel)
        eps = r.normal(0, sigma_pos)
        d_obs = d_true + closing * t_comm + eps
        if believed_unknown or base == "safe_parallel":
            d_obs = HORIZON
        d_obs = float(np.clip(d_obs, 5.0, HORIZON))
        spd_noise = 1.0 + 3.0 * (1.0 - rel)
        obs = dict(
            scenario=scen, route_type=ra.route.line_type if not believed_unknown else "main",
            direction=direction, weather=weather, signal_state=sig, authority=auth,
            speed_a_kmh=max(vA + r.normal(0, spd_noise), 0.0), speed_b_kmh=max(vB + r.normal(0, spd_noise), 0.0),
            accel_a=r.normal(0, 0.05), closing_speed_kmh=max(closing * 3.6 + r.normal(0, spd_noise), 0.0),
            track_distance_m=d_obs,
            euclid_distance_m=float(np.clip(euclidean(xA, xyB) + r.normal(0, sigma_pos), 5.0, HORIZON)),
            gradient=g, curvature=curv, adhesion_estimate=MU_NOMINAL[weather],
            occupancy_on_route=occ_on_route if not believed_unknown else 0,
            route_conflict=int((route_conf_true and r.random() < 0.9 and not believed_unknown) or r.random() < 0.03),
            obstruction_reported=obstr if obstr else int(r.random() < 0.01),
            point_mismatch=mismatch if mismatch else int(r.random() < 0.02),
            adjacent_occupancy=adj_occ, adjacent_distance_m=float(np.clip(adj_dist + r.normal(0, sigma_pos), 5, HORIZON)),
            true_speed_a_kmh=vA, true_speed_b_kmh=vB, true_closing_ms=closing, true_k_rst=k_rst,
            comm_delay_s=t_comm, reliability=rel,
            **{k: A[k] for k in ("train_type", "mass_t", "length_m", "a_emg_nominal", "brake_buildup_s")},
            a_serv_nominal=0.7 * A["a_emg_nominal"], t_sb=A["t_sb"],
            b_a_emg_nominal=B["a_emg_nominal"], b_buildup_s=B["brake_buildup_s"], b_t_sb=B["t_sb"],
            # hidden truth (never used as model input)
            true_distance_m=d_true, true_available_m=avail,
            true_d_drv=float(req_a[0] + (0 if req_b is None else req_b[0])),
            true_d_svc=float(req_a[1] + (0 if req_b is None else req_b[1])),
            true_d_emg=float(req_a[2] + (0 if req_b is None else req_b[2])), true_mu=mu, true_conflict=int(conflict),
            label=label,
        )
        return obs

    def generate(self, n: int) -> pd.DataFrame:
        names = list(SCENARIOS); p = np.array(list(SCENARIOS.values())); p /= p.sum()
        scen = self.rng.choice(names, size=n, p=p)
        df = pd.DataFrame([self._one(s) for s in scen])
        df.insert(0, "record_id", np.arange(n))
        return add_physics_features(df)


# ---------------------------------------------------------------- observed physics
def nominal_requirements(df, mu_table=MU_NOMINAL, distance=None, stale_correct=False, k_sigma=0.0,
                         k_rst=1.0, absolute_leader=False):
    """Physics evaluated on OBSERVED inputs (what a deterministic layer would compute).

    The conservative options are used by the physics shield:
      k_sigma         inflate speed (+k*sigma_v) and deflate distance (-k*sigma_pos)
      stale_correct   subtract the distance closed during the data latency T_c
      k_rst           lower bound of the rolling-stock braking-performance factor
      absolute_leader ignore a moving leader's own stopping distance (brick-wall assumption)
    """
    rel = df["reliability"].to_numpy()
    sig_v = (1.0 + 3.0 * (1.0 - rel)) / 3.6                       # m/s, speed-report noise s.d.
    sig_pos = 5.0 + 120.0 * (1.0 - rel)                            # m, position-report noise s.d.
    mu = df["weather"].map(mu_table).to_numpy()
    v = df["speed_a_kmh"].to_numpy() / 3.6 + k_sigma * sig_v
    vb_obs = df["speed_b_kmh"].to_numpy() / 3.6
    g = df["gradient"].to_numpy(); tc = df["comm_delay_s"].to_numpy()
    a_emg = ph.effective_decel(df["a_emg_nominal"].to_numpy(), mu, g, k_rst)
    a_srv = ph.effective_decel(df["a_serv_nominal"].to_numpy(), mu, g, k_rst)
    req = ph.tier_distances(v, tc, df["t_sb"].to_numpy(), df["brake_buildup_s"].to_numpy(), a_srv, a_emg)
    opp = (df["direction"] == "opposing").to_numpy() & (vb_obs > 0)
    vb = vb_obs + np.where(opp, k_sigma * sig_v, 0.0)
    aB = ph.effective_decel(df["b_a_emg_nominal"].to_numpy(), mu, -g, k_rst)
    aBs = ph.effective_decel(0.7 * df["b_a_emg_nominal"].to_numpy(), mu, -g, k_rst)
    reqB = ph.tier_distances(vb, tc, df["b_t_sb"].to_numpy(), df["b_buildup_s"].to_numpy(), aBs, aB)
    req = [a + np.where(opp, b, 0.0) for a, b in zip(req, reqB)]
    d = (df["track_distance_m"] if distance is None else df[distance]).to_numpy().astype(float)
    closing = df["closing_speed_kmh"].to_numpy() / 3.6
    if stale_correct:
        d = d - (closing + k_sigma * sig_v) * tc
    if k_sigma:
        d = d - k_sigma * sig_pos
    same = (df["direction"] == "same").to_numpy()
    leader = 0.0 if absolute_leader else np.where(same & (vb_obs > 0), vb_obs * vb_obs / (2 * aB), 0.0)
    d_avail = d + leader
    return d_avail, req


def drcri(df, d_safe, d_avail):
    """Dynamic Railway Collision Risk Index (patent formulation, 0-100) on observed data."""
    d = df["track_distance_m"].to_numpy()
    closing = np.maximum(df["closing_speed_kmh"].to_numpy() / 3.6, 0.1)
    ttc = np.minimum(d / closing, 600.0)
    margin = d_avail - d_safe
    Rd = np.clip(d_safe / np.maximum(d, 1.0), 0, 1)
    Rt = np.clip(1 - ttc / 120.0, 0, 1)
    Rb = np.clip(0.5 - margin / (2 * np.maximum(d_safe, 1.0)), 0, 1)
    Rr = np.clip(0.6 * df["route_conflict"] + 0.4 * df["occupancy_on_route"], 0, 1).to_numpy()
    Rs = df["signal_state"].map({"green": 0.0, "yellow": 0.4, "red": 0.8, "failed": 0.7, "red_passed": 1.0}).to_numpy()
    Ra = df["authority"].map({"valid": 0.0, "limited": 0.5, "written": 0.6, "invalid": 1.0}).to_numpy()
    Ro = (df["obstruction_reported"] * np.clip(1 - d / 3000.0, 0, 1)).to_numpy()
    Ru = 1 - df["reliability"].to_numpy()
    idx = 100 * (0.22 * Rd + 0.18 * Rt + 0.19 * Rb + 0.15 * Rr + 0.09 * Rs + 0.08 * Ra + 0.05 * Ro + 0.04 * Ru)
    return idx, ttc


def add_physics_features(df):
    d_avail, req = nominal_requirements(df)
    d_safe = req[2] + ph.MARGIN
    df["d_emg_nominal"] = req[2]
    df["d_svc_nominal"] = req[1]
    df["margin_emg_nominal"] = d_avail - d_safe
    idx, ttc = drcri(df, d_safe, d_avail)
    df["drcri"] = idx
    df["ttc_s"] = ttc
    return df
