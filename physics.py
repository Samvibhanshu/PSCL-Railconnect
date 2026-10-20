"""Longitudinal train-dynamics and intervention-feasibility model.

Stopping distance follows the structure of the ETCS emergency-brake deceleration
(EBD) model (ERA_ERTMS_040026): during command latency and brake build-up the
train is assumed to keep its speed, after which it decelerates at the
adhesion-limited, gradient-corrected deceleration.

    a_eff = min(K_rst * a_brake, mu * g) - g * G          (G > 0: downhill)
    D(v)  = v * (T_c + T_sys + T_bu) + v^2 / (2 a_eff)

Four intervention tiers are evaluated, ordered by severity:
    0 Low       - advisory to the driver is sufficient (driver reaction + normal,
                  comfort-limited service braking at COMFORT x full service rate)
    1 Moderate  - automatic service-brake intervention is required
    2 High      - automatic emergency-brake intervention is required
    3 Critical  - even immediate emergency braking cannot stop the train before
                  the conflict point minus the protective margin
"""
from __future__ import annotations

import numpy as np

G_ACC = 9.81
T_DRIVER = 7.0      # s: driver perception-reaction (~4 s) + service-brake build-up (~3 s)
T_SYS = 1.0         # s: onboard processing + actuation delay for automatic intervention
COMFORT = 0.6       # fraction of full service deceleration used in normal (driver) braking
T_BS = 3.0          # s: supervision time-shift between service- and emergency-intervention
                    #    curves (analogous to the ETCS SBI-EBI shift)
MARGIN = 50.0       # m: protective margin (overlap) before the conflict point
TIER_NAMES = ["Low", "Moderate", "High", "Critical"]


def effective_decel(a_brake, mu, gradient, k_rst=1.0):
    a = np.minimum(k_rst * a_brake, mu * G_ACC) - G_ACC * gradient
    return np.maximum(a, 0.03)


def stopping_distance(v, latency, a_eff):
    """v in m/s, latency in s, a_eff in m/s^2 -> metres."""
    return v * latency + v * v / (2.0 * a_eff)


def tier_distances(v, t_comm, t_sb, t_eb, a_serv, a_emg):
    """Required distances for the three escalating interventions."""
    d_drv = stopping_distance(v, t_comm + T_DRIVER, COMFORT * a_serv)
    d_svc = stopping_distance(v, t_comm + T_SYS + t_sb + T_BS, a_serv)
    d_emg = stopping_distance(v, t_comm + T_SYS + t_eb, a_emg)
    return d_drv, d_svc, d_emg


def required_tier(available, req_a, req_b=None, conflict=True):
    """Minimum sufficient intervention tier.

    available : distance (m) to the conflict point (relative-braking corrected)
    req_a     : tuple (d_drv, d_svc, d_emg) for the approaching train
    req_b     : optional tuple for an opposing train that brakes simultaneously
    conflict  : boolean array, False => no physical conflict exists (tier 0)
    """
    avail = np.asarray(available, float) - MARGIN
    d = [np.asarray(x, float) for x in req_a]
    if req_b is not None:
        d = [a + np.asarray(b, float) for a, b in zip(d, req_b)]
    tier = np.full(avail.shape, 3, dtype=int)
    tier[avail >= d[2]] = 2
    tier[avail >= d[1]] = 1
    tier[avail >= d[0]] = 0
    tier[~np.asarray(conflict, bool)] = 0
    return tier
