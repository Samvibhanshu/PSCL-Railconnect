# RailConflict-PSCL

Code and data for the manuscript

> R. R. Pathak, "Physics-Shielded Conformal Learning for Network-Aware Railway Collision-Risk Assessment Under Degraded Observations," submitted to *IEEE Access*, 2026.

**PSCL** (Physics-Shielded Conformal Learning) turns a probabilistic railway collision-risk classifier into a graded intervention decision with two independent safety layers:

1. **Mondrian conformal escalation**: per-severity-class conformal prediction sets. The decision is the most severe tier in the set, which gives a finite-sample bound on the per-class *under-classification* rate (Critical ≤ 2 %, High ≤ 5 %, Moderate ≤ 10 % by default).
2. **Physics shield**: a conservative, uncertainty-inflated ETCS-style braking envelope used as a severity floor. It is independent of the data distribution.

Labels are **intervention-grounded**. Each label is the minimum intervention (advisory, service brake, emergency brake, or infeasible) that physically avoids the conflict, computed from the *hidden true state*. The predictors only see *degraded observations*: stale, noisy positions, weather-nominal adhesion, interlocking-reported routes and failed track circuits.

| Item | Location |
|---|---|
| Source code (this repository) | https://github.com/RakshitRPathak/RailConflict-PSCL |
| Archived release + dataset (Zenodo) | https://doi.org/10.5281/zenodo.XXXXXXX  *(fill in after the first release)* |
| Reproducible capsule (Code Ocean) | https://doi.org/10.24433/CO.XXXXXXX.v1  *(fill in after publishing the capsule)* |
| Licence | Code: MIT. Data: CC BY 4.0 |

## Repository layout (Code Ocean compatible)

```
code/
  run                    # Code Ocean entry point -> python main.py
  main.py                # reproduces every table and figure of the paper into ../results
  ingest_digitraffic.py  # OPTIONAL: snapshot real Finnish train locations (CC BY 4.0); not used for results
  railsafe/
    network.py           # graph digital twin: track graph, interlocking route table, chainage geometry
    physics.py           # ETCS-style braking model and intervention-tier ground truth
    simulate.py          # 12-scenario encounter simulator, hidden truth vs degraded observations, DRCRI
    models.py            # LR / RF / HGB / MLP, fixed-threshold, nominal physics, DRCRI rule, physics shield
    conformal.py         # Mondrian (class-conditional) split-conformal escalation
    evaluate.py          # safety metrics (missed-Critical, severe-miss, false-emergency, cost)
    figures.py           # manuscript figures
data/
  railconflict_encounters_seed2026_n24000.csv   # released canonical dataset (regenerated + SHA-256 checked)
  LICENSE-DATA.txt
environment/
  Dockerfile, requirements.txt                  # pinned Python 3.13 environment
results/                                        # written by the run (empty in the repository)
results_reference/                              # outputs of the reference run reported in the paper
docs/fig_framework.*                            # framework diagram (Graphviz source + render)
```

## Reproduce

```bash
pip install -r requirements.txt
cd code
python main.py            # full study: 10 seeds x 24,000 encounters (about 10 min on 2 cores)
python main.py --quick    # smoke test: 2 seeds x 9,000 encounters (about 3 min)
```

On **Code Ocean**, press *Reproducible Run*. The `code/run` script writes everything to `/results`.

All randomness is seeded (simulation seeds 1000+s, split/model seeds s = 0..9, canonical dataset seed 2026). `results/summary.json` records the library versions, the runtime and a SHA-256 check that the regenerated canonical dataset is identical to the one shipped in `data/`.

### Main outputs

| File | Paper item |
|---|---|
| `table_main.csv` | Table: main comparison (mean ± s.d. over 10 seeds) |
| `table_ablation.csv` | Table: feature and graph-distance ablation |
| `table_shift.csv` | Table: distribution shift (low adhesion + degraded communications) |
| `table_conformal_sweep.csv` | Fig.: empirical conditional coverage vs. nominal level |
| `table_scenarios.csv` | Fig./Table: scenario-wise severe-miss rates |
| `table_importance.csv` | Fig.: permutation importance |
| `summary.json` | Protocol-A (index-label) audit, latency, confusion matrices, environment |
| `fig_*.png / .pdf` | All result figures |

## Dataset dictionary (`railconflict_encounters_*.csv`)

One row per train-pair encounter. **Observed** columns are model inputs. **true_*** columns and `label` are hidden ground truth and must never be used as inputs. `scenario` is metadata for analysis only.

| Column | Meaning |
|---|---|
| scenario | one of 12 scenario families (metadata, not an input) |
| route_type, direction | interlocking-reported route type (main/loop/branch) and train-pair relation |
| speed_a_kmh, speed_b_kmh, accel_a, closing_speed_kmh | observed kinematics (noisy) |
| track_distance_m | observed track distance to the nearest conflict point along the set route (stale + noisy; 8,000 m horizon if none known) |
| euclid_distance_m | straight-line distance between the train and the conflict object (ablation only) |
| train_type, mass_t, length_m, a_emg_nominal, a_serv_nominal, brake_buildup_s, t_sb | rolling-stock data of the approaching train |
| b_a_emg_nominal, b_buildup_s, b_t_sb | rolling-stock data of the other train |
| gradient, curvature | track geometry at the train (gradient > 0 = downhill) |
| weather, adhesion_estimate | weather class and its nominal adhesion coefficient |
| signal_state, authority | green / yellow / red / red_passed / failed; valid / limited / invalid / written |
| occupancy_on_route, route_conflict, obstruction_reported, point_mismatch, adjacent_occupancy, adjacent_distance_m | multi-source indicators |
| comm_delay_s, reliability | data latency and source-reliability score |
| d_emg_nominal, d_svc_nominal, margin_emg_nominal, ttc_s, drcri | physics-informed features computed on observed data |
| true_distance_m, true_available_m, true_mu, true_conflict, true_d_drv, true_d_svc, true_d_emg | hidden truth |
| label | ground-truth minimum intervention tier: 0 Low, 1 Moderate, 2 High, 3 Critical |

## Scope

This is a TRL-4 software study on simulated data. It is **not** a certified safety function and must not be used for train control. See the paper's *Limitations* section.

## Citation

See `CITATION.cff`.
