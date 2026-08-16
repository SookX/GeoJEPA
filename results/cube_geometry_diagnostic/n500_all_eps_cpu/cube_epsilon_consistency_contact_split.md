# Cube per-state epsilon consistency and contact split

N=500 pretrained LeWM Cube diagnostic. Metrics are in normalized-action coordinates.

## Per-state epsilon consistency

| metric | eps pair | Spearman | Pearson | median abs diff | median signed diff |
|---|---:|---:|---:|---:|---:|
| D_G | 1e-03 vs 3e-03 | 0.588 | 0.605 | 0.131 | -0.040 |
| D_G | 1e-03 vs 1e-02 | 0.487 | 0.495 | 0.146 | -0.079 |
| D_G | 3e-03 vs 1e-02 | 0.654 | 0.683 | 0.117 | -0.049 |
| top1_angle_deg | 1e-03 vs 3e-03 | 0.371 | 0.365 | 15.670 | -3.286 |
| top1_angle_deg | 1e-03 vs 1e-02 | 0.286 | 0.286 | 18.099 | -5.089 |
| top1_angle_deg | 3e-03 vs 1e-02 | 0.501 | 0.506 | 14.048 | -2.333 |
| subspace2_angle_max_deg | 1e-03 vs 3e-03 | 0.404 | 0.401 | 14.124 | -3.282 |
| subspace2_angle_max_deg | 1e-03 vs 1e-02 | 0.217 | 0.199 | 16.319 | -4.992 |
| subspace2_angle_max_deg | 3e-03 vs 1e-02 | 0.351 | 0.336 | 12.155 | -1.489 |
| effective_rank_env | 1e-03 vs 3e-03 | 0.517 | 0.526 | 0.290 | 0.054 |
| effective_rank_env | 1e-03 vs 1e-02 | 0.492 | 0.501 | 0.322 | 0.102 |
| effective_rank_env | 3e-03 vs 1e-02 | 0.625 | 0.638 | 0.245 | 0.048 |

## Contact split

Contact signal: `proprio_gripper_contact`; main split is any value > 0.5 during the 5-step transition. Counts: non-contact 257, contact 243.

| eps | split | N | median D_G | median top1 angle | median top2 max angle | erank env | erank model | median D_kappa | median E_kappa |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-03 | non-contact | 257 | 0.508 | 47.84 deg | 39.58 deg | 2.135 | 2.487 | 1.296 | 2.117 |
| 1e-03 | contact | 243 | 0.696 | 60.21 deg | 44.55 deg | 1.634 | 2.278 | -3.013 | 4.198 |
| 3e-03 | non-contact | 257 | 0.482 | 43.91 deg | 38.15 deg | 2.248 | 2.487 | 1.218 | 1.831 |
| 3e-03 | contact | 243 | 0.640 | 54.71 deg | 40.74 deg | 1.672 | 2.278 | -4.025 | 4.289 |
| 1e-02 | non-contact | 257 | 0.427 | 36.06 deg | 36.34 deg | 2.263 | 2.487 | 1.168 | 1.786 |
| 1e-02 | contact | 243 | 0.594 | 55.95 deg | 35.60 deg | 1.735 | 2.278 | -4.089 | 4.236 |

Takeaway: the geometry mismatch remains large in the non-contact split. Contact transitions are not required for the Cube mismatch, although contact changes the exact eccentricity/angle statistics.
