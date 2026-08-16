# Apple-to-apple local action geometry comparison

Definition: pretrained LeWM, N=500, frameskip=5, normalized-action coordinates, finite-difference epsilon 1e-3.

| Environment | action dim | median D_G | median k_env | median k_model | median D_kappa | median E_kappa | frac k_model>k_env | median top1 angle | median D_scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reacher | 2 | 0.256 | 1.86 | 4.44 | 0.854 | 0.956 | 88.6% | 4.41 deg | 2.592 |
| Cube | 5 | 0.585 | 3e+03 | 1.06e+04 | 0.563 | 2.961 | 54.2% | 55.22 deg | 1.159 |

Interpretation:
- Reacher shows the earlier pattern: dominant direction is close and LeWM strongly exaggerates eccentricity.
- Cube shows a larger shape mismatch and much weaker dominant-direction agreement; eccentricity exaggeration is only slight at eps=1e-3.
- Cube is contact-heavy, so the full Cube folder also includes 3e-3 and 1e-2 epsilon robustness outputs.
