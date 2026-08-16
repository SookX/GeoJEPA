# Cube epsilon robustness summary

N=500 pretrained LeWM Cube diagnostic. All metrics compare model and simulator counterfactual geometry in normalized-action coordinates.

| eps | D_G median | D_G 95% CI | top1 angle median | top1 95% CI | top2 subspace max median | top2 subspace mean median | erank env | erank model | k_env median | k_model median | D_kappa median | frac model>env | D_scale median |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-03 | 0.585 | [0.5649, 0.6158] | 55.22 deg | [50.49, 58.6] | 42.71 deg | 24.98 deg | 1.959 | 2.355 | 3001 | 1.056e+04 | 0.563 | 54.2% | 1.159 |
| 3e-03 | 0.541 | [0.5142, 0.5628] | 48.59 deg | [44.74, 54.07] | 39.60 deg | 22.91 deg | 2.002 | 2.355 | 2691 | 1.056e+04 | 0.359 | 53.2% | 0.677 |
| 1e-02 | 0.490 | [0.4603, 0.5127] | 45.87 deg | [40.19, 50.23] | 35.95 deg | 20.38 deg | 2.022 | 2.355 | 3360 | 1.056e+04 | 0.243 | 52.2% | 0.695 |

Takeaway: D_G remains large and top-1 direction error remains large across all perturbation radii. The exact condition numbers are contact/epsilon-sensitive, but the qualitative Cube mismatch is stable.
