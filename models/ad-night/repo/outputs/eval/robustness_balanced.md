# Robustness to unseen formats

| run | privacy | high | high+med | keep | prec | concepts | evidence | ICD F1 orig | ICD F1 anon | ICD agree | ms/enc | RAM MB | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| robust_holdout2_balanced | 0.710 | 0.771 | 0.719 | 0.926 | 0.953 | 0.995 | 0.953 | 0.298 | 0.362 | 0.723 | 41.9 | 96.8 | 0.000 |
| robust_holdout2_balanced_nocodes | 0.714 | 0.777 | 0.723 | 0.921 | 0.951 | 0.995 | 0.906 | 0.298 | 0.319 | 0.723 | 35.9 | 82.7 | 0.000 |
| robust_holdout_balanced | 0.758 | 0.858 | 0.773 | 0.959 | 0.945 | 0.998 | 0.997 | 0.304 | 0.319 | 0.768 | 37.3 | 97.1 | 0.000 |
| robust_holdout_balanced_nocodes | 0.758 | 0.858 | 0.773 | 0.941 | 0.940 | 0.998 | 0.970 | 0.304 | 0.304 | 0.768 | 34.2 | 83.0 | 0.000 |
| robust_discharge_balanced | 0.992 | 1.000 | 0.992 | 1.000 | 0.994 | 0.996 | 1.000 | 0.444 | 0.632 | 0.640 | 5.6 | 97.8 | 0.000 |
| robust_discharge_balanced_nocodes | 0.992 | 1.000 | 0.992 | 1.000 | 0.994 | 0.996 | 1.000 | 0.444 | 0.632 | 0.640 | 5.5 | 84.0 | 0.000 |
| robust_medconv_balanced_nocodes | 0.970 | 1.000 | 0.970 | - | 0.936 | 1.000 | - | - | - | - | 11.3 | 82.8 | 0.000 |
| robust_synthetic_balanced | 0.958 | 0.987 | 0.964 | 1.000 | 0.984 | 0.997 | 0.980 | 0.339 | 0.393 | 0.750 | 26.5 | 96.8 | 0.000 |
| robust_synthetic_balanced_nocodes | 0.958 | 0.987 | 0.964 | 0.985 | 0.977 | 0.993 | 0.941 | 0.339 | 0.393 | 0.732 | 24.1 | 82.8 | 0.000 |
