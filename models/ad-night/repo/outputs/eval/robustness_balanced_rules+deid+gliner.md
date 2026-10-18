# Robustness to unseen formats

| run | privacy | high | high+med | keep | prec | concepts | evidence | ICD F1 orig | ICD F1 anon | ICD agree | ms/enc | RAM MB | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| robust_holdout2_balanced_rules+deid+gliner | 0.891 | 0.928 | 0.902 | 0.842 | 0.930 | 0.995 | 0.953 | 0.298 | 0.319 | 0.660 | 3,474.7 | 1,882.2 | 2,139.7 |
| robust_holdout2_balanced_nocodes_rules+deid+gliner | 0.895 | 0.934 | 0.907 | 0.837 | 0.929 | 0.995 | 0.906 | 0.298 | 0.277 | 0.660 | 1,366.1 | 2,166.9 | 2,139.7 |
| robust_holdout_balanced_rules+deid+gliner | 0.896 | 0.948 | 0.913 | 0.897 | 0.919 | 0.998 | 0.993 | 0.304 | 0.348 | 0.710 | 1,113.3 | 2,517.0 | 2,163.7 |
| robust_holdout_balanced_nocodes_rules+deid+gliner | 0.896 | 0.948 | 0.913 | 0.879 | 0.914 | 0.998 | 0.960 | 0.304 | 0.319 | 0.725 | 1,090.5 | 2,505.6 | 2,171.7 |
| robust_discharge_balanced_rules+deid+gliner | 0.997 | 1.000 | 0.997 | 1.000 | 0.957 | 0.996 | 1.000 | 0.444 | 0.634 | 0.642 | 207.9 | 2,518.8 | 2,097.7 |
| robust_discharge_balanced_nocodes_rules+deid+gliner | 0.997 | 1.000 | 0.997 | 1.000 | 0.957 | 0.996 | 1.000 | 0.444 | 0.634 | 0.642 | 201.0 | 2,505.0 | 2,097.7 |
| robust_medconv_balanced_nocodes_rules+deid+gliner | 0.970 | 1.000 | 0.970 | - | 0.923 | 1.000 | - | - | - | - | 358.9 | 2,506.8 | 2,131.7 |
| robust_synthetic_balanced_rules+deid+gliner | 0.970 | 0.994 | 0.976 | 0.970 | 0.955 | 0.997 | 0.980 | 0.339 | 0.411 | 0.768 | 864.2 | 2,518.6 | 2,131.7 |
| robust_synthetic_balanced_nocodes_rules+deid+gliner | 0.970 | 0.994 | 0.976 | 0.955 | 0.949 | 0.993 | 0.941 | 0.339 | 0.411 | 0.750 | 904.1 | 2,504.7 | 2,131.7 |
