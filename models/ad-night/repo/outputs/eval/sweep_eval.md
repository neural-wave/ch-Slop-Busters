# Privacy / utility sweep on `eval`

| run | privacy | high | high+med | keep | prec | concepts | evidence | ICD F1 orig | ICD F1 anon | ICD agree | ms/enc | RAM MB | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sweep_eval_conservative | 0.740 | 0.991 | 0.845 | 1.000 | 0.984 | 1.000 | 1.000 | 0.235 | 0.235 | 0.935 | 82.3 | 98.4 | 0.000 |
| sweep_eval_balanced | 0.828 | 0.996 | 0.910 | 1.000 | 0.962 | 1.000 | 1.000 | 0.235 | 0.235 | 0.928 | 81.3 | 99.0 | 0.000 |
| sweep_eval_aggressive | 0.869 | 0.996 | 0.921 | 0.984 | 0.902 | 1.000 | 1.000 | 0.235 | 0.235 | 0.922 | 82.2 | 99.0 | 0.000 |
| sweep_eval_balanced_nocodes | 0.832 | 0.996 | 0.915 | 0.994 | 0.956 | 1.000 | 0.999 | 0.235 | 0.235 | 0.928 | 80.2 | 84.8 | 0.000 |
| sweep_eval_conservative_rules+deid+gliner | 0.745 | 0.998 | 0.851 | 1.000 | 0.984 | 1.000 | 1.000 | 0.235 | 0.235 | 0.935 | 2,210.9 | 2,137.2 | 2,139.7 |
| sweep_eval_balanced_rules+deid+gliner | 0.839 | 0.998 | 0.921 | 1.000 | 0.958 | 1.000 | 1.000 | 0.235 | 0.235 | 0.928 | 2,014.6 | 2,412.8 | 2,123.7 |
| sweep_eval_aggressive_rules+deid+gliner | 0.879 | 0.998 | 0.932 | 0.984 | 0.900 | 1.000 | 1.000 | 0.235 | 0.235 | 0.922 | 2,102.7 | 2,519.2 | 2,123.7 |
| sweep_eval_balanced_nocodes_rules+deid+gliner | 0.843 | 0.998 | 0.925 | 0.994 | 0.953 | 1.000 | 0.999 | 0.235 | 0.235 | 0.928 | 1,936.2 | 2,506.8 | 2,123.7 |
