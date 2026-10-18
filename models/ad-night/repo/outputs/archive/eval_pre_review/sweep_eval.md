# Privacy / utility sweep on `eval`

| run | privacy | high | high+med | keep | prec | concepts | evidence | ICD F1 orig | ICD F1 anon | ICD agree | ms/enc | RAM MB | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sweep_eval_conservative | 0.714 | 0.991 | 0.815 | 1.000 | 0.972 | 1.000 | 1.000 | 0.235 | 0.235 | 0.935 | 50.8 | 98.8 | 0.000 |
| sweep_eval_balanced | 0.806 | 0.996 | 0.913 | 1.000 | 0.961 | 0.999 | 1.000 | 0.235 | 0.235 | 0.928 | 50.5 | 98.1 | 0.000 |
| sweep_eval_aggressive | 0.881 | 0.996 | 0.929 | 0.978 | 0.807 | 0.995 | 0.999 | 0.235 | 0.235 | 0.922 | 49.5 | 97.0 | 0.000 |
| sweep_eval_balanced_nocodes | 0.812 | 0.996 | 0.919 | 1.000 | 0.962 | 0.999 | 1.000 | 0.235 | 0.235 | 0.928 | 48.3 | 81.2 | 0.000 |
| sweep_eval_conservative_rules+deid+gliner | 0.740 | 0.998 | 0.845 | 0.994 | 0.970 | 1.000 | 1.000 | 0.235 | 0.235 | 0.935 | 834.4 | 2,528.6 | 2,059.7 |
| sweep_eval_balanced_rules+deid+gliner | 0.823 | 0.998 | 0.929 | 0.990 | 0.932 | 0.998 | 1.000 | 0.235 | 0.235 | 0.928 | 802.2 | 2,565.3 | 2,059.7 |
| sweep_eval_aggressive_rules+deid+gliner | 0.896 | 0.998 | 0.943 | 0.968 | 0.792 | 0.994 | 0.999 | 0.235 | 0.235 | 0.922 | 801.1 | 2,525.1 | 2,059.7 |
| sweep_eval_balanced_nocodes_rules+deid+gliner | 0.828 | 0.998 | 0.935 | 0.987 | 0.931 | 0.998 | 0.999 | 0.235 | 0.235 | 0.928 | 805.7 | 2,523.1 | 2,059.7 |
