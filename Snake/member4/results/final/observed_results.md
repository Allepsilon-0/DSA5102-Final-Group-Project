# Observed results

Execution mode: **final**. Smoke results are pipeline checks, not final findings.

| Clip | Test score (mean ± seed SD) | Success rate (mean ± seed SD) |
|---|---:|---:|
| 0.1 | 2.603 ± 0.523 | 9.9% ± 6.0 pp |
| 0.2 | 2.085 ± 0.432 | 4.8% ± 2.8 pp |
| 0.3 | 2.258 ± 0.465 | 2.4% ± 3.5 pp |

The unit of replication is the training seed. Episode-level variation is not a substitute for independent training runs.

Paired differences relative to clip=0.2 (descriptive bootstrap intervals; only five seeds in final mode):

- clip=0.1 minus 0.2: 0.518, paired bootstrap 95% interval [-0.056, 1.210].
- clip=0.3 minus 0.2: 0.173, paired bootstrap 95% interval [-0.407, 0.725].

Do not select or retune a configuration on these test results. A follow-up experiment needs new held-out seeds.
A positive mean difference alone does not establish a reliable improvement. Check individual seeds and learning curves.