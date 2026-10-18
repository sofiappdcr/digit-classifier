# Sprint 3 - Experiments (validation set)

Each row changes ONE thing from the baseline. Results are the mean of 3 runs with different random initializations (seeds 42, 1, 2).

| # | Configuration | Val acc (mean ± std) | Train acc | Epochs | Time (s) |
|---|---|---|---|---|---|
| 1 | Baseline (64) | 97.87% ± 0.35% | 100.00% | 181 | 0.2 |
| 2 | Fewer neurons (32) | 96.94% ± 0.39% | 100.00% | 239 | 0.2 |
| 3 | More neurons (128) | 98.06% ± 0.23% | 100.00% | 142 | 0.2 |
| 4 | More neurons (256) **(best)** | 98.43% ± 0.35% | 100.00% | 115 | 0.2 |
| 5 | Two layers (64, 64) | 96.94% ± 0.60% | 100.00% | 108 | 0.2 |
| 6 | Two layers (128, 64) | 97.78% ± 0.45% | 100.00% | 88 | 0.2 |
| 7 | Lower learning rate (0.0001) | 95.00% ± 0.60% | 97.96% | 300 | 0.3 |
| 8 | Higher learning rate (0.01) | 98.15% ± 0.13% | 100.00% | 47 | 0.0 |
| 9 | Regularization alpha=0.01 | 97.96% ± 0.26% | 100.00% | 179 | 0.2 |
| 10 | Regularization alpha=0.1 | 98.06% ± 0.23% | 100.00% | 190 | 0.2 |
