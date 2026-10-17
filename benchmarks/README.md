# Benchmarks

Backtest scores per model, `benchmarks/<model>.json`, each with the git commit, snapshot ID and seasons scored (P2-13). The CI regression gate compares model PRs against the best score recorded here (P2-14).

Recorded on seasons 2021–2022 (44 races), snapshot 7, by `f1sim backtest --model all --seasons 2021-2022 --record`. Lower RPS is better:

| Model | RPS | Log loss | Brier win |
|---|---|---|---|
| grid_noise | 0.1304 | 2.5791 | 0.0376 |
| plackett_luce | 0.1368 | 2.6683 | 0.0397 |
| elo | 0.1498 | 2.7708 | 0.0440 |
| pace | 0.1630 | 3.2767 | 0.0437 |

The `Model regression` workflow (`.github/workflows/model-regression.yml`) restores or syncs those seasons and runs `f1sim backtest --model all --seasons 2021-2022 --check`.
