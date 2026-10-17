# Benchmarks

Backtest scores per model, `benchmarks/<model>.json`, each with the git commit, snapshot ID and seasons scored (P2-13). The CI regression gate compares model PRs against the best score recorded here (P2-14).

Latest runs: seasons 2021–2022 (44 races), snapshot 10, by `f1sim backtest --model all --seasons 2021-2022 --record`, with the pace hyperparameters in `configs/pace_hyper.json` (fitted on 2021 by `f1sim pace fit --seasons 2021`). Lower RPS is better:

| Model | RPS | Log loss | Brier win | RPS 2022 only |
|---|---|---|---|---|
| simulator | 0.1249 | 2.4823 | 0.0328 | 0.1292 |
| grid_noise | 0.1304 | 2.5791 | 0.0376 | 0.1353 |
| pace | 0.1312 | 2.5830 | 0.0361 | 0.1372 |
| plackett_luce | 0.1368 | 2.6683 | 0.0397 | 0.1427 |
| elo | 0.1498 | 2.7708 | 0.0440 | 0.1470 |

The simulator row is its latest run, after the track-position calibration (ADR-0008); its earlier run on the same snapshot scored 0.1277 (0.1325 on 2022). The pace hyperparameters were fitted on 2021, and so were the simulator's three track-position settings, so the 2021 half of the pace and simulator scores is in sample; 2022 is the out-of-sample check.

Earlier runs (snapshot 7) scored the pace model at 0.1630, before the backtest cutoff fix (ADR-0005 amendment), the pace observation fixes (ADR-0006) and the fitted hyperparameters. The baselines read only results and score the same on both snapshots.

The `Model regression` workflow (`.github/workflows/model-regression.yml`) restores or syncs those seasons and runs `f1sim backtest --model all --seasons 2021-2022 --check`.
