# 0003. Inference backend for Bayesian models

- Status: Accepted
- Date: 2026-10-13 (proposed 2026-10-11)
- Roadmap: P3-04, P3-11, P3-12

## Context

The pace model is refitted after every session of a race weekend, and the post-qualifying run has a time budget. The options were PyMC (NUTS), NumPyro on JAX (NUTS), and variational or Laplace approximations. The decision criteria were the time for a weekend update on a laptop, the posterior predictive check, and R-hat and effective sample size (ESS).

## Decision

Split the work in two, because the pace model is linear and Gaussian once its hyperparameters are fixed (ADR-0006):

1. **Weekend updates use an exact Kalman filter** in NumPy, with the hyperparameters held fixed. Each session is one exact Bayesian update, so there is nothing to sample and nothing to converge.
2. **Hyperparameters are fitted with NumPyro**, occasionally (once per season, or after a regulation change), with `f1sim pace fit`:
   - The latent effects are integrated out exactly by the same Kalman filter, written in JAX, so NUTS samples 15 hyperparameters instead of thousands of latent effects. A test checks that the JAX likelihood equals the NumPy filter's to 1e-4.
   - `--method nuts` (the default) samples the posterior and reports R-hat and ESS for every hyperparameter.
   - `--method laplace` finds the posterior mode with BFGS and uses its curvature. It is much faster, and it reports a failed fit when a hyperparameter sits at zero, where the approximation breaks down.

Race-day ranking noise is not part of the pace model. It is chosen by backtesting a grid of values on the training seasons.

## Measurements

All numbers are on synthetic data, measured in a 2-core cloud container: four seasons of 12 rounds, five teams, 144 sessions and 1,440 observations. Synthetic data is used because no real seasons have been synced yet. Re-measure on real data and record the numbers here.

| Step | Time | Notes |
|---|---|---|
| Build pace observations | 1.6 s | Lap fits for all 144 sessions; cached across backtest races |
| Kalman update, per session | 0.3–0.4 ms | 48 latent effects |
| Weekend update with a saved state | 0.1 s | Two new sessions (practice, qualifying), against 0.5 s for a full refit |
| Laplace fit | 31 s | Mostly JAX compilation |
| NUTS fit, 2 chains × (500 + 500) | 956 s | Every R-hat ≤ 1.006; ESS 515–2,000 for 1,000 draws |
| Earlier design: NUTS on explicit latent effects | 162 s | ESS 12–75 and R-hat up to 1.23: not usable |

Posterior predictive check on the held-out 2024 season (12 races, 120 driver forecasts), using the NUTS medians above: 55% of drivers fell inside their 50% interval and 90% inside their 90% interval, with a chi-squared ratio of 0.86. With untuned defaults the check fails (86% and 100%, ratio 0.21), which shows it can catch a miscalibrated model.

## Alternatives considered

- **PyMC.** Its default C backend needs a compiler toolchain, which is fragile on Windows, the main development machine. It was not benchmarked. The model only uses `numpyro.sample` and `numpyro.factor` around a pure JAX likelihood, so a port remains possible if NumPyro ever becomes a problem.
- **NUTS over explicit latent effects.** This was built first. It sampled thousands of dimensions with a funnel between the random-walk scales and their steps, and it mixed poorly.
- **Variational inference only.** It is fast, but there is no R-hat or ESS to show that it has worked. The Laplace option gives that speed for routine refits, with a check for when it breaks down.

## Consequences

- JAX and NumPyro are in the `models` extra; JAX publishes wheels for Windows, macOS and Linux.
- NUTS cost grows with the number of sessions times the square of the number of effects. On four real seasons (about 450 sessions, about 150 effects) expect hours on a laptop. Use `--method laplace` for routine refits, and NUTS overnight or on fewer seasons when the diagnostics matter.
- CI runs a short NUTS fit and the predictive check on a small synthetic fixture (P3-11).
