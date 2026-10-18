# 0010. Regulation-reset study: experiments as data, resets as a config strategy

- Status: Accepted
- Date: 2026-10-18
- Roadmap: P7-01 to P7-05

## Context

Phase 7 answers the project's headline question: when a regulation change makes the history before it a poor guide, how many races does the pace model need to be accurate again, and which way of handling that history gets there fastest? Answering it needs four things — the resets to study, a switchable way to handle each one, a recovery metric, and a run anyone can reproduce. The pace model (P3-04) and the backtester (P2-04) already exist, and its gate is the same as Phase 6's: nothing is added that changes the Phase 3–6 numbers unless it is asked for.

## Decision

**Natural experiments are data, not code (P7-01).** `configs/natural-experiments.yaml` declares each reset — seasons (pre-reset included), the reset season and round, the data level (`lap` or `results`) and, when it is runnable, the snapshot that pinned a published result. `NaturalExperiment` (core domain) names one; `load_natural_experiments` validates the manifest. An entry with no snapshot is *declared but not pinned*: 2022 (ground-effect cars) runs on snapshot 10; 2014 (hybrid era) is results-level only, because lap timing starts in 2018; 2026 has no data yet. The study reports an unpinned entry instead of failing, so the manifest records intent the moment a reset is announced.

**A reset is a pure transform of the filter's posterior, chosen by name (P7-02).** `f1sim.adapters.models.resets` holds one strategy per way of handling the reset, each with `at_weekend(state, season, new_season)` and `noise_scale(group, season)`:

- `full` keeps the Phase 3 behaviour — the filter's own random walk already ages the effects;
- `decay` inflates the whole covariance by `2 ** (1 / half_life)` every weekend, so a race a half-life back counts half;
- `reset` at the reset season's first weekend sets each car effect's mean to zero and its own variance to `sd ** 2`, clearing every covariance that touches it;
- `testing` is a `reset` whose new era is guided by its opening weekend's practice (the pre-season testing the store cannot see): those long runs carry `practice_weight` times the weight, by shrinking the practice noise scale.

Only the **car** effects — `team:`, `sup:`, `tq:`, `tc:` — are reset; driver effects carry over, because the reset replaces the cars, not the drivers. `PaceContext.reset` carries the strategy, `ModelOptions.reset` the config, `PaceFilter.advance` applies it at a season boundary and `update` reads `noise_scale`; nothing in the engine or the state changes. `reset_signature` enters the pace model's `version`, so a fitted state saved under one strategy is never resumed under another.

**Recovery is RPS by race number after the reset (P7-03).** `f1sim.reports.recovery.run_recovery` walks the pace model forward across the experiment's seasons once per strategy, sharing one seed, and scores every race. A curve is the post-reset races' RPS in race order (race 1 is the reset round), read against `pre_reset_rps`, the mean RPS of the season before the reset; `recovery_race` is the first race at or below that level. `render_recovery` prints a milestone summary and the full per-race table; `f1sim report recovery` writes it to `docs/research/regulation-reset.md`.

**The notebook runs in CI against a pinned fixture (P7-04, P7-05).** `notebooks/01-regulation-reset.ipynb` runs the study end to end and executes in CI from a clean environment. CI has no raw data and no network (ADR-0001), so the notebook reads the synthetic `tests/pace_world` world committed as one in-memory snapshot; it exercises the same `run_recovery` the CLI does. The narrative write-up is `docs/research/regulation-reset-writeup.md`.

## Alternatives considered

- **Reset inside the engine or a `LapStep`.** A regulation reset is not a lap mechanic; putting it in the pipeline would couple a research knob to the engine and need a step that no race uses. A config-selected strategy on the filter's context keeps the engine closed.
- **Re-drawing every effect, drivers included.** Simple, but the regulation changes the cars, not the drivers; the driver and teammate comparisons from Phase 3 are still the best information available and carry over.
- **Weighting older races through observation weights.** The Kalman update has no per-observation weight; inflating the prior covariance every weekend is the exact, closed-form equivalent of fading history and keeps the filter a Kalman filter.
- **Fitting reset priors from pre-season testing.** The store has no testing sessions; the reset season's opening practice is the closest observable stand-in, which is why `testing` reweights it rather than inventing data.
- **Running the notebook on the real lake in CI.** ADR-0001 keeps raw data out of CI, and a notebook that silently skips when data is missing is worse evidence than one that runs on a fixture and fails loudly.
- **A single hard-coded 2022 study.** Encoding the experiment in code would make the next reset a code change; a manifest makes it a data change and lets the roadmap's 2014 and 2026 entries be declared now.

## Consequences

- The default strategy is `full`, so Phases 3–6 reproduce exactly; the reset only appears when it is asked for through options or the CLI.
- The pace model's version digest now includes the reset signature, so a fitted-state cache entry is not shared across strategies.
- Recovery is measured on one season of 22 races per experiment and is deterministic for the pace model (its forecast has no Monte Carlo draw), so the numbers are reproducible — but the race-to-race spread in RPS is far wider than the between-strategy differences, so the findings are directional, not a ranking with error bars.
- 2014 and 2026 are declared but not runnable until their data is synced; `f1sim report recovery` says so rather than failing.
- The `research` dependency group (Jupyter, matplotlib) is separate from `dev`, so the test suite stays fast and does not need a notebook stack.
