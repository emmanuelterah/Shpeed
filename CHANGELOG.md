# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Regulation-reset study (Phase 7): `configs/natural-experiments.yaml` plus
  `load_natural_experiments` declare each reset (2022 lap-level, 2014 results-level, 2026
  lap-level) with the snapshot that pinned it, so an unsynced reset is recorded without being
  run. Prior-handling strategies are config options (`ModelOptions.reset` / `PaceContext.reset`)
  in `f1sim.adapters.models.resets`: full history, exponential decay, reset priors and
  testing-informed priors. Only the car effects are reset; driver effects carry over. A
  strategy's signature enters the pace model's version, so saved states are not reused across
  strategies. `f1sim report recovery` walks the pace model across a reset once per strategy and
  writes the recovery curves to `docs/research/regulation-reset.md` (P7-01 to P7-05).
- `notebooks/01-regulation-reset.ipynb` runs the study end to end and executes in CI from a
  clean environment against a pinned synthetic fixture (ADR-0001), with a `research`
  dependency group and a `notebook` CI job.

### Changed
- The simulator's track position is calibrated against real 2021 races (ADR-0008): a
  `pass_offset` of -1.4 log-odds brings on-track passing from 1.8 times the real rate to the
  real rate, the race-day strategy draw uses a 2 s softmax instead of 8 s, and race-day pace
  noise is 0.3% instead of the pace model's 0.45%, which counted strategy and passing
  randomness twice. The three are `SimulatorContext` settings; `OvertakeStep` and
  `OvertakingModel.pass_probability` take the offset, and `simulator_context` builds the
  registry's context.
- `configs/pace_hyper.json`: pace hyperparameters fitted by NUTS on 2021 (all R-hat at most
  1.003, race-day noise 0.45%), and every model's benchmark re-recorded on snapshot 10. The
  simulator scores RPS 0.128 on 2021–2022 and 0.133 on 2022 alone, the best recorded model.
- The backtester always loads the `laps` table, to read session end times.
- The strategy generator's dynamic program and stint-cost table are vectorised: the same plans,
  about 3.6 times faster.
- `RaceEntry` has a `pit_lane` flag (default `False`) for cars starting from the pit lane (P5-08).
- `Settings.n_sims` (`F1SIM_N_SIMS`) now sets the simulator's runs per race through
  `ModelOptions.n_sims`.
- The `Model regression` workflow also triggers on the engine, steps and perf tests (its path
  filter named a folder that does not exist) and runs the engine benchmark.
- The statistical stack moved from PyMC to NumPyro and JAX (ADR-0003), plus statsmodels.
- `get_model` takes `ModelOptions` (pace hyperparameters, fitted-state store).
- Data dependencies (FastF1, DuckDB, pandera, httpx and others) and the statistical stack
  (NumPyro, JAX, statsmodels) are now core dependencies, so `pip install f1sim` gives a working
  CLI. The `data` and `models` extras are gone; only `api` remains.
- The team lineage table moved into the package (`f1sim/resources/`), so it loads after a
  `pip install`; it previously relied on the repository layout.
- `HttpClient` retry settings are grouped into a `RetryPolicy`.
- A sync's `_sync_runs` metadata now records the snapshot ID, contributing sources and
  versions alongside the row counts, and `f1sim data verify` fails when a table's row count
  disagrees with that metadata (P1-10, P1-14).

### Fixed
- Backtest cutoffs for 2021 and 2022 fell at 15:30 UTC on qualifying day, because Jolpica gives
  only the date of qualifying and the time was assumed; any later qualifying (Bahrain, Jeddah,
  Abu Dhabi, Austin, Mexico City, São Paulo) was hidden from the pace model and the simulator.
  The cutoff now takes the latest session end recorded by the results or the lap timing, never
  past the race (ADR-0005 amendment).
- A session whose fit was rank-deficient (a compound run only by drivers who ran nothing else)
  was discarded whole, losing 23% of practice sessions and some dry qualifying and races; the
  least-used compound is now dropped and the session refitted.
- A driver whose only timed laps were compromised could sit 5–44% off the field in a session's
  pace observation; drivers beyond four robust deviations of the median are now left out
  (ADR-0006). Pace `MODEL_VERSION` is 2, so old saved states are not reused.
- A sync whose data has practice or qualifying but no completed race no longer feeds each
  session to the pace filter twice.
- `_sync_runs` recorded snapshot ID 1 for every commit after the catalog was reopened (it
  read `ducklake_last_committed_snapshot`, which returns NULL once a process reconnects); the
  metadata now uses `ducklake_current_snapshot`, so each run records the snapshot it creates
  and `f1sim data verify` compares against the right rows (P1-10, P1-14).
- Jolpica session listings put a sprint before qualifying on a sprint weekend.
- The test network guard allows loopback sockets, so the API tests run on Windows.
- Pit losses, team stop tendencies and pit-window comparisons use only race-session stops, so
  practice and qualifying stops no longer shift them.
- A held-out stint error with no measurable stints is NaN and fails the Phase 4 gate instead of
  passing on a zero, and pit windows are matched to actual stops by stop index rather than
  position, so a missing stop cannot shift the later windows.
- FastF1 laps were stored with no start time and the weather, track-status and race-control
  timestamps were anchored at 1970-01-01: FastF1 fills `LapStartDate` (and its session time
  zero) only when telemetry is loaded, and the old conversion derived the zero from that same
  missing date. The adapter now falls back to the session's scheduled start, so every timing
  column shares one clock; this repairs the timeline and re-activates the Phase 3
  track-evolution term and the Phase 4 temperature term.

### Added
- Phase 6 race events (P6-01 to P6-09), all fitted with shrinkage toward the field priors and
  switchable off for the ablation gate:
  - safety car and VSC (P6-01) from the race-lap `track_status` (a lap's concatenated status codes
    are read as a set, so a lap is neutralised when any digit is a safety-car or VSC code), pooled
    per circuit, plus the fitted neutralisation length and the red-flag rate (P6-02);
  - `SafetyCarStep` draws and times each neutralisation: a full safety car bunches the field to
    `SC_GAP_S` a car and cheapens stops by `SC_PIT_DISCOUNT_S`, a VSC cheapens them by
    `VSC_PIT_DISCOUNT_S` without bunching, and a red flag gives a free tyre change and a standing
    restart that reuses the lap-1 launch; `RaceState.safety_car` carries the level and
    `RaceState.neutralised` lets `DirtyAirStep`, `OvertakeStep` and `PitStopStep` react to it;
  - team mechanical (P6-03) and driver accident (P6-04) hazards that retire a car mid-race through
    `ReliabilityStep`, replacing the Phase 5 pre-start DNF draw (kept as the disabled fallback);
  - a lap-1 model (P6-05): per-driver launch and a first-corner incident rate, in `Lap1Step`;
  - should-haves: a prior in-race 5/10 s penalty (`PenaltyStep`, P6-06), a rain regime as an input
    probability mixing wet and dry runs (`RainStep`, P6-07) and a defending time loss for a car
    under threat (`TrafficStep`, P6-08);
  - the ablation runner and table (P6-09): `f1sim.reports.ablation.run_ablation` scores the full
    simulator then each component switched off, `f1sim report ablation` publishes `docs/ablation.md`
    and `--check` fails when a kept component makes the backtest worse; `ModelOptions.disabled` and
    `p_rain` reach the simulator's step pipeline;
  - ADR-0009 (race events).
- The published ablation (2021–2022, 1000 runs per race) keeps every component by the gate, with
  one accepted caveat: the safety car's RPS delta is −0.0010, at the edge of the noise band, so it
  is recorded as a marginal finding rather than a clear loss; every other component sits within
  ±0.0001. `docs/ablation.md` states the tolerance so the verdict is read in that light.
- `f1sim backtest --ablate` switches named simulator event components off through `ModelOptions`.
- Phase 5 race simulator (P5-01 to P5-12), registered as the `simulator` model:
  - vectorised `RaceState` over `(n_sims, n_cars)` with cumulative race time as the source of
    the order and gaps (P5-01), run by the ordered `LapStep` pipeline (P5-02);
  - lap times from per-run pace posterior draws, compound offsets, sampled tyre-loss curves,
    fuel and lap noise (P5-03);
  - a pass model fitted on close adjacent pairs in the end-of-lap running order, green-flag
    laps without a stop only, with circuit difficulty and a log-odds bonus for DRS or Overtake Mode read from the season rules (P5-04, P5-06), and an
    overtake step that resolves passes in start-of-lap order, one place per lap;
  - a separate dirty-air step (P5-05) and a pit-stop step with per-stop sampled stop times,
    slow-stop tail included (P5-07);
  - the starting grid with slot offsets, pit-lane starts and `apply_grid_penalties` (P5-08);
  - bit-identical matrices for a seed (P5-09), property tests of every lap's invariants
    (P5-10), a pytest-benchmark engine benchmark (10,000 runs of a 22-car race in about 5 s,
    P5-11) and aggregation into a `PositionMatrix` with P(DNF) (P5-12);
  - a prior-only tyre model for when there is too little history to fit one, and ADR-0008.
- `f1sim data sync --force` re-fetches and replaces sessions already stored, so a change in
  how a source's data is converted can be republished.
- Phase 4 tyres and pit strategy (P4-01 to P4-10):
  - pooled per-compound degradation curves with a spline that bends up into a cliff, shrunk
    per circuit toward the global shape (P4-01);
  - a warm-up penalty on the first two laps of a stint, held out of the fit (P4-02), and an
    optional track-temperature interaction kept only if it improves held-out error, exposed
    through a `--temperature` flag that picks the winner (P4-03);
  - pit loss per circuit measured from the in- and out-laps (it already includes the stop,
    P4-04) and a per-team pit-lane stop-duration distribution with a slow-stop tail (P4-05);
  - season rules in `configs/rules/<season>.yaml` (2018-2026) validated by `SeasonRules` and
    loaded by `f1sim.adapters.config_files` (P4-06);
  - a strategy generator that enumerates legal one- to three-stop plans and their expected
    race time (P4-07), a softmax chooser mixing the optimum with team tendencies (P4-08), and
    a tyre-sets-remaining limit on the plans (P4-09);
  - a pit-window calibration report, the Phase 4 gate: windows are the chooser's mean stop lap
    widened by a stop-lap spread learned from training residuals, and the report measures the
    share of actual stops inside them per circuit (P4-10);
  - `f1sim tyres fit | check | windows`, a `load_tyre_model` readback of the fitted model, and
    ADR-0007 (tyres and strategy design).
- Domain entities for the F1 data (P1-01): frozen `Season`, `Event`, `Session`, `Driver`,
  `Team`, `Lap`, `Stint`, `PitStop`, `TrackStatus`, `RaceControl`, `Result` and `Standing`
  value objects, each buildable from its canonical table row.
- `stints` and `race_control` tables (P1-02), derived from FastF1 laps and race-control
  messages, and a `standings` table (P1-03) fetched from the Jolpica driver and constructor
  standings after each race.
- `SourceAdapter` and `Predictor` contract suites in `tests/contract/`, so every adapter
  and model is checked against its port.
- Phase 3 pace model (P3-01 to P3-12, except the engine clipping feature in P3-08):
  - fuel effect per circuit and track evolution per session, pooled with earlier sessions at
    the same circuit; corrected pace observations from qualifying, races, sprints and
    practice long runs;
  - circuit speed clusters from lap length and qualifying speed, and an engine-supplier
    table (`f1sim/resources/circuits.yaml`, `engine_suppliers.yaml`);
  - Bayesian state-space pace model: team random walk, supplier, team quali offset, team x
    cluster and correlated driver quali/race effects, updated exactly by a Kalman filter;
  - hyperparameters fitted by NUTS (or a Laplace approximation) in NumPyro on the exact
    marginal likelihood, with R-hat, ESS and a posterior predictive check on held-out races;
  - statsmodels mixed-effects reference model with held-out lap error;
  - versioned, atomic fitted-state store (`FittedStateStore` port, `FileStateStore`), so a
    weekend update resumes from the last race weekend instead of refitting;
  - `pace` predictor registered for backtests; `f1sim pace fit | check | reference | update`;
  - ADR-0003 accepted (NumPyro) and ADR-0006 (pace model design).
- Phase 2 baselines and evaluation harness (P2-01 to P2-14, except the market benchmark P2-11):
  - `Predictor` port now takes the race entries, plus `ModelSpec` and `PredictorFactory` so a
    backtest builds a fresh, seeded model per race;
  - per-race scoring: Brier (win, podium, top 10), log loss, RPS with DNF as the last category,
    Spearman, and calibration bins;
  - walk-forward backtester with a leakage guard (`CutoffReader` raises `LeakageError`) and the
    cutoff and grid rules of ADR-0005; races run in parallel with per-race seeds, so results are
    reproducible;
  - baselines: grid with fitted noise, driver + team Elo following team lineage, and
    Plackett–Luce fitted by MM with a gamma prior and time decay;
  - Markdown backtest report, `benchmarks/<model>.json` run records and an RPS regression gate;
  - `f1sim backtest --model --seasons --report --record --check --workers`;
  - real 2021–2022 benchmark runs for every registered model, recorded in `benchmarks/`, and a
    PR-scoped `Model regression` workflow that restores (or syncs) those seasons and runs
    `f1sim backtest --check` (P2-13, P2-14).
- `data gc` also protects snapshots cited by benchmark runs.
- Repository scaffold: package layout, ports, CI, ADRs 0001–0004, frontend workspace (P0-02, P0-03, P0-12, P0-18).
- Docker Compose stack: `api` (read-only FastAPI over `predictions/`) and `web` (nginx serving the frontend, proxying `/api`), plus `compose.dev.yaml` with hot reload; CI builds and smoke-tests it (P8-04, P8-05).
- Phase 1 data ingestion (P1-01 to P1-19):
  - canonical table registry with `available_at` on every row, and pandera validation for every table;
  - `clean_lap` flag;
  - transactional DuckLake store with snapshot-pinned reads, a single-writer lock, `verify` and retention;
  - in-memory test store and a shared store contract suite;
  - resilient HTTP layer (cache, rate limits, backoff with jitter);
  - Jolpica-F1 and FastF1 adapters, canonical Jolpica IDs and the team lineage table;
  - incremental `sync_season` use case;
  - `f1sim data sync | verify | report | gc` and a per-session data-quality report;
  - offline fixtures, and `scripts/record_fixtures.py` to record real ones.
- Code standards enforced in CI and pre-commit: pylint 10/10 (with Google docstring checks),
  ruff pydocstyle (Google convention), 100-character lines in every code file, imports only at
  the top of files, pyright strict, and `twine check` / `check-wheel-contents` on every build.
- `f1sim.api.healthcheck` for container health checks.
- PyPI metadata: SPDX licence, project URLs, keywords, classifiers, single-source version,
  README links rewritten to GitHub on the PyPI page.
- Read-only API routes: `/health`, `/predictions`, `/predictions/latest`, `/predictions/{season}/{round}` (P8-04).
