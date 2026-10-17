# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
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
- A sync whose data has practice or qualifying but no completed race no longer feeds each
  session to the pace filter twice.
- `_sync_runs` recorded snapshot ID 1 for every commit after the catalog was reopened (it
  read `ducklake_last_committed_snapshot`, which returns NULL once a process reconnects); the
  metadata now uses `ducklake_current_snapshot`, so each run records the snapshot it creates
  and `f1sim data verify` compares against the right rows (P1-10, P1-14).
- Jolpica session listings put a sprint before qualifying on a sprint weekend.
- The test network guard allows loopback sockets, so the API tests run on Windows.

### Added
- Phase 4 tyres and pit strategy (P4-01 to P4-10):
  - pooled per-compound degradation curves with a spline that bends up into a cliff, shrunk
    per circuit toward the global shape (P4-01);
  - a warm-up penalty on the first two laps of a stint, held out of the fit (P4-02), and an
    optional track-temperature interaction kept only if it improves held-out error (P4-03);
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
  - `f1sim tyres fit | check | windows` and ADR-0007 (tyres and strategy design).
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
