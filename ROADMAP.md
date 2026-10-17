# Shpeed Roadmap

Snapshot of the Shpeed Roadmap tracker, 11 Oct 2026. The tracker is the source of truth; regenerate this file from it ("Export ROADMAP.md") when statuses change.

- **Plan:** 26 weeks at 10–15 h/week, week 1 starting Mon 12 Oct 2026, v1.0 by 11 Apr 2027
- **Progress:** 0 of 115 must-have features done through v1.0, 36 in progress or in review (130 features in total, including should-haves and stretch goals)
- **Releases:** v0.1 (MVP) at the end of Phase 5; v1.0 at the end of Phase 9

## How to use this roadmap

- Every feature has an ID (`P1-02`). Use it in the branch name (`feat/P1-02-fastf1-adapter`), the PR title, commits and the CHANGELOG.
- A phase starts only when the previous phase's **gate** has passed.
- `must` features count toward progress; `should` features are cut first if time slips (weather before anything else); `stretch` waits until after v1.0.
- *In review* means built and passing locally, waiting to be merged with CI green on GitHub.
- Tags show where a feature puts a SOLID or ACID principle into practice.

### Definition of done

1. Code merged to main through a reviewed PR titled with the feature ID
2. Tests added; CI green (lint, strict types, coverage, import contracts)
3. The feature's "done when" criterion demonstrated
4. Docs or ADR updated if behaviour or a decision changed
5. CHANGELOG entry; PR link saved on the feature in the tracker

## Schedule

| Phase | Weeks | Dates | Must-haves done | Gate | Release |
|---|---|---|---|---|---|
| 0 · Foundations | 1 | 12 Oct – 18 Oct 2026 | 0/18 | open |  |
| 1 · Data ingestion | 2–4 | 19 Oct – 8 Nov 2026 | 0/18 | open |  |
| 2 · Baselines & evaluation | 5–6 | 9 Nov – 22 Nov 2026 | 0/13 | open |  |
| 3 · Pace model | 7–10 | 23 Nov – 20 Dec 2026 | 0/11 | open |  |
| 4 · Tyres & strategy | 11–13 | 21 Dec – 10 Jan 2027 | 0/8 | open |  |
| 5 · Race simulator | 14–16 | 11 Jan – 31 Jan 2027 | 0/13 | open | v0.1 |
| 6 · Race events | 17–19 | 1 Feb – 21 Feb 2027 | 0/6 | open |  |
| 7 · Regulation-reset study | 20–22 | 22 Feb – 14 Mar 2027 | 0/5 | open |  |
| 8 · Delivery | 23–24 | 15 Mar – 28 Mar 2027 | 0/15 | open |  |
| 9 · Open-source launch | 25–26 | 29 Mar – 11 Apr 2027 | 0/8 | open | v1.0 |
| Stretch goals | — | after v1.0 | 6 ideas | — | — |

## Phase 0 — Foundations

**Goal:** A repo anyone can clone, install and test with one command, with the engineering standards in place from day one.

**Gate:** A fresh clone runs `uv sync && pytest` green in CI, with lint, types and coverage enforced.

| | ID | Feature | Area | Priority | Done when | Depends on | Principles |
|---|---|---|---|---|---|---|---|
| ⬜ | `P0-01` | Project name and MIT licence | Foundations | must | LICENSE committed; name reserved on PyPI and GitHub. | — |  |
| 🔍 | `P0-02` | Python 3.12 project on uv with src/ layout | Foundations | must | `uv sync` from a clean clone creates the env from the lockfile. | — |  |
| 🔍 | `P0-03` | Package skeleton with module boundaries | Architecture | must | data/, store/, models/, sim/, eval/, delivery/ exist; an import-linter contract forbids upward imports (e.g. data cannot import sim). | P0-02 | SOLID:SRP, SOLID:DIP |
| 🔍 | `P0-04` | Core interfaces defined as Protocols | Architecture | must | SourceAdapter, DataRepository, Predictor, LapStep and Publisher are typed Protocols in a `ports` module with docstrings. | P0-03 | SOLID:ISP, SOLID:DIP |
| 🔍 | `P0-05` | Ruff lint and format | Quality | must | Ruff runs in pre-commit and CI; zero warnings on main. | P0-02 |  |
| 🔍 | `P0-06` | Pyright in strict mode | Quality | must | Strict type-check passes in CI; no blanket `type: ignore`. | P0-02 |  |
| 🔍 | `P0-07` | Pre-commit hooks | Quality | must | Hooks run ruff, pyright, trailing-whitespace and a large-file guard. | P0-05, P0-06 |  |
| 🔍 | `P0-08` | pytest, hypothesis and coverage gate | Quality | must | Tests run with coverage; CI fails below 85% line coverage on core packages. | P0-02 |  |
| 🔍 | `P0-09` | CI pipeline on every pull request | Quality | must | GitHub Actions runs lint, types, tests and import contracts; required to merge. | P0-07, P0-08 |  |
| ⬜ | `P0-10` | Branch protection and PR workflow | Process | must | main is protected; squash merges; PR template with checklist; Conventional Commits titles referencing a feature ID. | P0-09 |  |
| 🔍 | `P0-11` | README, CONTRIBUTING, code of conduct, templates | Process | must | Files present; issue templates for bug, feature and model-change proposals. | — |  |
| 🔍 | `P0-12` | ADR log with template and ADR-001 | Process | must | docs/adr/ holds a template and ADR-001: raw data is fetched by code, never committed. | — |  |
| 🔍 | `P0-13` | Typed settings and config loading | Architecture | must | pydantic-settings object for paths, seeds and budgets; no module reads env vars directly. | P0-04 | SOLID:DIP |
| 🔍 | `P0-14` | Structured logging with run IDs | Observability | must | structlog JSON logs; every CLI run gets a run_id that appears on all its log lines. | P0-03 |  |
| 🔍 | `P0-15` | Domain error hierarchy | Architecture | must | ShpeedError base with DataError, ValidationError, LeakageError, BudgetExceeded; no bare `except`. | P0-03 |  |
| 🔍 | `P0-16` | SemVer, CHANGELOG and release tagging | Process | must | CHANGELOG.md in keep-a-changelog format; tags trigger a release job. | P0-09 |  |
| 🔍 | `P0-17` | Dependency updates and audit | Security | must | Dependabot enabled; pip-audit runs in CI and fails on known vulnerabilities. | P0-09 |  |
| 🔍 | `P0-18` | Frontend workspace scaffold | Frontend | must | frontend/ (React + TypeScript + Vite, pnpm) lints, type-checks, tests and builds in CI; ADR-0004 records the stack. | P0-09 |  |

## Phase 1 — Data ingestion

**Goal:** A local, validated, transactional store of every session from 2018, plus results back to 1950.

**Gate:** A full 2018–2026 sync is reproducible from scratch, passes validation, and an interrupted sync leaves no partial snapshot.

| | ID | Feature | Area | Priority | Done when | Depends on | Principles |
|---|---|---|---|---|---|---|---|
| 🔍 | `P1-01` | Domain model for F1 entities | Data | must | Frozen dataclasses for Season, Event, Session, Driver, Team, Lap, Stint, PitStop, TrackStatus, Result with unit tests. | P0-04 | SOLID:SRP |
| 🔍 | `P1-02` | FastF1 source adapter | Data | must | Implements SourceAdapter: laps, stints, compounds, weather, track status, race-control messages. | P1-01 | SOLID:LSP, SOLID:DIP |
| 🔍 | `P1-03` | Jolpica-F1 source adapter | Data | must | Implements SourceAdapter: results, grids, pit stops, finishing status, standings. | P1-01 | SOLID:LSP, SOLID:DIP |
| 🔍 | `P1-04` | Resilient fetching | Data | must | Shared HTTP layer with cache, exponential backoff with jitter and a rate limiter; tested with simulated 429/5xx. | P1-02, P1-03 | SOLID:SRP |
| 🔍 | `P1-05` | Canonical driver, team and event keys | Data | must | Stable IDs across sources; mapping tables versioned in configs/. | P1-01 |  |
| 🔍 | `P1-06` | Team lineage table | Data | must | Rebrands resolve to one entity (e.g. Sauber → Audi); tested on 2018–2026 entries. | P1-05 |  |
| 🔍 | `P1-07` | Validation schemas | Data | must | pandera schemas for every table: lap-time ranges, no duplicate laps, contiguous stints. | P1-01 | ACID:C |
| 🔍 | `P1-08` | clean_lap flag | Data | must | Excludes in/out laps, lap 1, and SC/VSC/red-flag laps; property-tested. | P1-07 |  |
| 🔍 | `P1-09` | DuckLake store over Parquet (ADR-002) | Store | must | Tables live in a DuckLake catalog (DuckDB file) with Parquet data files; ADR-002 records the choice over Delta Lake, Iceberg and a hand-rolled catalog. | P1-01 | ACID:A, ACID:D |
| 🔍 | `P1-10` | One sync = one transaction | Store | must | Each sync writes all its tables inside a single DuckLake transaction; the resulting snapshot ID is stored with sync metadata (sources, versions, run_id). | P1-09 | ACID:A, ACID:D |
| 🔍 | `P1-11` | Atomic commit protocol | Store | must | Data is validated before COMMIT and a failed check rolls back; a kill test mid-sync leaves the previous snapshot intact and readable. | P1-07, P1-10 | ACID:A, ACID:C, ACID:D |
| 🔍 | `P1-12` | Single-writer lock | Store | must | A file lock prevents two syncs at once (the DuckDB-file catalog is single-writer); the second writer exits with a clear message. | P1-11 | ACID:I |
| 🔍 | `P1-13` | Snapshot-pinned DataRepository | Store | must | read(table, snapshot_id, as_of) queries AT (VERSION => snapshot_id); readers never see a sync in progress. | P1-10 | ACID:I, SOLID:DIP |
| 🔍 | `P1-14` | Integrity verification | Store | must | `f1sim data verify` checks every data file the catalog references exists and row counts match the sync metadata. | P1-10 | ACID:D |
| 🔍 | `P1-15` | Incremental sync | Data | must | Sync skips sessions already present in the latest snapshot; a re-run with no new data creates no snapshot. | P1-11 |  |
| 🔍 | `P1-16` | Per-session data-quality report | Data | must | Markdown report of row counts, validation failures and clean-lap share per session. | P1-07 |  |
| 🔍 | `P1-17` | CLI: f1sim data sync | Delivery | must | `f1sim data sync --season 2025` runs the full pipeline and prints the new snapshot ID. | P1-11, P1-15 |  |
| 🔍 | `P1-18` | Recorded fixtures for offline CI | Quality | must | Adapter tests replay recorded responses; CI blocks outbound network. | P1-02, P1-03 |  |
| 🔍 | `P1-19` | Snapshot retention | Store | should | `f1sim data gc` expires old snapshots and cleans unreferenced files, never one a published prediction or benchmark cites. | P1-10 | ACID:D |

## Phase 2 — Baselines & evaluation

**Goal:** Know what good looks like before building anything clever.

**Gate:** Baseline scores for 2021–2022 are recorded in the repo; Plackett–Luce is the bar to clear.

| | ID | Feature | Area | Priority | Done when | Depends on | Principles |
|---|---|---|---|---|---|---|---|
| ⬜ | `P2-01` | PositionMatrix prediction type | Eval | must | Immutable type: drivers × positions plus P(DNF); constructor enforces invariants. | P0-04 | ACID:C |
| ⬜ | `P2-02` | Property tests on PositionMatrix | Eval | must | Hypothesis tests: rows and columns sum to 1, values in [0,1]. | P2-01 |  |
| ⬜ | `P2-03` | Predictor protocol and registry | Eval | must | Every model implements fit/predict; models register by name so eval code never imports a concrete model. | P2-01 | SOLID:LSP, SOLID:OCP |
| ⬜ | `P2-04` | Walk-forward backtester | Eval | must | For each round, fits on data up to its cutoff and predicts; parallel across rounds. | P2-03, P1-13 |  |
| ⬜ | `P2-05` | Leakage guard | Eval | must | Repository raises LeakageError on any read past as_of; a test plants future data and must fail loudly. | P2-04 | ACID:I |
| ⬜ | `P2-06` | Scoring metrics | Eval | must | Brier (win, podium, top 10), log loss, ranked probability score, Spearman; tested on hand-computed cases. | P2-01 |  |
| ⬜ | `P2-07` | Calibration curves | Eval | must | Reliability diagrams per probability bucket in the report. | P2-06 |  |
| ⬜ | `P2-08` | Baseline: grid order with noise | Models | must | Registered predictor with backtest scores recorded. | P2-03 |  |
| ⬜ | `P2-09` | Baseline: Elo ratings | Models | must | Driver and team Elo predictor with backtest scores recorded. | P2-03 |  |
| ⬜ | `P2-10` | Baseline: Plackett–Luce | Models | must | Ranking model on recent results; its RPS becomes the merge bar. | P2-03 |  |
| ⬜ | `P2-11` | Market benchmark | Eval | should | If historical closing odds are sourced, margin-removed odds scored the same way. | P2-06 |  |
| ⬜ | `P2-12` | Backtest report generator | Eval | must | One HTML/markdown report per season with metrics, calibration and per-race breakdown. | P2-06, P2-07 |  |
| ⬜ | `P2-13` | Benchmark registry in repo | Eval | must | benchmarks/<model>.json stores scores with commit and snapshot ID. | P2-12 |  |
| ⬜ | `P2-14` | Regression gate for model PRs | Quality | must | CI fails a model-change PR whose backtest RPS is worse than the recorded best. | P2-13 |  |

## Phase 3 — Pace model

**Goal:** Estimate each car-and-driver's race pace with honest uncertainty, updated through a weekend.

**Gate:** Posterior predictive checks pass on held-out races, and pace ranking alone beats the Phase 2 baseline.

| | ID | Feature | Area | Priority | Done when | Depends on | Principles |
|---|---|---|---|---|---|---|---|
| ⬜ | `P3-01` | Fuel-effect estimation | Models | must | Per-circuit fuel effect estimated from data; corrected laps comparable across a race. | P1-08 |  |
| ⬜ | `P3-02` | Track-evolution estimation | Models | must | Session lap-time trend modelled and removed. | P1-08 |  |
| ⬜ | `P3-03` | Mixed-effects reference model | Models | must | statsmodels lap time ~ team + driver + compound + tyre age + fuel + evolution; error on held-out laps reported. | P3-01, P3-02 |  |
| ⬜ | `P3-04` | Bayesian team pace random walk | Models | must | PyMC/NumPyro model with team pace drifting race to race. | P3-03 |  |
| ⬜ | `P3-05` | Driver effect vs teammate | Models | must | Partially pooled driver effect across seasons. | P3-04 |  |
| ⬜ | `P3-06` | Circuit clustering and interactions | Models | must | Circuits clustered by speed profile; team × cluster term. | P3-04 |  |
| ⬜ | `P3-07` | Correlated qualifying and race pace | Models | must | Separate but correlated quali and race effects per driver. | P3-05 |  |
| ⬜ | `P3-08` | Engine-supplier and 2026 energy effect | Models | should | Supplier effect nested above team; clipping feature from telemetry tested. | P3-04 |  |
| ⬜ | `P3-09` | Weekend updating | Models | must | Posterior updated after each practice, sprint and qualifying session from cached state. | P3-04, P3-10 |  |
| ⬜ | `P3-10` | Versioned fitted-state store | Store | must | Posteriors saved atomically (temp file + rename) keyed by model version, round and snapshot ID. | P1-11 | ACID:A, ACID:D |
| ⬜ | `P3-11` | Automated model diagnostics | Quality | must | R-hat, ESS and posterior predictive checks run in CI on a small fixture. | P3-04 |  |
| ⬜ | `P3-12` | Inference speed budget and ADR | Architecture | must | Weekend update timed; ADR records PyMC vs NumPyro vs VI decision. | P3-09 |  |

## Phase 4 — Tyres & strategy

**Goal:** Model compound degradation per circuit and the strategies teams are likely to run.

**Gate:** Degradation predicts held-out stints well; actual stop laps fall inside predicted windows at a measured rate.

| | ID | Feature | Area | Priority | Done when | Depends on | Principles |
|---|---|---|---|---|---|---|---|
| ⬜ | `P4-01` | Hierarchical degradation splines | Models | must | Lap-time loss vs tyre age per compound × circuit with a cliff; pooled across circuits. | P3-01 |  |
| ⬜ | `P4-02` | Tyre warm-up penalty | Models | must | First-laps penalty after a stop, by compound. | P4-01 |  |
| ⬜ | `P4-03` | Track-temperature interaction | Models | should | Temperature term on degradation, kept only if it improves held-out error. | P4-01 |  |
| ⬜ | `P4-04` | Pit loss per circuit | Models | must | Pit-in plus pit-out loss measured from timing data. | P1-08 |  |
| ⬜ | `P4-05` | Stop-time distribution per team | Models | must | Stationary-time distribution with a slow-stop tail. | P1-03 |  |
| ⬜ | `P4-06` | Season rules schema and loader | Architecture | must | configs/rules/<season>.yaml validated by a schema for 2018–2026: compounds, distance, sprint format, overtake aid. | P0-13 | SOLID:OCP, ACID:C |
| ⬜ | `P4-07` | Strategy generator | Models | must | Enumerates legal 1–3 stop plans from the rules and expected race time. | P4-01, P4-04, P4-06 |  |
| ⬜ | `P4-08` | Strategy chooser | Models | must | P(plan) per driver mixing the time-optimal plan with team tendencies. | P4-07 |  |
| ⬜ | `P4-09` | Tyre-sets-remaining constraint | Models | should | Sets used in practice and quali limit available plans. | P4-07 |  |
| ⬜ | `P4-10` | Pit-window calibration report | Eval | must | Share of actual stop laps inside predicted windows, reported per circuit. | P4-08 |  |

## Phase 5 — Race simulator (v0.1)

**Goal:** Run each race thousands of times lap by lap and output the probability matrix.

**Gate:** Backtested on 2022–2025, the simulator beats Plackett–Luce on ranked probability score.

| | ID | Feature | Area | Priority | Done when | Depends on | Principles |
|---|---|---|---|---|---|---|---|
| ⬜ | `P5-01` | Vectorised race state | Sim | must | RaceState arrays of shape N × cars: order, gaps, compound, tyre age, fuel, pit status. | P2-01 | SOLID:SRP |
| ⬜ | `P5-02` | Pluggable lap-step pipeline | Sim | must | Engine runs an ordered list of LapStep objects; adding an event type needs no engine edit. | P5-01, P0-04 | SOLID:OCP, SOLID:ISP |
| ⬜ | `P5-03` | Lap-time sampling from posterior draws | Sim | must | Each run draws pace and degradation parameters; noise from driver consistency. | P5-02, P3-09, P4-01 |  |
| ⬜ | `P5-04` | Overtaking model | Models | must | Pass probability from pace gap and circuit difficulty, fitted on historical position changes. | P1-08 |  |
| ⬜ | `P5-05` | Dirty-air following penalty | Sim | must | Lap-time loss when within ~1 s of the car ahead. | P5-02 |  |
| ⬜ | `P5-06` | Era-specific overtake aid | Sim | must | DRS (to 2025) vs Overtake Mode (2026) read from rules config. | P5-04, P4-06 | SOLID:OCP |
| ⬜ | `P5-07` | Pit-stop execution step | Sim | must | Stops executed from each driver's sampled plan with sampled stationary time. | P4-05, P4-08 |  |
| ⬜ | `P5-08` | Starting state from grid | Sim | must | Grid after penalties and pit-lane starts applied before lap 1. | P5-01 |  |
| ⬜ | `P5-09` | Seeded, reproducible runs | Sim | must | Same seed + snapshot + model version gives a bit-identical matrix. | P5-02 |  |
| ⬜ | `P5-10` | Simulator property tests | Quality | must | Valid ordering every lap, non-negative gaps, every car finishes or is DNF. | P5-02 |  |
| ⬜ | `P5-11` | Performance benchmark in CI | Quality | must | 10,000 runs of one race under 1 minute; pytest-benchmark tracks it. | P5-03 |  |
| ⬜ | `P5-12` | Aggregate runs into PositionMatrix | Sim | must | Finishing orders across runs become the matrix plus P(DNF). | P5-02 |  |
| ⬜ | `P5-13` | Release v0.1 (MVP) | Process | must | Gate passed; tagged and changelog written. | P5-12 |  |

## Phase 6 — Race events

**Goal:** Add the randomness that decides real races, and prove each piece earns its place.

**Gate:** No component kept that worsens the backtest; ablation table published.

| | ID | Feature | Area | Priority | Done when | Depends on | Principles |
|---|---|---|---|---|---|---|---|
| ⬜ | `P6-01` | Safety car and VSC | Sim | must | Per-circuit hazards; SC bunches field and cheapens stops. | P5-02 |  |
| ⬜ | `P6-02` | Red flag and restart | Sim | must | Free tyre change and standing restart reusing the lap 1 model. | P6-01, P6-05 |  |
| ⬜ | `P6-03` | Team reliability hazard | Models | must | DNF hazard per team-season with Bayesian shrinkage. | P1-03 |  |
| ⬜ | `P6-04` | Driver crash hazard | Models | must | Accident hazard per driver, shrunk toward field average. | P1-03 |  |
| ⬜ | `P6-05` | Lap 1 model | Sim | must | Launch performance and first-corner incident probability. | P5-08 |  |
| ⬜ | `P6-06` | In-race penalties | Sim | should | 5/10-second penalty hazard per driver. | P5-02 |  |
| ⬜ | `P6-07` | Rain regime | Sim | should | Forecast rain probability mixes dry and wet runs; first thing cut if time slips. | P5-02 |  |
| ⬜ | `P6-08` | Traffic and defending penalties | Sim | should | Time loss behind backmarkers and in battles. | P5-05 |  |
| ⬜ | `P6-09` | Ablation runner and table | Eval | must | Each component switched off in turn; table published in docs. | P2-04, P5-02 | SOLID:OCP |

## Phase 7 — Regulation-reset study

**Goal:** Measure how fast the model recovers after a regulation reset, and which approach recovers fastest.

**Gate:** The notebook reruns end to end from a clean environment and the findings are written up.

| | ID | Feature | Area | Priority | Done when | Depends on | Principles |
|---|---|---|---|---|---|---|---|
| ⬜ | `P7-01` | Natural-experiment datasets | Research | must | 2022 and 2026 at lap level, 2014 at results level, each as a pinned snapshot. | P1-13 |  |
| ⬜ | `P7-02` | Reset strategies compared | Research | must | Full history, exponential decay, reset priors, testing-informed priors as config options. | P3-04 |  |
| ⬜ | `P7-03` | Recovery curves | Research | must | Score by race number after reset for each strategy. | P7-02 |  |
| ⬜ | `P7-04` | Reproducible notebook | Research | must | Executes in CI from a clean env against pinned snapshots. | P7-03 |  |
| ⬜ | `P7-05` | Blog write-up | Research | must | Findings published as a post. | P7-04 |  |

## Phase 8 — Delivery

**Goal:** Predictions publish themselves every race weekend with no manual steps.

**Gate:** One full race weekend runs end to end hands-off.

| | ID | Feature | Area | Priority | Done when | Depends on | Principles |
|---|---|---|---|---|---|---|---|
| ⬜ | `P8-01` | CLI: predict, backtest, report | Delivery | must | All three commands documented with --help and tested. | P5-12, P2-04 |  |
| ⬜ | `P8-02` | Versioned prediction record schema | Delivery | must | JSON Schema v1 with provenance: git commit, model version, snapshot ID, data cutoff, seed. | P2-01 | ACID:C |
| ⬜ | `P8-03` | Atomic, idempotent publishing | Delivery | must | Record written via temp + rename and committed in one git commit; re-running a round yields the same file; a published record is never changed after lights out. | P8-02 | ACID:A, ACID:D, ACID:I |
| 🚧 | `P8-04` | Read-only FastAPI service | Delivery | must | /predictions/{season}/{round} serves records; contract-tested against the schema. | P8-02 | SOLID:DIP |
| 🔍 | `P8-05` | Docker images and Compose stack | Delivery | must | docker compose up --build starts api (:8000) and web (:8080) with health checks; compose.dev.yaml gives hot reload; images are multi-stage, non-root and read-only; CI builds and smoke-tests the stack. | P8-04 |  |
| ⬜ | `P8-06` | Scheduled post-qualifying run | Delivery | must | Workflow syncs, predicts and publishes after qualifying. | P8-01, P8-03 |  |
| ⬜ | `P8-07` | Scheduled post-race scoring | Delivery | must | Workflow syncs results, scores the record and updates the scoreboard. | P8-06 |  |
| ⬜ | `P8-08` | Failure alerting and retries | Observability | must | Failed workflow retries once, then notifies; run logs kept as artifacts. | P8-06 |  |
| ⬜ | `P8-09` | Race prediction screen | Frontend | must | Race screen shows the latest prediction with Win / Podium / Points views, a provenance footer and accessible probability bars, reading predictions/ JSON. | P8-03, P8-11, P8-12 |  |
| ⬜ | `P8-10` | Season scoreboard and calibration screens | Frontend | must | Season screen lists scored races; Calibration screen plots predicted vs observed frequency with an equivalent text table. | P8-07, P8-09 |  |
| ⬜ | `P8-11` | Design system on Apple HIG | Frontend | must | Tokens for type (Dynamic Type scale), semantic colour in light and dark, spacing, materials and motion; primitives (LargeTitle, SegmentedControl, GroupedList, ProbabilityBar, EmptyState, Badge) tested; DESIGN.md current. | P0-18 | SOLID:SRP |
| ⬜ | `P8-12` | App shell and navigation | Frontend | must | Tab bar on phone widths, sidebar on iPad and Mac widths, safe-area insets respected, hash routing works on GitHub Pages. | P8-11 |  |
| ⬜ | `P8-13` | Types generated from the shared schema | Frontend | must | pnpm gen:types regenerates src/types/prediction.ts from schemas/prediction-record.v1; CI fails if the committed types are stale. | P8-02, P0-18 | ACID:C |
| ⬜ | `P8-14` | Frontend deployed to GitHub Pages | Frontend | must | pages.yml builds the app with predictions/ included and deploys on every merge to main. | P8-09 |  |
| ⬜ | `P8-15` | Accessibility and performance audit | Frontend | must | WCAG AA contrast in both themes, VoiceOver labels checked, full keyboard navigation, Lighthouse accessibility and performance 95+ on mobile. | P8-10 |  |
| ⬜ | `P8-16` | End-to-end smoke tests | Frontend | should | Playwright runs on desktop Safari and iPhone viewports in CI. | P8-12 |  |

## Phase 9 — Open-source launch (v1.0)

**Goal:** v1.0 is public and usable, with live predictions from the 2027 opener.

**Gate:** v1.0 is on PyPI and automated predictions are live for the 2027 opener.

| | ID | Feature | Area | Priority | Done when | Depends on | Principles |
|---|---|---|---|---|---|---|---|
| ⬜ | `P9-01` | Docs site | Docs | must | mkdocs-material site deployed from main. | P0-11 |  |
| ⬜ | `P9-02` | Guides | Docs | must | Quickstart, how the model works, how to add a component (a LapStep or Predictor). | P9-01 | SOLID:OCP |
| ⬜ | `P9-03` | API reference | Docs | must | Generated from docstrings with mkdocstrings. | P9-01 |  |
| ⬜ | `P9-04` | PyPI publishing | Process | must | Trusted publishing from a tagged release. | P0-16 |  |
| ⬜ | `P9-05` | Community issues | Process | must | Good-first-issue labels and roadmap issues for stretch goals. | — |  |
| ⬜ | `P9-06` | Non-affiliation disclaimer | Docs | must | Disclaimer in README, docs and site footer. | — |  |
| ⬜ | `P9-07` | Launch post | Docs | must | Methodology and reg-reset findings shared with the FastF1 community. | P7-05 |  |
| ⬜ | `P9-08` | Release v1.0 | Process | must | Gate passed; tagged; live predictions running for the 2027 opener. | P9-04, P8-10 |  |

## Stretch goals

**Goal:** Ideas held back until after v1.0, so scope stays fixed.

**Gate:** Picked up only after v1.0 ships.

| | ID | Feature | Area | Priority | Done when | Depends on | Principles |
|---|---|---|---|---|---|---|---|
| ⬜ | `S-01` | In-race live predictions | Sim | stretch | Predictions update from live timing during a race. | — |  |
| ⬜ | `S-02` | Pit-strategy optimiser | Models | stretch | Best strategy for a given race state. | — |  |
| ⬜ | `S-03` | Sprint race predictions | Sim | stretch | Sprint races predicted and scored. | — |  |
| ⬜ | `S-04` | What-if mode | Sim | stretch | Swap driver A into car B. | — |  |
| ⬜ | `S-05` | OpenF1 as a second source | Data | stretch | OpenF1 adapter as fallback and for intervals. | — |  |
| ⬜ | `S-06` | Drying-track tyre switching | Sim | stretch | Track wetness state and crossover timing. | — |  |

---

Legend: ⬜ to do · 🚧 in progress · 🔍 in review · ✅ done · ~~cut~~ dropped.
