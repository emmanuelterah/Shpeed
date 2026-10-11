# F1 Race Simulator: Phase-by-Phase Plan

Oct 11, 2026 · @Immah

## Overview

The plan takes an open-source F1 race simulator from an empty repo to v1.0 in about 26 weeks at 10–15 hours a week. Starting now, v1.0 lands around April 2027, early in the 2027 season, so the live track record can start near round 1.

The simulator forecasts each race by modelling it lap by lap: pace, tyre wear, pit strategy, safety cars and reliability. It outputs a probability for every driver finishing in every position. Its headline research question comes from the 2026 regulation reset: when history stops being reliable, how quickly can a model become accurate again?

Each phase ends with a gate, a test it must pass before the next phase starts. The MVP gate is the end of Phase 5 (v0.1); Phases 6–9 make it better, prove it, and ship it.

&#91;embedded content: roadmap · 10 phases, 2 release gates\]

Phases 3 and 5 are the critical path; if time slips, cut Phase 6's weather work before anything else.

## Phase 0 — Foundations (week 1)

Goal: a repo anyone can clone, install and test with one command.

- [ ] Pick a name and a licence. MIT matches FastF1 and keeps adoption easy.
- [ ] Python 3.12, `uv` for environments and the lockfile, `src/` layout.
- [ ] Ruff for lint and format, pyright (strict) for types, pre-commit hooks.
- [ ] pytest plus hypothesis for property tests.
- [ ] GitHub Actions: lint, type-check and test on every pull request.
- [ ] README stub, CONTRIBUTING, code of conduct, issue and PR templates.
- [ ] `docs/adr/` for architecture decision records. ADR-001: raw data is fetched by code, never committed to the repo.

Suggested layout:

```
f1sim/
  src/f1sim/
    data/        # ingestion, schema, validation
    models/      # baselines, pace, tyres, events
    sim/         # Monte Carlo engine
    eval/        # backtester, metrics, reports
    cli.py
    api/
  tests/
  notebooks/     # research write-ups
  configs/       # season rules, circuits
  docs/
```

**Gate:** a fresh clone runs `uv sync && pytest` green in CI.

## Phase 1 — Data ingestion (weeks 2–4)

Goal: a local, validated store of every session from 2018 to now, plus results back to 1950. FastF1 gives lap timing, telemetry and position data from 2018 onwards; older seasons come as results only, through Jolpica-F1 (the successor to Ergast). [FastF1 data reference](https://docs.fastf1.dev/data_reference/index.html)

- [ ] Fetch layer over FastF1: laps, stints, tyre compounds, weather, track status, race control messages.
- [ ] Fetch layer over Jolpica-F1 ([FastF1 interface](https://docs.fastf1.dev/api_reference/jolpica.html)): results, grids, pit stops, finishing status, standings.
- [ ] Respect rate limits: FastF1's cache, retry with backoff, incremental sync that skips what is already stored.
- [ ] Storage: Parquet partitioned by season / event / session, with DuckDB views on top.
- [ ] Canonical keys for drivers, teams and events, plus a team lineage table so rebrands stay one entity (for example Sauber → Audi).
- [ ] Validation with pandera: plausible lap-time ranges, no duplicate laps, stints that join up.
- [ ] A `clean_lap` flag that excludes in/out laps, lap 1, and laps under safety car, VSC or red flag.
- [ ] CLI: `f1sim data sync --season 2025` and a per-session data-quality report.
- [ ] Tests run on recorded fixtures, so CI never touches the network.

**Gate:** a full sync of 2018–2026 is reproducible from scratch and passes validation.

## Phase 2 — Baselines and evaluation harness (weeks 5–6)

Goal: know what "good" looks like before building anything clever. Every later model has to beat these numbers to be merged.

- [ ] Fix the prediction target: for each race, a driver × finishing-position probability matrix, plus P(DNF). Win, podium and points probabilities fall out of it.
- [ ] Property tests on the matrix: every row and column sums to 1.
- [ ] Walk-forward backtester. For race *r*, the model sees only data up to its prediction cutoff (after qualifying). Add a leakage test that fails if any later data is read.
- [ ] Metrics: Brier score (win, podium, top 10), log loss, ranked probability score over positions, Spearman correlation with the actual order, calibration curves.
- [ ] Baselines: (1) grid order with noise, (2) an Elo-style rating for driver and team, (3) a Plackett–Luce ranking model on recent results.
- [ ] Optional market benchmark: if you can source historical closing odds, remove the bookmaker margin and score them the same way.
- [ ] Report generator: one markdown/HTML backtest report per season.

**Gate:** baseline scores for 2022–2025 are recorded in the repo. The Plackett–Luce score becomes the bar to clear.

## Phase 3 — Pace model (weeks 7–10)

Goal: estimate each car-and-driver's underlying race pace, with honest uncertainty, and update it through a race weekend. This is the core of the project.

1. **Fuel and track correction.** Estimate the fuel effect and track evolution per circuit from the data, so laps from different points in a race become comparable.
2. **Frequentist first.** A mixed-effects regression (statsmodels): lap time \~ team + driver + compound + tyre age + fuel + track evolution. Fast to fit, easy to debug, and a reference for step 3.
3. **Bayesian hierarchical model** (PyMC, or NumPyro if speed matters):
   - Team pace per season that drifts from race to race (a random walk), because cars develop.
   - Driver effect measured against the teammate.
   - Circuit-type interactions (high vs low downforce clusters).
   - Qualifying pace and race pace as separate but correlated quantities.
4. **Weekend updating.** Priors from earlier races, then posteriors updated after each practice session, sprint and qualifying. Long-run practice stints are the key signal.

Checks: R-hat and effective sample size, posterior predictive checks, error on held-out race laps.

**Gate:** posterior predictive checks pass on held-out races, and ranking drivers by predicted pace alone beats the Phase 2 baseline.

## Phase 4 — Tyres and pit strategy (weeks 11–13)

Goal: model how each compound degrades at each circuit, and which strategies teams are likely to run.

- [ ] Degradation model per compound × circuit: lap-time loss against tyre age, with a spline so it can capture a late "cliff". Pool it hierarchically across circuits so new or rarely used tracks borrow strength.
- [ ] Pit loss per circuit, measured from timing data (pit-in plus pit-out laps against a normal lap), plus a distribution for stationary time.
- [ ] Season rules in config (`configs/rules/2026.yaml`): compound-use rules, race distance, sprint format. Rules change; code should not have to.
- [ ] Strategy generator: enumerate legal one-, two- and three-stop plans and their expected race time.
- [ ] Strategy chooser: a probability over plans for each driver, mixing the time-optimal plan with each team's historical tendencies.

**Gate:** the degradation model predicts held-out stint lap times well, and actual stop laps fall inside the predicted pit windows at a rate you measure and report.

## Phase 5 — Monte Carlo race simulator (weeks 14–16)

Goal: run each race thousands of times, lap by lap, and turn the outcomes into the probability matrix from Phase 2. Finishing this phase is the MVP.

- [ ] Race state per lap: order, gaps, tyre compound and age, fuel, pit status.
- [ ] Each lap: sample a lap time from pace (Phase 3) + degradation (Phase 4) + noise, then update the gaps.
- [ ] Overtaking model: the chance of a pass given the pace gap and how hard the circuit is to pass on, estimated from historical position changes; a dirty-air penalty when following closely.
- [ ] Pit stops executed from each driver's sampled strategy.
- [ ] Vectorised with NumPy, so N races run as arrays at once. Set a speed target (for example 10,000 runs of one race in under a minute on a laptop) and reach for Numba only if you miss it.
- [ ] Seeded and reproducible. Property tests: the result is always a valid ordering, gaps never go negative, every car finishes or is classified DNF.

**Gate (MVP):** backtested on 2022–2025, the simulator beats the Plackett–Luce baseline on ranked probability score. Tag **v0.1**.

## Phase 6 — Race events (weeks 17–19)

Goal: add the randomness that decides real races, then prove each piece earns its place.

- [ ] **Safety car and VSC:** a per-circuit hazard model from track-status history. In the simulator, a safety car bunches the field and makes stops cheaper.
- [ ] **Reliability and incidents:** DNF hazard per team per season, and crash rate per driver, both with Bayesian shrinkage so a single bad weekend doesn't dominate.
- [ ] **Lap 1:** position changes from the start, per-driver launch performance, and the chance of a first-corner incident.
- [ ] **Weather:** rain probability as an input taken from the forecast at prediction time, switching the pace model to an intermediate/wet regime. Keep it simple at first.
- [ ] **Ablation study:** turn each component off in turn and record the change in backtest score. Publish the table in the docs.

**Gate:** no component is kept if it makes the backtest worse, and the ablation table is published.

## Phase 7 — Regulation-reset study (weeks 20–22)

Goal: answer the project's headline question. When a regulation change makes history unreliable, how many races does the model need to become accurate again, and which approach gets there fastest?

- [ ] Natural experiments: 2022 (ground-effect cars) and 2026 (new power units and aero) at full lap-level detail. 2014 (hybrid era) at results level only, because lap timing starts in 2018.
- [ ] Compare ways of handling the reset:
  - keep the full history as is
  - weight older races less (exponential decay)
  - reset team priors to wide distributions
  - informed priors from pre-season testing
- [ ] Plot a "recovery curve" for each: score by race number after the reset, against the pre-reset level.
- [ ] Write it up as a reproducible notebook, then as a blog post.

**Gate:** the notebook reruns end to end from a clean environment, and the findings are written up.

## Phase 8 — Delivery (weeks 23–24)

Goal: predictions publish themselves every race weekend, with no manual steps.

- [ ] CLI: `f1sim predict`, `f1sim backtest`, `f1sim report`.
- [ ] FastAPI service (`/predictions/{season}/{round}`) in a Docker image.
- [ ] Scheduled GitHub Actions workflow: after qualifying, sync data, run the prediction, commit the JSON and publish it. After the race, score it.
- [ ] Static site on GitHub Pages: a page per race prediction, a season scoreboard, a live calibration chart.
- [ ] Provenance on every published prediction: git commit, model version, data cutoff time. This is what makes the track record trustworthy.

**Gate:** one full race weekend runs end to end without you touching anything.

## Phase 9 — Open-source launch (weeks 25–26)

Goal: v1.0 is public and usable by others, with live predictions running from the first round of the 2027 season so the track record covers a full year.

- [ ] Docs site (mkdocs-material): quickstart, how the model works, how to add a component, the reg-reset findings.
- [ ] Publish the package to PyPI.
- [ ] Label "good first issue" tasks and open roadmap issues for the stretch goals below.
- [ ] A clear disclaimer that the project is unofficial and not affiliated with Formula 1.
- [ ] Launch post built around the methodology and the reg-reset study; share it with the FastF1 community and F1 data-analysis forums.

**Gate:** v1.0 is on PyPI and automated predictions are live for the 2027 opener.

## Risks, definition of done, stretch goals

| Risk | What it would cost | Mitigation |
| --- | --- | --- |
| Data leakage in backtests | Scores look great and mean nothing | Strict cutoff in the backtester plus a test that fails on any later data |
| FastF1 or Jolpica outage or API change | Missed live predictions | Pin versions, cache everything, FastF1's live-timing recorder as a fallback |
| Data licensing | Takedown or reputational damage | Never redistribute raw timing data; publish only derived predictions |
| Bayesian model too slow | Weekend updates miss the race | NumPyro/JAX, variational inference, cached posteriors |
| Small samples (about 24 races a season) | Overconfident forecasts | Hierarchical pooling; watch the calibration curves |
| Scope creep | Project never ships | Each phase has a gate; anything else goes to the stretch list |

**Done means:**

- [ ] v1.0 on PyPI with docs, CI green, typed and tested.
- [ ] Backtests for 2022–2025 that beat every baseline, published with the ablation table.
- [ ] The reg-reset study written up.
- [ ] Automated, timestamped predictions running for a live season.

**Stretch goals (after v1.0):** in-race predictions from live timing; a pit-strategy optimiser for a given race state; sprint-race support; what-if mode (put driver A in car B).
