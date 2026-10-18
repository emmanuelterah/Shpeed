# 0009. Race events: fitted hazards, a neutralisation level and an ablation gate

- Status: Accepted
- Date: 2026-10-18
- Roadmap: P6-01 to P6-09

## Context

Phase 5's simulator reorders the field with pace, tyres, strategy and passes, and retires cars before the start at the league DNF rate. Real races are decided as much by events as by pace: a safety car bunches the field and makes a cheap stop, a red flag hands out free tyres, and a car fails on lap 3 rather than before the lights. Phase 6 adds those events, but its gate is strict — **no component is kept if it makes the backtest worse** — so each mechanic has to be both plausible and switchable-off on its own.

## Decision

**Hazards are fitted with shrinkage (P6-01 to P6-06).** `f1sim.adapters.models.hazards` estimates every event from the stored history, each thin sample shrunk toward the league average (the same pooling as the pace and tyre models), with a prior so an early-season forecast still runs:

- **Safety car and VSC (P6-01).** FastF1 stores a lap's `track_status` as the codes that applied during it concatenated into one string (`"1264"`), so `_race_statuses` reads each lap as the set of digits and a lap is neutralised when *any* of them is `4` (safety car) or `6`/`7` (VSC). The transitions into a neutralised lap give deployments per race lap, pooled per circuit; `duration_laps` is the mean neutralised length. A lap reaching `5` is a red flag (P6-02), whose per-race rate is turned into a per-lap chance by the race distance.
- **Reliability and crashes (P6-03, P6-04).** A result with no finishing position is a retirement; its `status` text separates mechanical failures from accidents. Mechanical hazards are pooled per team and accident hazards per driver, both shrunk toward the field.
- **Lap 1 (P6-05).** A driver's launch is the places gained from the grid to the end-of-lap-1 position, shrunk toward zero; the first-corner incident rate is the share of starters eliminated on lap 1.
- **Penalties (P6-06).** The severity lives in the race-control feed, which is not in the model's read set, so the in-race penalty is a fixed prior rather than a fit.

**A neutralisation is a per-simulation level, not per car.** `RaceState.safety_car` is a `(n_sims,)` integer on the state (0 none, 1 VSC, 2 safety car, 3 red flag) and `RaceState.neutralised` broadcasts it to `(n_sims, n_cars)`, so every step sees the same flag with no wiring and a Phase 5 pipeline (where it is `None`) still runs. `SafetyCarStep` draws and times each neutralisation and writes the level each lap:

- a **full safety car** bunches the running cars to `SC_GAP_S` apart behind the leader and makes stops cheaper by `SC_PIT_DISCOUNT_S` (read by `PitStopStep`);
- a **VSC** preserves the gaps, so it does not bunch, and makes stops cheaper by `VSC_PIT_DISCOUNT_S`;
- a **red flag** lasts one lap: cars take a free tyre change (`tyre_age` reset to zero) and line up at `RESTART_GAP_S`, then the **shared lap-1 launch** (`apply_launch`, P6-05) scrubs the field again — the standing restart is the lap-1 model, not a second one.

`DirtyAirStep`, `OvertakeStep` and `TrafficStep` suppress their effects while `neutralised`, because no one attacks or defends under a safety car.

**Retirements move into the race (P6-03, P6-04).** `ReliabilityStep` retires each car on a pre-sampled lap (its infinite cumulative time ranks it behind every runner from then on), instead of the Phase 5 pre-start draw. When `reliability` is disabled, the simulator falls back to the Phase 5 pre-start draw at the league rate, so the ablation has something to compare against.

**The should-haves are coarse (P6-06 to P6-08).** `PenaltyStep` serves a pre-sampled five- or ten-second penalty on a sampled lap. Rain is an **input** (`SimulatorContext.p_rain`), not a fitted hazard: each run is drawn dry or wet and a wet run's laps carry `WET_EXTRA_S` more time, so the forecast is a mix of dry and wet races and the spread already reflects the chance of a downpour. `TrafficStep` charges the defender a small loss when a car is within 1.5 s behind — the defending half of "traffic", since lapping is still unmodelled.

**Ablation is the gate (P6-09).** `f1sim.reports.ablation.run_ablation` backtests the fully-featured simulator, then once per component with it switched off, sharing one seed so the comparison is on identical races. A component whose absence *lowers* RPS was costing accuracy and is flagged harmful; the table is published to `docs/ablation.md` by `f1sim report ablation`, and `--check` exits non-zero when a kept component is harming the backtest. Components are switched off through `ModelOptions.disabled` (a `frozenset`), which reaches the simulator's `_steps`, so no engine change is needed to add or drop one.

## Alternatives considered

- **A per-car `safety_car` array.** Symmetric with the other arrays, but a neutralisation applies to the whole field, so a scalar per simulation is simpler and cannot disagree car to car.
- **Fitting the red-flag and penalty rates from `race_control`.** The only source with severity and type, but it is not in the model's read set and adding it would widen the store footprint for a prior that is already reasonable.
- **Fitting rain from the weather table.** More faithful, but the store says little about grip and the roadmap marks rain as the first should-have to cut; an input keeps the model honest and the feature cheap.
- **A separate red-flag restart model.** Duplicates the lap-1 launch; reusing `apply_launch` keeps one launch implementation and lets P6-02 depend on P6-05 rather than fork it.
- **Automatically dropping a harmful component.** The gate should be a decision, not a side effect; the runner publishes the table and `--check` fails, and a human drops the component.
- **Backmarker lapping inside the traffic step.** The engine compares cars at equal lap counts, so a real lapping model needs lap counts first; only the defending half is built.

## Consequences

- The Phase 5 "retirements drawn before the start" path is deliberately kept (behind `disabled={"reliability"}`), so the ablation can measure the change and old forecasts stay reproducible.
- `track_status` is read as concatenated digits, so an adapter that stores one code per lap and one that concatenates them are both handled.
- Neutralisation reaches three Phase 5 steps through one state property; a new step that should ignore a safety car has to opt in to `state.neutralised`.
- The hazards shrink toward their priors, so a fitted red-flag rate is never exactly zero; the ablation and reports compare rates, not exact zeros.
- The should-haves are intentionally coarse: penalties are a flat prior, rain is one extra-seconds figure with no intermediates or drying line, and traffic is defending only. They are the first things cut, and the ablation table is what says whether they stay.
- Phase 6's gate evidence is `docs/ablation.md`, produced by `f1sim report ablation --check`; until 2023–2025 laps are in the store the table covers the seasons that are synced.
