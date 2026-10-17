# 0008. Race simulator: cumulative-time state and an ordered LapStep pipeline

- Status: Accepted
- Date: 2026-10-17
- Roadmap: P5-01 to P5-12

## Context

Phase 5 turns the pace, tyre and strategy models into a forecast by running each race lap by lap many times. The engine has to be fast (10,000 runs of a race in under a minute on a laptop), reproducible from a seed, open to new race mechanics without edits (Phase 6 adds safety cars, reliability and lap 1), and honest about track position: a faster car has to get past the car in front, not just post a better lap.

## Decision

**State (P5-01).** `RaceState` holds NumPy arrays of shape `(n_sims, n_cars)`. A car's place on the road is its cumulative race time `race_time_s`; the running order and gaps are derived from it by `update_order`, so a step that changes a time only has to keep `race_time_s` and `lap_time_s` in step. Their difference is where the car began the lap (`lap_start_times`), which gives every step the start-of-lap order without storing it twice. Retired cars carry an infinite time and are ranked behind every runner.

**Pipeline (P5-02).** `run_race` advances the lap counter and applies an ordered list of `LapStep` objects. The simulator builds four, in this order:

1. `LapTimeStep` (P5-03): pace + compound offset + tyre loss at the current age + fuel + lap noise. Pace is drawn once per run from the pace posterior (mean and covariance from the Kalman filter, plus race-day order noise); tyre loss curves are drawn once per run from the degradation curves and their standard errors (`TyreModel.sample_loss`), so model uncertainty is felt in the forecast (D4).
2. `DirtyAirStep` (P5-05): a car that starts the lap within 1 s of the car ahead loses up to 0.4 s, tapering to zero at the edge of the window. It is a separate step so the ablation runner can switch it off alone.
3. `PitStopStep` (P5-07): each run samples a plan per car from the strategy chooser; on a stop lap the car pays the circuit's pit loss plus a per-stop draw of its team's stop-duration deviation (slow-stop tail included) and takes fresh tyres.
4. `OvertakeStep` (P5-04, P5-06): walks the start-of-lap order front to back. A car that would end the lap ahead of, or within 0.3 s of, the car in front attempts a pass; it succeeds with the fitted pass probability for its pace advantage, the circuit and the season's aid (DRS to 2025, Overtake Mode from 2026, read from `configs/rules`). A failure holds it 0.3 s behind; a success gains exactly one place. Cars that pitted or retired take no part.

**Pass model (P5-04).** Logistic regression on adjacent cars in the end-of-lap running order (the timing data's `position`), within 1 s at the line (lap start plus lap time): a swap by the end of the next lap is a pass. Only green-flag laps on which neither car entered or left the pits count, since stops and safety cars reorder the field without a pass. The covariate is the pair's race-median clean-lap pace difference, because a single lap's difference regresses to the mean once pairs are conditioned on being close and biases the slope toward zero or below. Each circuit gets a clipped log-odds offset; the aid enters as a fixed bonus because position changes alone cannot separate it from pace. On 2021–2022 this gives Monaco the floor offset (−2.0) and Bahrain, Spa, Interlagos and Austin the easiest passing.

**Starting state (P5-08).** `apply_grid_penalties` turns qualifying into the grid (places dropped, back-of-grid, pit-lane starts) as a pure function, and `starting_state` gives slot *k* a 0.3 s offset per slot, with pit-lane starters a further 5 s behind the last slot. The grid therefore keeps mattering after lap 1 through track position. Sources do not carry penalties (ADR-0005), so backtests pass none; the live path can.

**Strategy per race.** The race's legal plans are generated once at the field-average stop cost and the chooser turns them into a distribution per team from its tendencies; each team's own stop times are still sampled when its stops are executed. A team's stop-time deviation is a fraction of a second against a pit loss of about twenty, so generating per team barely moved the optimum while costing most of a forecast's run time. The generator's dynamic program and stint-cost table were also vectorised (identical plans, about 3.6 times faster).

**Run count.** The simulator takes its runs per race from `Settings.n_sims` (`F1SIM_N_SIMS`, 10,000 by default) through `ModelOptions`; the baselines keep their own count.

**Track position calibration.** Measured against real 2021 races (places gained per lap on which no car stopped, and the grid-to-finish rank correlation), the first engine passed 1.8 times too often and kept less of the grid than real races. Three simulator settings in `SimulatorContext`, each chosen on 2021 only, fix this:

- `pass_offset` = −1.4 log-odds on every pass attempt. The pass model is fitted on pairs within 1 s at the line, while the engine attempts a pass whenever a car would end the lap within 0.3 s of the one ahead, which happens more often. With it, simulated passing is 0.56 places per clean lap, as in 2021.
- `chooser_temperature_s` = 2 s for the race-day strategy draw (the pit-window report keeps 8 s). The engine charges a sampled plan's full time penalty, but a team that runs a different plan does so because it is about as fast for them; at 8 s the strategy draw alone added seconds of random spread and scrambled the grid.
- `race_noise_pct` = 0.3% of lap time per car, instead of the pace model's `order_noise` (0.45%). That value stands in for everything that reorders a race when cars are only ranked by pace; the simulator models strategy, stops and passes itself, so reusing it counted that randomness twice.

Together they match the 2021 grid-to-finish correlation (0.757 against 0.754 real) and improve RPS on 2021 from 0.1234 to 0.1208 and, out of sample, on 2022 from 0.1326 to 0.1307 (2,000 runs per race). Judging passes on noise-free pace instead of each lap's time made no difference and was not adopted.

**Reproducibility (P5-09).** One seeded generator drives every draw in a fixed order, so the same seed, snapshot and model version give a bit-identical matrix.

**Aggregation (P5-12).** Finishing orders across runs become a `PositionMatrix` with P(DNF), mixed with 0.1% of the uniform forecast like the baselines, so log loss stays finite.

**Fallbacks.** When there is too little history the pass model falls back to `default_overtaking()` and the tyre model to `default_tyre_model()` (linear prior wear, the pit-loss prior and default stop times), so every early-season forecast still runs and every `Predictor` passes the same contract suite.

## Alternatives considered

- **Positions as the state, with pass probabilities per adjacent pair.** Simpler, but gaps, dirty air, pit-stop undercuts and safety-car bunching all need time, so the state would have to carry it anyway.
- **Letting lap times alone set the order.** Free passing: a car two tenths faster gets by on its first lap behind, which removes track position and makes the grid irrelevant after lap 1.
- **Unlimited passes per lap.** A fast car starting from the pit lane would clear the field in a lap or two; one place per lap matches how passes actually happen and leaves multi-car moves to the lap 1 model (P6-05).
- **Dirty air inside the overtake step.** Fewer classes, but the ablation could not separate the two mechanics.
- **Generating plans per team.** Exact for each team's stop cost, but most of the forecast's run time for a difference well inside the chooser's spread.
- **Ordering by summed lap times over the kept laps.** Any dropped lap (safety car, inaccurate timing, a stop) removed a whole lap from one driver's total and created phantom swaps; with pit swaps also counted, the fit put a 29% per-lap pass chance on an equal car and made Monaco only slightly harder than average.
- **Numba.** Not needed: 10,000 runs of a 22-car, 57-lap race take about 5 s with NumPy alone.

## Consequences

- Retirements are drawn before the start at the league DNF rate, so a retiring car never influences the race. Phase 6 replaces this with team reliability and driver crash hazards that strike mid-race (P6-03, P6-04).
- The pass covariate is race-median pace while the step sees each lap's pace plus noise, so pass rates are slightly overstated for cars whose noise happened to favour them on that lap.
- Lapping is not modelled: cars are compared at equal lap counts, so a leader never interacts with a backmarker. Traffic is a Phase 6 candidate (P6-08).
- The simulator reads `configs/rules/<season>.yaml`, which lives in the repository rather than the wheel; outside a checkout, `F1SIM_RULES_DIR` must point at a rules folder.
- As first built, the simulator scored RPS 0.154 against Plackett–Luce's 0.137 on 2021–2022. The gap was in its race-pace input, not the engine: a wrong backtest cutoff, discarded sessions, outlying qualifying laps and unfitted pace hyperparameters (ADR-0005 amendment, ADR-0006). With those fixed it scores 0.128, the best recorded model (`grid_noise` 0.130, Plackett–Luce 0.137), and on 2022, which the hyperparameters did not see, 0.133 against 0.143, beating Plackett–Luce in 19 of 22 races. The roadmap gate asks for 2022–2025, which needs 2023–2025 laps in the store.
- Real cars passed more in 2022 (0.84 places per clean lap, after the ground-effect rules) than in 2021 (0.56), so a single `pass_offset` under-passes in 2022; a per-era calibration is a candidate once 2023–2025 laps are in the store.
- Lap 1 barely reorders the field (simulated grid-to-lap-1 rank correlation 0.997 against 0.88 real); launches and first-corner incidents belong to the lap 1 model (P6-05).
- The three calibrated settings are constants chosen by hand-run experiments, not fitted by a command; recalibrating them as data grows means rerunning those comparisons.
- Every new mechanic is a new step; the engine and aggregation stay unchanged.
