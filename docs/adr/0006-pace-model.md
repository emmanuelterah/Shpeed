# 0006. Pace model: corrected session pace and a Gaussian state-space model

- Status: Accepted
- Date: 2026-10-13
- Roadmap: P3-01 to P3-10

## Context

The simulator needs each car and driver's underlying pace, with honest uncertainty, updated through a race weekend. Raw lap times mix pace with fuel load, tyre wear and track evolution, and every session is a different mix of these.

## Decision

**Observations.** Each session is reduced to one number per driver: corrected pace in percent of the session's median driver (0.5 means half a percent slower, about 0.45 s on a 90 s lap). Percent puts all circuits on one scale.

- *Races and sprints:* least squares on clean dry laps, `lap time ~ driver + compound + compound x tyre age + lap number`. The lap-number slope is the fuel effect; tyre age resets at each stop while the lap count does not, which separates the two. Each race's estimate is pooled with earlier races at the same circuit and a global prior (0.06 ± 0.03 s per lap).
- *Qualifying and sprint qualifying:* push laps (within 3% of the driver's best), `lap time ~ driver + compound + minutes into session`. The minutes slope is track evolution, pooled the same way.
- *Practice:* long runs (stints of five or more clean laps), corrected with the circuit's pooled fuel effect within each stint. They are the main pre-race signal of race pace.

Sessions are processed in time order and pool only with earlier sessions, so no observation depends on anything after it.

**Model.** With all quantities in percent:

`pace = team + supplier + team x cluster + (quali: team quali offset + driver quali | race: driver race) + session level + noise`

- Team pace (by lineage) is a random walk between races, with a larger step at each season start.
- Supplier pace changes only between seasons; engines are largely frozen within a season.
- Team x circuit-cluster effects capture cars that suit slow or fast tracks. Clusters are average qualifying speed bands (below 210 km/h, 210–235, above 235), from lap lengths in `f1sim/resources/circuits.yaml`.
- Driver qualifying and race effects are correlated and partially pooled across seasons: a new season adds variance instead of starting over. A driver is identified against their teammate.
- The session level absorbs each session's reference and is integrated out.
- Noise is separate for qualifying, races, sprints and practice.

Once the hyperparameters are fixed, the model is linear and Gaussian, so the posterior over all latent effects is exact and a Kalman filter updates it one session at a time.

**Weekend updates.** After each race weekend, the posterior and the fuel and evolution history are saved, keyed by model version, weekend and snapshot, with a fingerprint of the laps they came from. The next update resumes from that state if the fingerprint still matches, and only processes the new sessions.

**Race forecast.** The `pace` predictor samples race pace from the posterior, adds race-day noise (`order_noise`, chosen by backtest), and ranks. Retirements use the historical rate until Phase 6 adds hazards.

## Alternatives considered

- **Separate fuel and track evolution in races.** Both move with the lap count and cannot be told apart from timing data alone. The race slope absorbs both. Evolution is estimated where it can be, in qualifying and practice.
- **Raw-lap model (every lap an observation).** About 50 times more data for the sampler, little gain in information about pace, and the corrections are easier to check one session at a time. The statsmodels mixed-effects model (`f1sim pace reference`) works on raw laps and serves as the cross-check.
- **A clustering algorithm for circuits.** Speed bands from lap length and qualifying speed are stable across backtests and easy to explain. A learned clustering would move as data arrives.

## Consequences

- Fuel loads in practice are unknown, so practice observations are noisier; the model learns how much noisier.
- Grid penalties and mechanical problems in a session show up as noise in that session's observation.
- The engine clipping feature from telemetry (P3-08) is not built: lap data has no energy-deployment signal. The supplier effect captures engine differences between teams.
- Circuit lengths are approximate and only used for clustering; a wrong length can move a circuit one cluster at most.
