# Event-component ablation (P6-09)

Seasons 2021, 2022, snapshot 10. The fully-featured simulator scores RPS 0.1265; each row switches one component off. A negative delta means the component was making the backtest worse.

Verdict: drop safety_car (removing them improves RPS).

| Component off | RPS without | Delta RPS | Verdict |
|---|---:|---:|---|
| Safety car and VSC (P6-01) | 0.1255 | -0.0010 | harmful |
| Reliability (P6-03, P6-04) | 0.1266 | +0.0001 | kept |
| Lap 1 (P6-05) | 0.1264 | -0.0001 | kept |
| In-race penalties (P6-06) | 0.1265 | -0.0000 | kept |
| Traffic and defending (P6-08) | 0.1264 | -0.0001 | kept |
| Rain regime (P6-07) | 0.1265 | +0.0000 | kept |

Deltas within ±1e-03 RPS are inside the run-to-run noise band, so a
component flagged at that edge is a marginal finding rather than a clear loss.
