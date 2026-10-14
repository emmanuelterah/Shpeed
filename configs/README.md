# Configuration

Rules and reference data that change by season live here, so code does not have to (open/closed principle).

- `rules/` one YAML file per season, validated by a schema at load time (P4-06)
- Circuit lengths and speed-cluster bands (P3-06) and engine suppliers by team and season
  (P3-08) ship inside the package, in `src/f1sim/resources/`, so they load after
  `pip install`.
- `pace_hyper.json` is written by `f1sim pace fit`: the fitted pace-model hyperparameters,
  with diagnostics and provenance. Commit it so forecasts and backtests are reproducible.
- `mappings/` canonical driver, team and event IDs and the team lineage table (P1-05, P1-06)
