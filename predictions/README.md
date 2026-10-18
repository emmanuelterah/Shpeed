# Published predictions

One JSON file per race, `predictions/<season>/<round>.json`, written by `f1sim predict` (run
by the race-weekend workflow) and validated against
`schemas/prediction-record.v1.schema.json`. `index.json` lists every race with its score and
the calibration chart's data; it is rebuilt from the records on every publish and validated
against `schemas/predictions-index.v1.schema.json`.

Before lights out a record can be replaced by a forecast built on newer data; after lights
out it is locked and the scoring run only fills its `score` field (ADR-0011). Git history
keeps every version.
