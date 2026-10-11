# Published predictions

One JSON file per race, `predictions/<season>/<round>.json`, written by the race-weekend workflow and validated against `schemas/prediction-record.v1.schema.json`. A record is immutable after lights out; the scoring run only fills its `score` field.
