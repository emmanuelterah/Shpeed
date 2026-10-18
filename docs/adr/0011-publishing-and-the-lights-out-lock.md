# 0011. Publishing and the lights-out lock

- Status: Accepted
- Date: 2026-10-18
- Roadmap: P8-02, P8-03, P8-06, P8-07, P8-10

## Context

Phase 8 publishes a forecast for every race automatically and scores it afterwards. A
published forecast is only worth something if readers can trust that it was made before the
race and never quietly changed. Three forces pull against each other:

- **Grids change after qualifying.** Penalties are often announced hours after the session,
  so the first forecast on Saturday night can be stale by Sunday morning.
- **The schedule must be idempotent.** The workflow fires every two hours across the
  weekend so it catches qualifying in every time zone. Each run must be a no-op unless
  something real changed, or it commits on every run. A naive "same record?" check fails
  this: the record stores the git commit, and every publish moves `HEAD`.
- **Forecasts must be frozen once the race starts.** Anything else lets hindsight leak into
  the scoreboard.

## Decision

The core use case `publish_record` owns the policy; adapters only write files and commit.

1. **Nothing published yet:** write the record, unless lights out has passed, in which case
   refuse (`RecordLockedError`). A forecast made after the start is not a forecast.
2. **Same forecast already published** (`PredictionRecord.same_forecast`, which compares
   everything except the score and the provenance fields `git_commit` and `snapshot_id`):
   keep the published file, adding the score if the new record has one. A re-run that
   produces identical probabilities is a byte-for-byte no-op, whatever commit it ran from.
3. **Different forecast already published:** before lights out the new one replaces it (new
   data arrived; git history keeps the old version). After lights out, or when the calendar
   has no start time, it is refused.

Lights out is the race's `event_date` in the `events` table (Jolpica's scheduled start, in
UTC). Live forecasts also require a qualifying result, so on a sprint weekend the sprint
classification is never published as the race grid.

Each publish writes the record and a rebuilt `predictions/index.json` through temp file,
fsync and rename, validates both against their JSON Schemas, and commits exactly those two
files in one commit. Scoring stores each driver's official finishing position in the record,
so the scoreboard and the calibration chart in the index can always be rebuilt from the
records alone. Probabilities are stored to eight decimals: six were not enough for a 22-car
grid to pass `PositionMatrix`'s 1e-6 checks when read back for scoring.

## Alternatives considered

- **Immutable from the first publish.** Simplest, but a forecast published before a grid
  penalty could never be corrected, and every later scheduled run would fail on the "changed"
  forecast.
- **Compare whole records, including provenance.** Breaks idempotency: each publish commit
  changes `git_commit`, so the next scheduled run would always publish again.
- **Lock at the official grid's publication.** More precise, but our sources do not record
  when the official grid appears (ADR-0005).
- **Publish without git and serve from object storage.** Loses the free, public audit trail
  that git history gives every forecast.

## Consequences

- A record can change between qualifying and lights out; the git log shows every version and
  the provenance on the final one reproduces it.
- If a weekend's forecast was missed, the scheduled run reports the lock instead of
  publishing late, and the failure alert fires; it cannot be back-filled.
- When the calendar lacks a start time, the first forecast published is final.
- Changing the provenance fields that `same_forecast` ignores is a decision that needs a new
  ADR, since it changes what counts as "the same forecast".
