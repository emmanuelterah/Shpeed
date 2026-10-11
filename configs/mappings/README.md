# Mappings

## Canonical keys (P1-05)

Shpeed uses Jolpica-F1 (Ergast) identifiers everywhere:

| Entity | Key | Example |
|---|---|---|
| Driver | `driver_id` (Jolpica `driverId`) | `max_verstappen` |
| Team | `team_id` (Jolpica `constructorId`) | `red_bull` |
| Event | `(season, round)` | `(2024, 1)` |
| Session | `(season, round, session)` with session in FP1, FP2, FP3, SQ, S, Q, R | `(2024, 1, "R")` |

FastF1 session results carry the same `DriverId` and `TeamId`, so laps join to results
without a lookup table. If FastF1 lacks an ID for a driver, the lap is stored with
`driver_id = "code:<abbreviation>"` and the data-quality report flags it under `id_fallbacks`.

## Team lineage (P1-06)

The table lives inside the package, at `src/f1sim/resources/team_lineage.yaml`, so it ships
with the wheel and loads the same way after `pip install f1sim`. It groups constructor IDs that are one team under different names, such as
`sauber`, `alfa` and `audi`. Models use `TeamLineage.lineage_of(team_id)` to carry history
across rebrands; the stored data keeps the original constructor ID.
