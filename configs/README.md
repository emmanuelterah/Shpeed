# Configuration

Rules and reference data that change by season live here, so code does not have to (open/closed principle).

- `rules/` one YAML file per season, validated by a schema at load time (P4-06)
- `circuits/` circuit metadata and speed-profile clusters (P3-06)
- `mappings/` canonical driver, team and event IDs and the team lineage table (P1-05, P1-06)
