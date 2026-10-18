# Schemas

JSON Schemas shared by the backend and the frontend. The backend validates every file it publishes against its schema before writing it (P8-02, ACID consistency); the frontend generates its TypeScript types from them (`pnpm gen:types`), and CI fails if the committed types are stale (P8-13).

| Schema | Describes |
|---|---|
| `prediction-record.v1.schema.json` | One race's forecast with provenance, and its score after the race |
| `predictions-index.v1.schema.json` | `predictions/index.json`: every published race, the season scoreboard and calibration |

Breaking changes get a new versioned file (`prediction-record.v2.schema.json`); old records keep validating against the version they declare.
