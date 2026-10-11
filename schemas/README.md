# Schemas

JSON Schemas shared by the backend and the frontend. The backend validates every prediction record against the schema before writing it (P8-02, ACID consistency); the frontend generates its TypeScript types from it (`pnpm gen:types`).

Breaking changes get a new versioned file (`prediction-record.v2.schema.json`); old records keep validating against the version they declare.
