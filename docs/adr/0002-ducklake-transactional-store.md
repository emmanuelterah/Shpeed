# 0002. DuckLake as the transactional data store

- Status: Accepted
- Date: 2026-10-11
- Roadmap: P1-09 to P1-14, P1-19

## Context

Parquet files on their own are not transactional: an interrupted sync can leave a half-written season, and a prediction running during a sync could read a mix of old and new data. Backtests and published forecasts must be reproducible against exactly the data they used.

## Decision

Store all tables in DuckLake (DuckDB's lakehouse format, v1.0 released April 2026): a SQL catalog in a local DuckDB file over immutable Parquet data files.

- Each sync is one transaction; validation runs before COMMIT and failure rolls back (atomicity, consistency).
- Readers pin a snapshot with `AT (VERSION => n)`; a prediction never sees a sync in progress (isolation).
- Committed snapshots and immutable files survive crashes; `f1sim data verify` checks them (durability).
- A file lock enforces one writer, since a DuckDB-file catalog is single-writer.

## Alternatives considered

- **Plain Parquet + hand-rolled manifest:** possible, but reimplements what DuckLake provides and is easy to get subtly wrong.
- **Delta Lake / Apache Iceberg:** mature, but heavier tooling for a single-machine project.
- **A single DuckDB database file:** transactional, but loses the open Parquet layout and snapshot time travel.

## Consequences

- Every prediction record stores its `snapshot_id`; snapshot retention must never expire a cited snapshot.
- The store sits behind `DataWriter` / `DataReader` ports, so it can be replaced without touching the core.
