# 0001. Raw data is fetched by code, never committed

- Status: Accepted
- Date: 2026-10-11
- Roadmap: P0-12

## Context

Lap timing and telemetry come from FastF1 and Jolpica-F1. Redistributing raw timing data carries licensing and takedown risk, and large data in git makes the repo slow to clone.

## Decision

Raw data is fetched by code into the local store under `data/` (git-ignored). The repository contains only code, configuration, recorded test fixtures small enough to review, and derived outputs (predictions, benchmark scores).

## Consequences

- Anyone can reproduce the store with `f1sim data sync`; syncs must be incremental and resilient (P1-04, P1-15).
- Published predictions carry provenance (snapshot ID, data cutoff) instead of the data itself.
- CI uses recorded fixtures and never touches the network (P1-18).
