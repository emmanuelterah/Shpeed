# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Data dependencies (FastF1, DuckDB, pandera, httpx and others) are now core dependencies, so
  `pip install f1sim` gives a working CLI. The `data` extra is gone; `models` and `api` remain.
- The team lineage table moved into the package (`f1sim/resources/`), so it loads after a
  `pip install`; it previously relied on the repository layout.
- `HttpClient` retry settings are grouped into a `RetryPolicy`.

### Added
- Repository scaffold: package layout, ports, CI, ADRs 0001–0004, frontend workspace (P0-02, P0-03, P0-12, P0-18).
- Docker Compose stack: `api` (read-only FastAPI over `predictions/`) and `web` (nginx serving the frontend, proxying `/api`), plus `compose.dev.yaml` with hot reload; CI builds and smoke-tests it (P8-04, P8-05).
- Phase 1 data ingestion (P1-01 to P1-19):
  - canonical table registry with `available_at` on every row, and pandera validation for every table;
  - `clean_lap` flag;
  - transactional DuckLake store with snapshot-pinned reads, a single-writer lock, `verify` and retention;
  - in-memory test store and a shared store contract suite;
  - resilient HTTP layer (cache, rate limits, backoff with jitter);
  - Jolpica-F1 and FastF1 adapters, canonical Jolpica IDs and the team lineage table;
  - incremental `sync_season` use case;
  - `f1sim data sync | verify | report | gc` and a per-session data-quality report;
  - offline fixtures, and `scripts/record_fixtures.py` to record real ones.
- Code standards enforced in CI and pre-commit: pylint 10/10 (with Google docstring checks),
  ruff pydocstyle (Google convention), 100-character lines in every code file, imports only at
  the top of files, pyright strict, and `twine check` / `check-wheel-contents` on every build.
- `f1sim.api.healthcheck` for container health checks.
- PyPI metadata: SPDX licence, project URLs, keywords, classifiers, single-source version,
  README links rewritten to GitHub on the PyPI page.
- Read-only API routes: `/health`, `/predictions`, `/predictions/latest`, `/predictions/{season}/{round}` (P8-04).
