# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Repository scaffold: package layout, ports, CI, ADRs 0001–0004, frontend workspace (P0-02, P0-03, P0-12, P0-18).
- Docker Compose stack: `api` (read-only FastAPI over `predictions/`) and `web` (nginx serving the frontend, proxying `/api`), plus `compose.dev.yaml` with hot reload; CI builds and smoke-tests it (P8-04, P8-05).
- Read-only API routes: `/health`, `/predictions`, `/predictions/latest`, `/predictions/{season}/{round}` (P8-04).
