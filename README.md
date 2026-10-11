# Shpeed

An open-source F1 race simulator. Shpeed fetches public timing data, fits Bayesian models of pace, tyres, strategy and race events, simulates each race lap by lap thousands of times, and publishes a probability for every driver finishing in every position.

> **Status:** Phase 0 (foundations). See [ROADMAP.md](ROADMAP.md) for what's built and what's next.
> Unofficial project, not affiliated with Formula 1.

## Repository layout

```
Shpeed/
├── src/f1sim/            Python package (simulator, models, CLI, API)
│   ├── core/             pure domain logic: types, ports, engine, scoring, use cases
│   ├── adapters/         I/O and heavy libraries behind the ports
│   ├── api/              read-only FastAPI service
│   ├── reports/          backtest report generation
│   └── cli.py            composition root
├── frontend/             web app (React + TypeScript + Vite), Apple-HIG design system
├── schemas/              JSON Schemas shared by backend and frontend
├── configs/              season rules, circuits, ID mappings
├── tests/                unit · property · contract · integration · perf
├── benchmarks/           recorded backtest scores per model
├── predictions/          published prediction records (generated)
├── notebooks/            research write-ups
├── docs/                 ADRs, architecture, planning, guides
├── docker/               api and web Dockerfiles, nginx config
├── compose.yaml          local stack (compose.dev.yaml adds hot reload)
├── scripts/              developer helpers
└── data/                 local DuckLake store (git-ignored)
```

Read [ARCHITECTURE.md](ARCHITECTURE.md) for how the parts fit, and `docs/architecture/Shpeed Architecture.html` for the full diagrams.

## Run with Docker

The quickest way to run everything. Needs only [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine with Compose v2.24+).

```bash
docker compose up --build
```

| Service | URL | What it is |
|---|---|---|
| web | http://localhost:8080 | The frontend, served by nginx; `/api` is proxied to the API |
| api | http://localhost:8000/docs | Read-only API over `predictions/`, with interactive docs |

For development with hot reload on both sides:

```bash
docker compose -f compose.yaml -f compose.dev.yaml up --build
```

The frontend is still on http://localhost:8080, now served by the Vite dev server with hot reload, and the API reloads whenever files in `src/` change. Stop with `Ctrl+C`, clean up with `docker compose down`.

## Quickstart without Docker

Backend (Python 3.12, [uv](https://docs.astral.sh/uv/)):

```bash
uv sync                      # create the environment from the lockfile
uv run pytest                # tests
uv run ruff check . && uv run pyright && uv run lint-imports
uv run f1sim --help
```

Frontend (Node 20+, pnpm):

```bash
cd frontend
pnpm install
pnpm dev                     # http://localhost:5173
pnpm lint && pnpm typecheck && pnpm test
```

## Contributing

Every piece of work maps to a feature ID in [ROADMAP.md](ROADMAP.md) (for example `P1-02`). See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and definition of done.

## Licence

MIT. See [LICENSE](LICENSE).
