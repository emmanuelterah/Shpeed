# Contributing to Shpeed

## Workflow

1. Pick a feature from [ROADMAP.md](ROADMAP.md) whose dependencies are done, and set it to *In progress* in the roadmap tracker.
2. Branch from `main`: `feat/P1-02-fastf1-adapter` (or `fix/…`, `docs/…`, `chore/…`).
3. Commit with [Conventional Commits](https://www.conventionalcommits.org/): `feat(data): add FastF1 adapter (P1-02)`.
4. Open a PR titled the same way. The template repeats the definition of done.
5. Squash-merge once CI is green and the PR is reviewed. Mark the feature *Done* and save the PR link in the tracker.

## Definition of done

1. Code merged to `main` through a reviewed PR titled with the feature ID
2. Tests added; CI green (lint, strict types, coverage, import contracts)
3. The feature's "done when" criterion demonstrated
4. Docs or ADR updated if behaviour or a decision changed
5. `CHANGELOG.md` entry; PR link saved on the feature in the tracker

## Architecture rules (checked in CI)

- `f1sim.core` never imports I/O or framework libraries (FastF1, DuckDB, PyMC, NumPyro, FastAPI, HTTP clients) or `f1sim.adapters`. `uv run lint-imports` enforces this.
- New behaviour arrives as a new adapter, `LapStep`, `Predictor` or config file, not as edits to the engine.
- Every implementation of a port must pass that port's contract tests in `tests/contract/`.
- Decisions that are hard to reverse get an ADR in `docs/adr/`.

## Local checks

```bash
uv run pre-commit install    # once
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run lint-imports
uv run pytest
cd frontend && pnpm lint && pnpm typecheck && pnpm test
```

## Model changes

A PR that changes a model or the engine must include its backtest score. CI fails it if ranked probability score is worse than the best recorded in `benchmarks/`.
