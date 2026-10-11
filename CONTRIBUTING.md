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

## Code style (checked in CI and pre-commit)

| Rule | What it means | Checked by |
|---|---|---|
| PEP 8, with 100-character lines | Every line of code, in every language, is at most 100 characters, including comments, strings and lines with `# type:` or `# noqa` comments | ruff, pylint, `scripts/check_line_length.py` |
| Imports at the top | All imports sit at the top of the file, grouped standard library, third party, then `f1sim`, sorted. No imports inside functions. Absolute imports only | ruff (`I`, `PLC0415`, `TID`), pylint |
| Google-style docstrings | Every public module, class, function and method has one. Functions document `Args`, `Returns`, `Raises` and `Yields`. Private names (`_name`) and test functions are exempt; test helpers and fixtures are not | ruff (`D`, Google convention), pylint `docparams` |
| Pylint 10/10 | `pyproject.toml` configures `src/` and `scripts/`; `tests/pylintrc` adjusts for pytest idioms (fixture parameters, private access) | pylint |
| Strict typing | pyright strict for `src/` and `scripts/`, the same engine as Pylance; stubs for untyped libraries live in `typings/` | pyright |
| No blanket suppressions | No `# type: ignore` or file-wide `pylint: disable`. Fix the cause; if a library is untyped, add a stub to `typings/` | review |
| PyPI-ready packaging | The wheel and sdist build, pass `twine check --strict` and `check-wheel-contents`, and data files ship inside `src/f1sim/resources/` | CI |

Naming follows PEP 8: `snake_case` functions and variables, `CapWords` classes,
`UPPER_CASE` constants, a leading underscore for anything private to a module.

Editors: the repo's `.vscode/` settings use Pylance, ruff on save and a ruler at 100
characters, and `.editorconfig` sets the limit for other editors.

## Local checks

```bash
uv run pre-commit install    # once
uv run ruff check . && uv run ruff format --check .
uv run pylint src scripts && uv run pylint --rcfile=tests/pylintrc tests
uv run python scripts/check_line_length.py
uv run pyright
uv run lint-imports
uv run pytest
uv build && uv run twine check --strict dist/* && uv run check-wheel-contents dist/*.whl
cd frontend && pnpm lint && pnpm typecheck && pnpm test
```

## Model changes

A PR that changes a model or the engine must include its backtest score. CI fails it if ranked probability score is worse than the best recorded in `benchmarks/`.
