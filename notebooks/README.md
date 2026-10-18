# Notebooks

Research write-ups, numbered in reading order. The regulation-reset study lives here:

- `01-regulation-reset.ipynb` — recovery curves after a regulation reset (Phase 7, P7-04). It
  executes end to end in CI from a clean environment against a pinned fixture: CI has no raw
  data and no network, so it reads the synthetic `tests/pace_world` world instead of the data
  lake, and runs the same `run_recovery` as `f1sim report recovery`.

Notebooks import from `f1sim`; logic that matters moves into the package with tests. The
findings are written up in `docs/research/regulation-reset-writeup.md`, and the numbers the CLI
produces on the real snapshot are in `docs/research/regulation-reset.md`.
