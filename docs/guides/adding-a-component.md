# Adding a component

Shpeed is built so new behaviour arrives as a new class, not as edits to the engine (open/closed principle).

## A new race mechanic (LapStep)

1. Create `src/f1sim/adapters/steps/<name>.py` with a class implementing `f1sim.core.ports.LapStep`.
2. Register it in the composition root (`src/f1sim/cli.py`) behind a config flag.
3. Add unit and property tests; the engine's invariants in `tests/property/` must still hold.
4. Run the ablation (`f1sim backtest --ablate <name>`) and attach the table to the PR.

## A new model (Predictor)

1. Implement `f1sim.core.ports.Predictor` under `src/f1sim/adapters/models/`.
2. Add it to the contract suite parameters in `tests/contract/test_predictor_contract.py`.
3. Record its backtest scores in `benchmarks/` (P2-13).

## A new data source (SourceAdapter)

1. Implement `f1sim.core.ports.SourceAdapter` under `src/f1sim/adapters/sources/`.
2. Record fixtures into `tests/fixtures/recorded/<source>/` so CI never touches the network.
