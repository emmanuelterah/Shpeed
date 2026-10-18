"""Registered models, by the name the CLI and benchmark files use (P2-03).

Adding a model means adding one entry here; the backtester, report and benchmark
registry pick it up by name.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from f1sim.adapters.config_files import load_circuits, load_engine_suppliers, load_team_lineage
from f1sim.adapters.models.baselines import Elo, GridNoise, PlackettLuce
from f1sim.adapters.models.common import N_SIMS
from f1sim.adapters.models.pace import TABLES as PACE_TABLES
from f1sim.adapters.models.pace import PaceContext, PacePredictor
from f1sim.adapters.models.pace_filter import PaceHyper
from f1sim.adapters.models.simulator import TABLES as SIMULATOR_TABLES
from f1sim.adapters.models.simulator import SimulatorContext, SimulatorPredictor
from f1sim.core.errors import DataError
from f1sim.core.ports.predictor import ModelSpec, Predictor
from f1sim.core.ports.state import FittedStateStore


@dataclass(frozen=True)
class ModelOptions:
    """Settings some models take; models ignore the ones they do not use.

    Attributes:
        pace_hyper: Pace-model hyperparameters; the built-in defaults when None.
        state_store: Where the pace model saves and resumes fitted states.
        rules_dir: Folder holding ``<season>.yaml`` season-rules files, for the simulator.
        n_sims: Simulated races per forecast for the simulator; its default when None.
        disabled: Simulator event components to switch off, for the ablation runner (P6-09).
        p_rain: Chance the simulator runs a race in the rain (P6-07); zero for dry.
    """

    pace_hyper: PaceHyper | None = None
    state_store: FittedStateStore | None = None
    rules_dir: Path | None = None
    n_sims: int | None = None
    disabled: frozenset[str] = frozenset()
    p_rain: float = 0.0


def pace_context(options: ModelOptions) -> PaceContext:
    """Build the pace model's context from the packaged reference tables.

    Args:
        options: Hyperparameters and state store.

    Returns:
        A context with a fresh observation cache.
    """
    return PaceContext(
        hyper=options.pace_hyper or PaceHyper(),
        lineage=load_team_lineage(),
        suppliers=load_engine_suppliers(),
        circuits=load_circuits(),
        state_store=options.state_store,
    )


def _grid_noise(_: ModelOptions) -> ModelSpec:
    return ModelSpec(GridNoise.name, GridNoise)


def _elo(_: ModelOptions) -> ModelSpec:
    lineage = load_team_lineage()

    def _build(seed: int) -> Predictor:
        return Elo(seed, lineage=lineage)

    return ModelSpec(Elo.name, _build)


def _plackett_luce(_: ModelOptions) -> ModelSpec:
    return ModelSpec(PlackettLuce.name, PlackettLuce)


def _pace(options: ModelOptions) -> ModelSpec:
    context = pace_context(options)

    def _build(seed: int) -> Predictor:
        return PacePredictor(seed, context)

    return ModelSpec(PacePredictor.name, _build, PACE_TABLES)


def simulator_context(options: ModelOptions) -> SimulatorContext:
    """Build the simulator's context from the options, with its calibrated defaults.

    Args:
        options: Pace hyperparameters, rules folder and run count.

    Returns:
        The context every simulator built by the registry shares.
    """
    return SimulatorContext(
        pace=pace_context(options),
        circuits=load_circuits(),
        rules_dir=options.rules_dir,
        n_sims=options.n_sims if options.n_sims is not None else N_SIMS,
        disabled=options.disabled,
        p_rain=options.p_rain,
    )


def _simulator(options: ModelOptions) -> ModelSpec:
    context = simulator_context(options)

    def _build(seed: int) -> Predictor:
        return SimulatorPredictor(seed, context)

    return ModelSpec(SimulatorPredictor.name, _build, SIMULATOR_TABLES)


_MODELS: dict[str, Callable[[ModelOptions], ModelSpec]] = {
    GridNoise.name: _grid_noise,
    Elo.name: _elo,
    PlackettLuce.name: _plackett_luce,
    PacePredictor.name: _pace,
    SimulatorPredictor.name: _simulator,
}


def model_names() -> list[str]:
    """List every registered model.

    Returns:
        Model names in sorted order.
    """
    return sorted(_MODELS)


def get_model(name: str, options: ModelOptions | None = None) -> ModelSpec:
    """Look up a model by name.

    Args:
        name: A registered name, such as ``"elo"``.
        options: Model settings; defaults when None.

    Returns:
        The model's spec, ready for ``walk_forward``.

    Raises:
        DataError: If no model has that name.
    """
    if name not in _MODELS:
        raise DataError(f"Unknown model {name!r}; choose from {', '.join(model_names())}")
    return _MODELS[name](options or ModelOptions())
