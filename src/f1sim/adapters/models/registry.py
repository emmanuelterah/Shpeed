"""Registered models, by the name the CLI and benchmark files use (P2-07).

Adding a model means adding one entry here; the backtester, report and benchmark
registry pick it up by name.
"""

from collections.abc import Callable

from f1sim.adapters.config_files import load_team_lineage
from f1sim.adapters.models.baselines import Elo, GridNoise, PlackettLuce
from f1sim.core.errors import DataError
from f1sim.core.ports.predictor import ModelSpec, Predictor


def _grid_noise() -> ModelSpec:
    return ModelSpec(GridNoise.name, GridNoise)


def _elo() -> ModelSpec:
    lineage = load_team_lineage()

    def _build(seed: int) -> Predictor:
        return Elo(seed, lineage=lineage)

    return ModelSpec(Elo.name, _build)


def _plackett_luce() -> ModelSpec:
    return ModelSpec(PlackettLuce.name, PlackettLuce)


_MODELS: dict[str, Callable[[], ModelSpec]] = {
    GridNoise.name: _grid_noise,
    Elo.name: _elo,
    PlackettLuce.name: _plackett_luce,
}


def model_names() -> list[str]:
    """List every registered model.

    Returns:
        Model names in sorted order.
    """
    return sorted(_MODELS)


def get_model(name: str) -> ModelSpec:
    """Look up a model by name.

    Args:
        name: A registered name, such as ``"elo"``.

    Returns:
        The model's spec, ready for ``walk_forward``.

    Raises:
        DataError: If no model has that name.
    """
    if name not in _MODELS:
        raise DataError(f"Unknown model {name!r}; choose from {', '.join(model_names())}")
    return _MODELS[name]()
