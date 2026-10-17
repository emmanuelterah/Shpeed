"""Contract suite every ``Predictor`` must pass (Liskov substitution, DIP)."""

import numpy as np
import pytest

from f1sim.adapters.models.registry import ModelOptions, get_model
from f1sim.adapters.store.memory_store import InMemoryStore
from f1sim.core.domain.entities import EventKey
from f1sim.core.domain.prediction import PositionMatrix
from f1sim.core.domain.race import RaceEntry
from f1sim.core.ports.predictor import PredictorFactory
from f1sim.core.usecases.readers import FAR_FUTURE
from tests.pace_world import quali_entries, world_store
from tests.unit.test_pace_model import FITTED

pytestmark = pytest.mark.contract

MODELS = ("grid_noise", "elo", "plackett_luce", "pace", "simulator")
EVENT = EventKey(2024, 4)


@pytest.fixture(scope="module")
def store() -> InMemoryStore:
    """One synthetic season committed as a single snapshot."""
    return world_store()


@pytest.fixture(params=MODELS)
def factory(request: pytest.FixtureRequest) -> PredictorFactory:
    """Each registered model in turn; every test below runs against all of them."""
    if request.param in {"pace", "simulator"}:
        return get_model(request.param, ModelOptions(pace_hyper=FITTED)).factory
    return get_model(request.param).factory


def _entries(store: InMemoryStore, round_: int) -> list[RaceEntry]:
    return quali_entries(store, EventKey(EVENT.season, round_))


def test_name_is_a_non_empty_string(factory: PredictorFactory) -> None:
    assert factory(0).name


def test_fit_then_predict_returns_a_valid_position_matrix(
    factory: PredictorFactory, store: InMemoryStore
) -> None:
    model = factory(0)
    model.fit(store, snapshot=store.latest_snapshot(), cutoff=FAR_FUTURE)
    entries = _entries(store, EVENT.round)
    matrix = model.predict(EVENT, entries)
    assert isinstance(matrix, PositionMatrix)
    assert matrix.drivers == tuple(entry.driver_id for entry in entries)
    assert matrix.p_position.shape == (len(entries), len(entries))
    assert matrix.p_dnf.shape == (len(entries),)
    assert np.allclose(matrix.p_position.sum(axis=1) + matrix.p_dnf, 1.0)


def test_prediction_is_deterministic_for_a_seed(
    factory: PredictorFactory, store: InMemoryStore
) -> None:
    entries = _entries(store, EVENT.round)
    first, second = factory(7), factory(7)
    for model in (first, second):
        model.fit(store, snapshot=store.latest_snapshot(), cutoff=FAR_FUTURE)
    assert np.allclose(
        first.predict(EVENT, entries).p_position,
        second.predict(EVENT, entries).p_position,
    )


def test_every_entry_has_a_non_zero_chance_of_winning(
    factory: PredictorFactory, store: InMemoryStore
) -> None:
    model = factory(0)
    model.fit(store, snapshot=store.latest_snapshot(), cutoff=FAR_FUTURE)
    matrix = model.predict(EVENT, _entries(store, EVENT.round))
    assert (matrix.p_win() > 0).all()
