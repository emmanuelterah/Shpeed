"""The Monte Carlo simulator end to end: reproducibility (P5-09) and its use of the grid (P5-08)."""

import numpy as np
import pytest

from f1sim.adapters.models.registry import ModelOptions, get_model
from f1sim.adapters.models.simulator import SimulatorPredictor
from f1sim.adapters.models.tyre_priors import default_tyre_model
from f1sim.adapters.store.memory_store import InMemoryStore
from f1sim.core.domain.entities import EventKey
from f1sim.core.domain.race import RaceEntry
from f1sim.core.errors import InvariantError
from f1sim.core.usecases.readers import FAR_FUTURE
from f1sim.core.usecases.simulate import simulate_race
from tests.pace_world import quali_entries, world_store
from tests.race_sim import build_race
from tests.unit.test_pace_model import FITTED

EVENT = EventKey(2024, 4)


@pytest.fixture(scope="module")
def store() -> InMemoryStore:
    """One synthetic season committed as a single snapshot."""
    return world_store()


def _entries(store: InMemoryStore) -> list[RaceEntry]:
    return quali_entries(store, EVENT)


def _fitted(store: InMemoryStore, seed: int) -> SimulatorPredictor:
    model = get_model("simulator", ModelOptions(pace_hyper=FITTED)).factory(seed)
    model.fit(store, snapshot=store.latest_snapshot(), cutoff=FAR_FUTURE)
    assert isinstance(model, SimulatorPredictor)
    return model


def test_same_seed_and_snapshot_give_a_bit_identical_matrix(store: InMemoryStore) -> None:
    entries = _entries(store)
    first = _fitted(store, 11).predict(EVENT, entries)
    second = _fitted(store, 11).predict(EVENT, entries)
    assert np.array_equal(first.p_position, second.p_position)
    assert np.array_equal(first.p_dnf, second.p_dnf)


def test_a_different_seed_gives_a_different_matrix(store: InMemoryStore) -> None:
    entries = _entries(store)
    first = _fitted(store, 11).predict(EVENT, entries)
    other = _fitted(store, 12).predict(EVENT, entries)
    assert not np.array_equal(first.p_position, other.p_position)


def test_engine_runs_are_bit_identical_for_a_seed() -> None:
    matrices = []
    for _ in range(2):
        race = build_race(n_sims=200, n_cars=10, laps=20, seed=3)
        matrices.append(
            simulate_race(race.drivers, race.state, race.steps, race.ctx, np.random.default_rng(3))
        )
    assert np.array_equal(matrices[0].p_position, matrices[1].p_position)


def test_track_position_matters_for_equal_cars() -> None:
    race = build_race(n_sims=2000, n_cars=6, laps=10, seed=5, p_dnf=0.0, pace_spread_s=0.0)
    matrix = simulate_race(race.drivers, race.state, race.steps, race.ctx, np.random.default_rng(5))
    assert matrix.p_win()[0] > matrix.p_win()[-1]


def test_predict_before_fit_is_an_error() -> None:
    model = get_model("simulator", ModelOptions(pace_hyper=FITTED)).factory(0)
    with pytest.raises(InvariantError):
        model.predict(EVENT, [RaceEntry("a", "t", 1), RaceEntry("b", "t", 2)])


def test_default_tyre_model_samples_every_prior_compound() -> None:
    model = default_tyre_model()
    loss = model.sample_loss(np.random.default_rng(0), 5)
    assert loss.compounds == ("SOFT", "MEDIUM", "HARD")
    assert (loss.values >= 0.0).all()
    assert model.compound_offset("SOFT") < model.compound_offset("HARD")
