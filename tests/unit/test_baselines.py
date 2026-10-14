"""The three baselines learn the synthetic strength order and beat a uniform forecast."""

from datetime import datetime

import numpy as np
import pytest

from f1sim.adapters.models.baselines import Elo, GridNoise, PlackettLuce, elo_surprise
from f1sim.adapters.models.common import dnf_rate, finishing_orders, rankings
from f1sim.adapters.models.ranking_fit import fit_plackett_luce
from f1sim.adapters.models.registry import get_model, model_names
from f1sim.adapters.store.memory_store import InMemoryStore
from f1sim.core.domain.entities import EventKey
from f1sim.core.domain.lineage import TeamLineage
from f1sim.core.domain.race import RaceEntry
from f1sim.core.errors import DataError, InvariantError
from f1sim.core.ports.predictor import ModelSpec
from f1sim.core.usecases.backtest import walk_forward
from f1sim.core.usecases.readers import FAR_FUTURE
from tests.support import SYNTH_DRIVERS, UniformModel, synthetic_results

ENTRIES = tuple(RaceEntry(d, f"team_{i // 2}", i + 1) for i, d in enumerate(SYNTH_DRIVERS))
EVENT = EventKey(2025, 1)
BASELINES = ("elo", "grid_noise", "plackett_luce")
"""Models that learn from results alone; the pace model needs laps (test_pace_model)."""


@pytest.fixture(scope="module")
def store() -> InMemoryStore:
    """Two synthetic seasons of ten rounds."""
    s = InMemoryStore()
    s.commit({"results": synthetic_results(seasons=(2023, 2024), rounds=10)}, run_id="t")
    return s


END_OF_2024 = datetime(2024, 12, 31)


def _fit(model: GridNoise | Elo | PlackettLuce, store: InMemoryStore) -> None:
    model.fit(store, snapshot=store.latest_snapshot(), cutoff=END_OF_2024)


def test_finishing_orders_ranks_finishers_and_zeroes_dnfs() -> None:
    scores = np.array([[3.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    retired = np.array([[False, False, True], [False, False, False]])
    assert finishing_orders(scores, retired).tolist() == [[1, 2, 0], [3, 2, 1]]


def test_dnf_rate_is_shrunk_toward_the_prior(store: InMemoryStore) -> None:
    history = store.read("results", snapshot=store.latest_snapshot(), as_of=FAR_FUTURE)
    rate = dnf_rate(history)
    assert 0.02 < rate < 0.2
    assert len(rankings(history, "R")) == 20


def test_plackett_luce_fit_recovers_known_strengths() -> None:
    rng = np.random.default_rng(1)
    true_log_w = np.array([1.5, 1.0, 0.5, 0.0])
    orders = [np.argsort(-(true_log_w + rng.gumbel(size=4))).tolist() for _ in range(3000)]
    w = fit_plackett_luce(orders, [1.0] * len(orders), n_items=4, prior_strength=0.1)
    fitted = np.log(w / w[-1])
    assert np.allclose(fitted, true_log_w - true_log_w[-1], atol=0.1)


def test_plackett_luce_fit_validates_inputs() -> None:
    assert fit_plackett_luce([], [], n_items=2).tolist() == [1.0, 1.0]
    with pytest.raises(InvariantError):
        fit_plackett_luce([[0, 1]], [], n_items=2)
    with pytest.raises(InvariantError):
        fit_plackett_luce([[0, 1]], [-1.0], n_items=2)


def test_elo_surprise_is_zero_sum_and_rewards_the_winner() -> None:
    s = elo_surprise(np.array([1500.0, 1500.0, 1500.0]))
    assert s.sum() == pytest.approx(0.0)
    assert s[0] > 0 > s[-1]


@pytest.mark.parametrize("name", ["elo", "plackett_luce"])
def test_rating_models_rank_the_strong_driver_first(name: str, store: InMemoryStore) -> None:
    model = get_model(name).factory(3)
    assert isinstance(model, (Elo, PlackettLuce))
    _fit(model, store)
    matrix = model.predict(EVENT, ENTRIES)
    assert matrix.drivers == SYNTH_DRIVERS
    assert matrix.p_win()[0] > 2 * matrix.p_win()[-1]


def test_elo_carries_team_rating_through_lineage(store: InMemoryStore) -> None:
    lineage = TeamLineage({"team_0_line": ["team_0", "team_0_new"]})
    elo = Elo(0, lineage=lineage)
    _fit(elo, store)
    rookie = RaceEntry("new_driver", "team_0_new", 1)
    assert elo.strength(rookie) > 0
    assert elo.strength(RaceEntry("new_driver", "unknown_team", 1)) == 0


def test_grid_noise_fits_its_spread_and_favours_pole(store: InMemoryStore) -> None:
    model = GridNoise(0)
    _fit(model, store)
    assert model.sigma >= 1.0
    m = model.predict(EVENT, ENTRIES)
    assert m.p_win()[0] > m.p_win()[-1]


def test_invalid_hyperparameters_are_rejected() -> None:
    with pytest.raises(InvariantError):
        Elo(0, lineage=TeamLineage({}), season_regression=2.0)
    with pytest.raises(InvariantError):
        PlackettLuce(0, half_life_days=0)


def test_registry_lists_and_rejects_names() -> None:
    assert model_names() == ["elo", "grid_noise", "pace", "plackett_luce"]
    with pytest.raises(DataError):
        get_model("nope")


def test_every_baseline_beats_uniform_and_is_deterministic(store: InMemoryStore) -> None:
    snap = store.latest_snapshot()
    uniform = ModelSpec("uniform", UniformModel)
    baseline = walk_forward(store, snapshot=snap, seasons=(2024,), model=uniform).summary()
    for name in BASELINES:
        spec = get_model(name)
        first = walk_forward(store, snapshot=snap, seasons=(2024,), model=spec, workers=2)
        again = walk_forward(store, snapshot=snap, seasons=(2024,), model=spec, workers=1)
        assert first.summary() == again.summary(), name
        assert first.summary()["rps"] < baseline["rps"], name
        assert first.summary()["log_loss"] < baseline["log_loss"], name
        assert first.summary()["spearman"] > 0.2, name


def test_cutoff_is_respected_by_fitting(store: InMemoryStore) -> None:
    pl = PlackettLuce(0)
    pl.fit(store, snapshot=store.latest_snapshot(), cutoff=datetime(2000, 1, 1))
    assert not pl.strengths
    assert pl.predict(EVENT, ENTRIES).p_win()[0] == pytest.approx(0.1 * (1 - pl.p_dnf), abs=0.03)
