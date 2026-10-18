"""Prior-handling strategies for a regulation reset (P7-02)."""

import numpy as np
import pytest

from f1sim.adapters.config_files import load_engine_suppliers, load_team_lineage
from f1sim.adapters.models.pace_filter import PaceFilter, PaceHyper, PaceState
from f1sim.adapters.models.resets import (
    STRATEGIES,
    ExponentialDecay,
    FullHistory,
    ResetConfig,
    ResetPriors,
    build_reset,
    reset_car_effects,
    reset_signature,
)
from f1sim.adapters.models.resets import (
    TestingInformedPriors as InformedPriors,
)
from f1sim.core.errors import DataError


def _state() -> PaceState:
    """A small posterior with two car effects and one driver's correlated pair."""
    labels = ["team:woking", "dr:max_verstappen", "dq:max_verstappen", "sup:honda"]
    mean = np.array([0.5, 0.2, -0.1, 0.3])
    cov = np.eye(4) * 0.04 + 0.01
    return PaceState(labels, mean, cov, last_event=(2021, 20))


def test_full_history_leaves_the_posterior_alone() -> None:
    state = _state()
    before = state.cov.copy()
    FullHistory().at_weekend(state, season=2022, new_season=True)
    assert np.array_equal(state.cov, before)
    assert state.mean[0] == 0.5
    assert FullHistory().noise_scale("FP", 2022) == 1.0


def test_decay_inflates_the_whole_covariance_every_weekend() -> None:
    strategy = ExponentialDecay(half_life=10.0)
    assert strategy.factor == pytest.approx(2.0**0.1)
    state = _state()
    before = state.cov.copy()
    strategy.at_weekend(state, season=2021, new_season=False)
    assert np.allclose(state.cov, before * 2.0**0.1)


def test_decay_needs_a_positive_half_life() -> None:
    with pytest.raises(DataError):
        ExponentialDecay(half_life=0.0)


def test_reset_replaces_only_car_effects() -> None:
    state = _state()
    driver_cov = state.cov[1:3, 1:3].copy()
    reset_car_effects(state, sd=1.5)
    for i, label in enumerate(state.labels):
        if label.startswith(("team:", "sup:", "tq:", "tc:")):
            assert state.mean[i] == 0.0
            assert state.cov[i, i] == pytest.approx(1.5**2)
            assert np.all(state.cov[i, [j for j in range(4) if j != i]] == 0.0)
    assert np.array_equal(state.cov[1:3, 1:3], driver_cov)
    assert state.mean[1] == 0.2


def test_reset_only_fires_at_the_first_weekend_of_the_reset_season() -> None:
    late = _state()
    ResetPriors(2022).at_weekend(late, season=2021, new_season=True)
    assert late.mean[0] == 0.5
    ResetPriors(2022).at_weekend(late, season=2022, new_season=False)
    assert late.mean[0] == 0.5
    ResetPriors(2022).at_weekend(late, season=2022, new_season=True)
    assert late.mean[0] == 0.0


def test_reset_prior_needs_a_positive_spread() -> None:
    with pytest.raises(DataError):
        ResetPriors(2022, sd=0.0)


def test_testing_shrinks_practice_noise_only_in_the_new_era() -> None:
    strategy = InformedPriors(2022, practice_weight=4.0)
    assert strategy.noise_scale("FP", 2022) == pytest.approx(0.5)
    assert strategy.noise_scale("FP", 2021) == 1.0
    assert strategy.noise_scale("R", 2022) == 1.0


def test_testing_needs_a_positive_practice_weight() -> None:
    with pytest.raises(DataError):
        InformedPriors(2022, practice_weight=0.0)


def test_build_reset_reads_each_name_and_its_knobs() -> None:
    assert isinstance(build_reset(None), FullHistory)
    assert isinstance(build_reset(ResetConfig("full")), FullHistory)
    decay = build_reset(ResetConfig("decay", half_life=5.0))
    assert isinstance(decay, ExponentialDecay)
    assert decay.half_life == 5.0
    reset = build_reset(ResetConfig("reset", reset_season=2022, sd=2.0))
    assert isinstance(reset, ResetPriors)
    assert reset.sd == 2.0
    testing = build_reset(ResetConfig("testing", reset_season=2022, practice_weight=9.0))
    assert isinstance(testing, InformedPriors)
    assert testing.practice_weight == 9.0


def test_build_reset_rejects_unknown_names_and_missing_seasons() -> None:
    with pytest.raises(DataError):
        build_reset(ResetConfig("teleport"))
    with pytest.raises(DataError):
        build_reset(ResetConfig("reset"))
    with pytest.raises(DataError):
        build_reset(ResetConfig("testing"))


def test_reset_signature_separates_strategies_by_name_and_knobs() -> None:
    short = reset_signature(ExponentialDecay(half_life=5.0))
    long = reset_signature(ExponentialDecay(half_life=20.0))
    assert short["name"] == "decay"
    assert short != long
    assert set(reset_signature(FullHistory())) == {"name"}
    assert set(STRATEGIES) == {"full", "decay", "reset", "testing"}


def test_the_filter_applies_the_strategy_at_the_season_boundary() -> None:
    state = _state()
    driver_cov = state.cov[1:3, 1:3].copy()
    filt = PaceFilter(
        PaceHyper(),
        load_team_lineage(),
        load_engine_suppliers(),
        state=state,
        reset=ResetPriors(2022, sd=1.5),
    )
    filt.advance(2022, 1)
    index = state.index()
    assert state.mean[index["team:woking"]] == 0.0
    assert state.cov[index["team:woking"], index["team:woking"]] == pytest.approx(1.5**2)
    assert state.cov[index["sup:honda"], index["sup:honda"]] == pytest.approx(1.5**2)
    assert state.mean[index["dr:max_verstappen"]] == 0.2
    assert np.all(state.cov[1:3, 1:3] >= driver_cov)
