"""Engine performance benchmark (P5-11): 10,000 runs of one race in under a minute."""

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from f1sim.core.domain.prediction import PositionMatrix
from f1sim.core.usecases.simulate import simulate_race
from tests.race_sim import build_race

pytestmark = pytest.mark.perf

N_SIMS = 10_000
BUDGET_S = 60.0


def _run() -> PositionMatrix:
    race = build_race(n_sims=N_SIMS, n_cars=22, laps=57, seed=1)
    return simulate_race(race.drivers, race.state, race.steps, race.ctx, np.random.default_rng(1))


def test_ten_thousand_runs_fit_the_budget(benchmark: BenchmarkFixture) -> None:
    matrix = benchmark.pedantic(_run, rounds=1, iterations=1)
    assert matrix.p_position.shape == (22, 22)
    assert benchmark.stats is not None
    assert float(benchmark.stats["max"]) < BUDGET_S
