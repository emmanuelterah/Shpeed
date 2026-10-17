"""Property tests on the race engine (P5-10): every lap, every simulation, stays a valid race."""

from typing import Any

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from f1sim.core.domain.race_state import RaceState, finishing_orders
from f1sim.core.engine.engine import run_race
from f1sim.core.ports.lap_step import LapContext
from tests.race_sim import build_race


class InvariantCheck:
    """A step appended after the real ones that checks the state at the end of every lap."""

    name = "invariant_check"

    def __init__(self) -> None:
        """Start with no lap seen."""
        self.previous: np.ndarray | None = None
        self.laps_checked = 0

    def apply(self, state: RaceState, ctx: LapContext, rng: np.random.Generator) -> None:
        """Assert the ordering, gap and time invariants."""
        del ctx, rng
        n = state.n_cars
        expected = np.arange(1, n + 1)
        assert (np.sort(state.position, axis=1) == expected).all(), "order is a permutation"
        running = state.running
        assert (state.gap_to_leader_s[running] >= 0.0).all(), "gaps are non-negative"
        assert np.isfinite(state.race_time_s[running]).all(), "running cars have a time"
        assert np.isinf(state.race_time_s[state.retired]).all(), "retired cars have none"
        n_running = running.sum(axis=1, keepdims=True)
        assert ((state.position <= n_running) == running).all(), "retired cars are at the back"
        assert (state.lap_time_s[running] > 0.0).all(), "every lap takes time"
        if self.previous is not None:
            assert (state.race_time_s[running] > self.previous[running]).all(), "time moves on"
        self.previous = state.race_time_s.copy()
        self.laps_checked += 1


@st.composite
def races(draw: st.DrawFn) -> dict[str, Any]:
    """Random small races: field size, length, run count, seed, DNF rate and overtake aid."""
    return {
        "n_cars": draw(st.integers(min_value=2, max_value=12)),
        "laps": draw(st.integers(min_value=3, max_value=25)),
        "n_sims": draw(st.integers(min_value=1, max_value=30)),
        "seed": draw(st.integers(min_value=0, max_value=2**32 - 1)),
        "p_dnf": draw(st.sampled_from([0.0, 0.2, 0.6])),
        "aid": draw(st.sampled_from([None, "drs", "overtake_mode"])),
    }


@settings(max_examples=40, deadline=None)
@given(params=races())
def test_every_lap_is_a_valid_race(params: dict[str, Any]) -> None:
    race = build_race(**params)
    check = InvariantCheck()
    final = run_race(
        race.state, [*race.steps, check], race.ctx, np.random.default_rng(params["seed"])
    )
    assert check.laps_checked == params["laps"]
    orders = finishing_orders(final)
    for row, retired in zip(orders, final.retired, strict=True):
        finishers = int((~retired).sum())
        assert sorted(row[~retired].tolist()) == list(range(1, finishers + 1))
        assert (row[retired] == 0).all(), "every car finishes or is a DNF"
