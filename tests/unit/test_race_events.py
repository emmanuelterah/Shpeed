"""Phase 6 event steps: the safety car, reliability, lap 1, penalties, rain and traffic."""

import numpy as np
import pytest

from f1sim.adapters.models.hazards import SafetyCarHazard
from f1sim.adapters.steps.lap1 import Lap1Step, launch_fixture
from f1sim.adapters.steps.penalty import PenaltyStep
from f1sim.adapters.steps.rain import RainStep
from f1sim.adapters.steps.reliability import ReliabilityStep
from f1sim.adapters.steps.safety_car import SC_GAP_S, SafetyCarStep, level_of
from f1sim.adapters.steps.traffic import TrafficStep
from f1sim.core.domain.race import RaceEntry
from f1sim.core.domain.race_state import RaceState
from f1sim.core.engine.grid import starting_state
from f1sim.core.errors import InvariantError
from f1sim.core.ports.lap_step import LapContext

NO_AID = LapContext(total_laps=10, pit_loss_s=20.0, overtake_aid=None)


def make_state(n_sims: int, *, n_cars: int = 3, tyre_age: int = 0) -> RaceState:
    """A grid-ready state of ``n_cars`` on a so-far-idle race."""
    entries = [RaceEntry(f"d{i}", f"t{i}", i + 1) for i in range(n_cars)]
    state = starting_state(
        entries, n_sims=n_sims, start_compounds=np.zeros((n_sims, n_cars), dtype=np.int64)
    )
    state.tyre_age = np.full((n_sims, n_cars), tyre_age, dtype=np.int64)
    return state


def mid_lap(race: list[float], lap: list[float]) -> RaceState:
    """A one-simulation state that began the lap at ``race`` and ran ``lap`` seconds."""
    state = make_state(1, n_cars=len(race))
    state.lap_time_s = np.array([lap], dtype=np.float64)
    state.race_time_s = np.array([race], dtype=np.float64) + state.lap_time_s
    return state


def sc_hazard(*, deploy: float, red: float, duration: float = 2.0) -> SafetyCarHazard:
    """A safety-car hazard with no per-circuit overrides."""
    return SafetyCarHazard(deploy, {}, duration, red)


# ---------------------------------------------------------------- safety car


def test_safety_car_bunches_the_field_into_gaps() -> None:
    state = make_state(1, n_cars=4)
    step = SafetyCarStep(sc_hazard(deploy=1.0, red=0.0), total_laps=10, sc_share=1.0)
    step.apply(state, NO_AID, np.random.default_rng(1))
    assert state.safety_car is not None
    assert state.safety_car.tolist() == [2]
    assert state.neutralised.all()
    assert level_of(state).min() == 2
    running = state.gap_to_leader_s[0, ~state.retired[0]]
    assert running.max() == pytest.approx(3 * SC_GAP_S)


def test_virtual_safety_car_does_not_bunch_the_field() -> None:
    state = make_state(1, n_cars=3)
    before = state.race_time_s.copy()
    step = SafetyCarStep(sc_hazard(deploy=1.0, red=0.0), total_laps=10, sc_share=0.0)
    step.apply(state, NO_AID, np.random.default_rng(1))
    assert state.safety_car is not None
    assert state.safety_car.tolist() == [1]
    np.testing.assert_allclose(state.race_time_s, before)


def test_red_flag_resets_tyres_and_runs_a_standing_restart() -> None:
    state = make_state(2, n_cars=3, tyre_age=5)
    launch = launch_fixture(np.array([0.0, -1.0, 1.0], dtype=np.float64), noise_places=0.0)
    step = SafetyCarStep(sc_hazard(deploy=0.0, red=1.0, duration=1.0), total_laps=1, launch=launch)
    step.apply(state, NO_AID, np.random.default_rng(0))
    assert (state.safety_car == 3).all()
    assert (state.tyre_age == 0).all()


def test_safety_car_rejects_a_zero_length_race() -> None:
    with pytest.raises(InvariantError):
        SafetyCarStep(sc_hazard(deploy=0.1, red=0.0), total_laps=0)


# ---------------------------------------------------------------- reliability


def test_reliability_step_retires_a_car_on_its_planned_lap() -> None:
    state = make_state(1)
    state.lap = 1
    step = ReliabilityStep(np.array([[0, 2, 0]], dtype=np.int64))
    step.apply(state, NO_AID, np.random.default_rng(0))
    assert not state.retired.any()
    state.lap = 2
    step.apply(state, NO_AID, np.random.default_rng(0))
    assert state.retired.tolist() == [[False, True, False]]
    assert state.race_time_s[0, 1] == np.inf


def test_reliability_step_rejects_a_bad_plan() -> None:
    with pytest.raises(InvariantError):
        ReliabilityStep(np.zeros(3, dtype=np.int64))
    with pytest.raises(InvariantError):
        ReliabilityStep(np.array([[-1]], dtype=np.int64))


# ---------------------------------------------------------------- lap 1


def test_lap1_launch_reorders_the_field() -> None:
    state = mid_lap([0.0, 0.3, 0.6], [90.0, 90.0, 90.0])
    state.lap = 1
    step = Lap1Step(np.array([0.0, 0.0, 2.0]), incident=0.0, noise_places=0.0)
    step.apply(state, NO_AID, np.random.default_rng(0))
    assert state.position.tolist() == [[2, 3, 1]]


def test_lap1_incident_retires_on_lap_one_only() -> None:
    state = make_state(1, n_cars=2)
    state.lap = 2
    step = Lap1Step(np.zeros(2), incident=1.0)
    step.apply(state, NO_AID, np.random.default_rng(0))
    assert not state.retired.any()
    state.lap = 1
    step.apply(state, NO_AID, np.random.default_rng(0))
    assert state.retired.all()


# ---------------------------------------------------------------- penalties and rain


def test_penalty_step_adds_seconds_on_the_planned_lap() -> None:
    state = mid_lap([90.0, 90.3], [90.0, 90.0])
    state.lap = 2
    PenaltyStep(np.array([[2, 0]]), np.array([[5.0, 0.0]])).apply(
        state, NO_AID, np.random.default_rng(0)
    )
    np.testing.assert_allclose(state.race_time_s, [[185.0, 180.3]])


def test_penalty_step_rejects_mismatched_plans() -> None:
    with pytest.raises(InvariantError):
        PenaltyStep(np.zeros((1, 2), dtype=np.int64), np.zeros((1, 3)))
    with pytest.raises(InvariantError):
        PenaltyStep(np.zeros(2, dtype=np.int64), np.zeros(2))


def test_rain_step_slows_only_the_wet_runs() -> None:
    state = make_state(2, n_cars=2)
    state.lap_time_s = np.full((2, 2), 90.0)
    state.race_time_s = np.full((2, 2), 90.0)
    RainStep(np.array([True, False]), extra_s=12.0, noise_s=0.0).apply(
        state, NO_AID, np.random.default_rng(0)
    )
    assert state.lap_time_s[0, 0] == pytest.approx(102.0)
    assert state.lap_time_s[1, 0] == pytest.approx(90.0)


def test_rain_step_rejects_a_two_dimensional_wet_flag() -> None:
    with pytest.raises(InvariantError):
        RainStep(np.zeros((2, 2), dtype=bool))


# ---------------------------------------------------------------- traffic


def test_traffic_step_charges_the_car_being_followed() -> None:
    state = mid_lap([0.0, 0.3], [90.0, 90.0])
    TrafficStep(window_s=1.5, loss_s=0.3).apply(state, NO_AID, np.random.default_rng(0))
    np.testing.assert_allclose(state.race_time_s, [[90.24, 90.3]])


def test_traffic_step_is_lifted_under_a_neutralisation() -> None:
    state = mid_lap([0.0, 0.3], [90.0, 90.0])
    state.safety_car = np.array([2])
    TrafficStep(window_s=1.5, loss_s=0.3).apply(state, NO_AID, np.random.default_rng(0))
    np.testing.assert_allclose(state.race_time_s, [[90.0, 90.3]])
