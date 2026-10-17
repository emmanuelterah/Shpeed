"""The Monte Carlo race engine: state ordering, the starting grid, the steps and aggregation."""

import numpy as np
import pytest

from f1sim.adapters.models.overtaking import OvertakingModel
from f1sim.adapters.models.tyres import AGE_GRID, MAX_TYRE_AGE, SampledLoss
from f1sim.adapters.steps.dirty_air import DirtyAirStep
from f1sim.adapters.steps.lap_time import LapTimeStep
from f1sim.adapters.steps.overtake import OvertakeStep
from f1sim.adapters.steps.pit_stop import PitStopStep
from f1sim.core.domain.race import RaceEntry
from f1sim.core.domain.race_state import RaceState, finishing_orders, update_order
from f1sim.core.engine.grid import (
    FULL_FUEL_KG,
    GRID_SLOT_S,
    PIT_LANE_START_S,
    apply_grid_penalties,
    starting_state,
)
from f1sim.core.errors import InvariantError
from f1sim.core.ports.lap_step import LapContext
from f1sim.core.usecases.simulate import simulate_race

NO_AID = LapContext(total_laps=5, pit_loss_s=20.0, overtake_aid=None)


def make_entries(grids: tuple[int, ...]) -> list[RaceEntry]:
    """Race entries with the given grid slots."""
    return [RaceEntry(f"d{i}", f"t{i}", grid) for i, grid in enumerate(grids)]


def make_state(
    n_sims: int, compounds: tuple[int, ...], grids: tuple[int, ...] | None = None
) -> RaceState:
    """A starting state with the given starting compounds, one column per compound code."""
    entries = make_entries(grids if grids is not None else tuple(range(1, len(compounds) + 1)))
    start = np.tile(np.asarray(compounds, dtype=np.int64), (n_sims, 1))
    return starting_state(entries, n_sims=n_sims, start_compounds=start)


def loss_curves(
    compounds: tuple[str, ...], rates: tuple[float, ...], n_sims: int, warm: float = 0.0
) -> SampledLoss:
    """Loss curves that grow ``rate`` seconds per lap, with an optional fixed warm-up."""
    values = np.zeros((len(compounds), n_sims, AGE_GRID.shape[0]), dtype=np.float64)
    for k, rate in enumerate(rates):
        for age in range(AGE_GRID.shape[0]):
            values[k, :, age] = (warm if age in (1, 2) else 0.0) + rate * age
    return SampledLoss(compounds, values)


def test_update_order_ranks_by_time_and_sets_gaps() -> None:
    state = make_state(2, (0, 0), grids=(1, 2))
    state.race_time_s = np.array([[10.0, 9.0], [5.0, 6.0]])
    update_order(state)
    assert state.position.tolist() == [[2, 1], [1, 2]]
    assert state.gap_to_leader_s.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_update_order_pushes_retired_cars_to_the_back() -> None:
    state = make_state(1, (0, 0))
    state.race_time_s = np.array([[3.0, 1.0]])
    state.retired = np.array([[True, False]])
    update_order(state)
    assert state.position.tolist() == [[2, 1]]
    assert state.gap_to_leader_s[0, 0] == np.inf


def test_finishing_orders_codes_retirements_as_zero() -> None:
    state = make_state(1, (0, 0))
    state.race_time_s = np.array([[1.0, 2.0]])
    state.retired = np.array([[False, True]])
    update_order(state)
    assert finishing_orders(state).tolist() == [[1, 0]]


def test_starting_state_uses_grid_order_and_full_tanks() -> None:
    entries = make_entries((2, 1))
    start = np.array([[0, 1]])
    state = starting_state(entries, n_sims=1, start_compounds=start)
    assert state.position.tolist() == [[2, 1]]
    assert state.fuel_kg.tolist() == [[FULL_FUEL_KG, FULL_FUEL_KG]]
    assert state.lap == 0
    assert not state.pit_this_lap.any()


def test_starting_state_rejects_mismatched_shapes() -> None:
    entries = make_entries((1, 2))
    with pytest.raises(InvariantError):
        starting_state(entries, n_sims=1, start_compounds=np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(InvariantError):
        starting_state(
            entries,
            n_sims=1,
            start_compounds=np.zeros((1, 2), dtype=np.int64),
            retired=np.zeros((2, 2), dtype=bool),
        )


def test_lap_time_step_adds_pace_loss_and_noise() -> None:
    state = make_state(1, (0, 0))
    pace = np.array([[90.0, 91.0]])
    loss = loss_curves(("MEDIUM",), (0.5,), 1, warm=0.7)
    step = LapTimeStep(
        pace, loss, fuel_s_per_lap=0.0, burn_kg_per_lap=0.0, noise_s=0.0, reference_fuel_kg=0.0
    )
    step.apply(state, NO_AID, np.random.default_rng(1))
    np.testing.assert_allclose(state.lap_time_s, [[91.2, 92.2]])
    assert state.tyre_age.tolist() == [[1, 1]]
    np.testing.assert_allclose(state.race_time_s, [[91.2, 92.2 + GRID_SLOT_S]])


def test_lap_time_step_burns_fuel_and_orders_the_cars() -> None:
    state = make_state(1, (0, 0))
    pace = np.array([[90.0, 91.0]])
    loss = loss_curves(("MEDIUM",), (0.0,), 1)
    step = LapTimeStep(
        pace, loss, fuel_s_per_lap=0.1, burn_kg_per_lap=10.0, noise_s=0.0, reference_fuel_kg=100.0
    )
    step.apply(state, NO_AID, np.random.default_rng(1))
    assert state.position.tolist() == [[1, 2]]
    assert state.race_time_s[0, 0] < state.race_time_s[0, 1]
    assert state.fuel_kg.tolist() == [[90.0, 90.0]]


def test_lap_time_step_rejects_bad_pace() -> None:
    loss = loss_curves(("MEDIUM",), (0.0,), 1)
    with pytest.raises(InvariantError):
        LapTimeStep(
            np.zeros(3), loss, fuel_s_per_lap=0.0, burn_kg_per_lap=0.0, reference_fuel_kg=0.0
        )
    with pytest.raises(InvariantError):
        LapTimeStep(
            np.array([[np.nan]]),
            loss,
            fuel_s_per_lap=0.0,
            burn_kg_per_lap=0.0,
            reference_fuel_kg=0.0,
        )
    with pytest.raises(InvariantError):
        LapTimeStep(
            np.array([[90.0]]),
            loss,
            fuel_s_per_lap=-1.0,
            burn_kg_per_lap=0.0,
            reference_fuel_kg=0.0,
        )


def test_pit_stop_step_applies_the_plan_on_the_stop_lap() -> None:
    state = make_state(1, (0, 0))
    state.lap = 3
    state.race_time_s = np.array([[270.0, 271.0]])
    state.lap_time_s = np.array([[90.0, 91.0]])
    state.tyre_age = np.array([[3, 3]])
    stop_laps = np.array([[[3], [0]]])
    stop_compounds = np.array([[[1], [0]]])
    pit_loss = np.array([[[12.0], [12.0]]])
    PitStopStep(stop_laps, stop_compounds, pit_loss).apply(state, NO_AID, np.random.default_rng(1))
    np.testing.assert_allclose(state.race_time_s, [[282.0, 271.0]])
    assert state.compound.tolist() == [[1, 0]]
    assert state.tyre_age.tolist() == [[0, 3]]
    assert state.pit_this_lap.tolist() == [[True, False]]


def test_pit_stop_step_rejects_bad_shapes() -> None:
    plan = np.zeros((1, 2, 1), dtype=np.int64)
    with pytest.raises(InvariantError):
        PitStopStep(plan, np.zeros((1, 2, 2), dtype=np.int64), np.zeros((1, 2, 1)))
    with pytest.raises(InvariantError):
        PitStopStep(plan, plan, np.zeros((1, 2)))
    flat = np.zeros((1, 2), dtype=np.int64)
    with pytest.raises(InvariantError):
        PitStopStep(flat, flat, np.zeros((1, 2)))


def _overtaking(intercept: float) -> OvertakingModel:
    return OvertakingModel(intercept, 0.0, {}, {})


def _mid_lap(start: list[float], lap: list[float]) -> RaceState:
    """A one-simulation state after the lap-time step: started at ``start``, ran ``lap``."""
    state = make_state(1, tuple(0 for _ in start))
    state.lap_time_s = np.array([lap])
    state.race_time_s = np.array([start]) + state.lap_time_s
    return state


def test_overtake_step_passes_when_the_pass_succeeds() -> None:
    state = _mid_lap([0.0, 0.5], [90.0, 89.0])
    OvertakeStep(_overtaking(20.0)).apply(state, NO_AID, np.random.default_rng(1))
    assert state.position.tolist() == [[2, 1]]


def test_overtake_step_holds_the_follower_when_the_pass_fails() -> None:
    state = _mid_lap([0.0, 0.5], [90.0, 89.0])
    OvertakeStep(_overtaking(-20.0), min_gap_s=0.3).apply(state, NO_AID, np.random.default_rng(1))
    assert state.position.tolist() == [[1, 2]]
    assert state.race_time_s[0, 1] == pytest.approx(90.3)
    assert state.lap_time_s[0, 1] == pytest.approx(89.8)


def test_overtake_step_pass_offset_can_rule_out_a_pass() -> None:
    state = _mid_lap([0.0, 0.5], [90.0, 89.0])
    OvertakeStep(_overtaking(20.0), pass_offset=-40.0).apply(
        state, NO_AID, np.random.default_rng(1)
    )
    assert state.position.tolist() == [[1, 2]]


def test_overtake_step_leaves_a_car_out_of_range_alone() -> None:
    state = _mid_lap([0.0, 0.5], [90.0, 91.0])
    OvertakeStep(_overtaking(20.0)).apply(state, NO_AID, np.random.default_rng(1))
    np.testing.assert_allclose(state.race_time_s, [[90.0, 91.5]])


def test_overtake_step_gains_at_most_one_place_a_lap() -> None:
    state = _mid_lap([0.0, 1.0, 2.0], [90.0, 90.0, 80.0])
    OvertakeStep(_overtaking(20.0)).apply(state, NO_AID, np.random.default_rng(1))
    assert state.position.tolist() == [[1, 3, 2]]


def test_overtake_step_ignores_a_car_that_pitted() -> None:
    state = _mid_lap([0.0, 0.5], [110.0, 90.0])
    state.pit_this_lap = np.array([[True, False]])
    OvertakeStep(_overtaking(-20.0)).apply(state, NO_AID, np.random.default_rng(1))
    np.testing.assert_allclose(state.race_time_s, [[110.0, 90.5]])


def test_overtake_step_rejects_bad_gaps() -> None:
    with pytest.raises(InvariantError):
        OvertakeStep(_overtaking(0.0), min_gap_s=0.0)


def test_dirty_air_step_charges_close_followers_only() -> None:
    state = _mid_lap([0.0, 0.5, 5.0], [90.0, 90.0, 90.0])
    DirtyAirStep(window_s=1.0, penalty_s=0.4).apply(state, NO_AID, np.random.default_rng(1))
    np.testing.assert_allclose(state.lap_time_s, [[90.0, 90.2, 90.0]])
    np.testing.assert_allclose(state.race_time_s, [[90.0, 90.7, 95.0]])


def test_dirty_air_step_rejects_a_bad_window() -> None:
    with pytest.raises(InvariantError):
        DirtyAirStep(window_s=0.0)


def test_grid_slots_start_staggered_and_pit_lane_starts_last() -> None:
    entries = [
        RaceEntry("a", "t", 1),
        RaceEntry("b", "t", 2, pit_lane=True),
        RaceEntry("c", "t", 3),
    ]
    state = starting_state(entries, n_sims=1, start_compounds=np.zeros((1, 3), dtype=np.int64))
    assert state.position.tolist() == [[1, 3, 2]]
    np.testing.assert_allclose(
        state.race_time_s, [[0.0, 2 * GRID_SLOT_S + PIT_LANE_START_S, GRID_SLOT_S]]
    )


def test_cars_out_before_the_start_are_retired_at_lap_zero() -> None:
    entries = make_entries((1, 2))
    retired = np.array([[True, False]])
    state = starting_state(
        entries, n_sims=1, start_compounds=np.zeros((1, 2), dtype=np.int64), retired=retired
    )
    assert state.position.tolist() == [[2, 1]]
    assert state.race_time_s[0, 0] == np.inf


def test_grid_penalties_drop_places_and_others_move_up() -> None:
    entries = make_entries((1, 2, 3, 4))
    grid = apply_grid_penalties(entries, {"d0": 2})
    assert [e.grid for e in grid] == [3, 1, 2, 4]


def test_back_of_grid_penalties_keep_qualifying_order() -> None:
    entries = make_entries((1, 2, 3, 4))
    grid = apply_grid_penalties(entries, {"d0": 99, "d1": 99})
    assert [e.grid for e in grid] == [3, 4, 1, 2]


def test_shared_targets_keep_the_earlier_qualifier_ahead() -> None:
    entries = make_entries((1, 2, 3, 4))
    grid = apply_grid_penalties(entries, {"d0": 2, "d1": 1})
    assert [e.grid for e in grid] == [3, 4, 1, 2]


def test_pit_lane_starters_go_behind_the_grid() -> None:
    entries = make_entries((1, 2, 3))
    grid = apply_grid_penalties(entries, {"d2": 1}, pit_lane={"d0"})
    assert [(e.grid, e.pit_lane) for e in grid] == [(3, True), (1, False), (2, False)]


def test_grid_penalties_reject_unknown_drivers_and_negative_drops() -> None:
    entries = make_entries((1, 2))
    with pytest.raises(InvariantError):
        apply_grid_penalties(entries, {"nobody": 1})
    with pytest.raises(InvariantError):
        apply_grid_penalties(entries, {"d0": -1})


def test_lap_time_step_adds_the_compound_offset() -> None:
    state = make_state(1, (0, 1))
    pace = np.array([[90.0, 90.0]])
    loss = loss_curves(("SOFT", "HARD"), (0.0, 0.0), 1)
    step = LapTimeStep(
        pace,
        loss,
        fuel_s_per_lap=0.0,
        burn_kg_per_lap=0.0,
        noise_s=0.0,
        compound_offset_s=np.array([-0.5, 0.4]),
    )
    step.apply(state, NO_AID, np.random.default_rng(1))
    np.testing.assert_allclose(state.lap_time_s, [[89.5, 90.4]])
    with pytest.raises(InvariantError):
        LapTimeStep(
            pace, loss, fuel_s_per_lap=0.0, burn_kg_per_lap=0.0, compound_offset_s=np.zeros(3)
        )


def test_simulate_race_counts_finishing_positions() -> None:
    drivers = ("d0", "d1")
    state = make_state(400, (0, 0))
    pace = np.array([[90.0, 91.0]])
    loss = loss_curves(("MEDIUM",), (0.0,), 400)
    step = LapTimeStep(
        pace, loss, fuel_s_per_lap=0.0, burn_kg_per_lap=0.0, noise_s=0.0, reference_fuel_kg=0.0
    )
    matrix = simulate_race(drivers, state, [step], NO_AID, np.random.default_rng(1))
    assert matrix.drivers == drivers
    assert matrix.p_win()[0] > 0.99
    assert matrix.p_position.sum(axis=0).max() <= 1.0 + 1e-6


def test_max_tyre_age_is_used_for_old_tyres() -> None:
    assert int(AGE_GRID[-1]) == MAX_TYRE_AGE
