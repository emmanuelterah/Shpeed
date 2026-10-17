"""Vectorised race state for the Monte Carlo engine (P5-01).

Every array has shape ``(n_sims, n_cars)``, so one state object holds many parallel races.
A car's place on the road is its cumulative race time (``race_time_s``); the running order and
the gaps follow from it, so any step that changes a time only has to call :func:`update_order`.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class RaceState:
    """Mutable per-lap state for ``n_sims`` parallel races; LapSteps update it in place.

    Attributes:
        lap: Laps completed so far.
        position: Running order, from 1; retired cars take the places behind the runners.
        gap_to_leader_s: Gap to the leader in seconds; infinite for a retired car.
        compound: Encoded tyre compound currently fitted.
        tyre_age: Laps completed on the current tyres.
        fuel_kg: Fuel on board.
        retired: Whether the car is out of the race.
        race_time_s: Cumulative race time in seconds; infinite once retired.
        lap_time_s: Time of the lap just completed in seconds.
        pit_this_lap: Whether the car stopped on the lap just completed.
    """

    lap: int
    position: NDArray[np.int_]
    gap_to_leader_s: NDArray[np.float64]
    compound: NDArray[np.int_]
    tyre_age: NDArray[np.int_]
    fuel_kg: NDArray[np.float64]
    retired: NDArray[np.bool_]
    race_time_s: NDArray[np.float64]
    lap_time_s: NDArray[np.float64]
    pit_this_lap: NDArray[np.bool_]

    @property
    def n_sims(self) -> int:
        """Number of parallel simulations."""
        return int(self.position.shape[0])

    @property
    def n_cars(self) -> int:
        """Number of cars in each simulation."""
        return int(self.position.shape[1])

    @property
    def running(self) -> NDArray[np.bool_]:
        """Which cars are still running, of shape ``(n_sims, n_cars)``."""
        return ~self.retired


def update_order(state: RaceState) -> None:
    """Recompute the running order and gaps from the cars' cumulative times, in place.

    A retired car is pushed behind every runner, so it never blocks a pass or the leader's gap.

    Args:
        state: The state to update.
    """
    times = np.where(state.retired, np.inf, state.race_time_s)
    order = np.argsort(times, axis=1, kind="stable")
    ranks = np.empty_like(order)
    sims = np.arange(state.n_sims, dtype=np.int64).reshape(-1, 1)
    ranks[sims, order] = np.arange(1, state.n_cars + 1)
    state.position = ranks.astype(np.int64)
    leader = np.min(times, axis=1, keepdims=True)
    finite_leader = np.where(np.isfinite(leader), leader, 0.0)
    known = np.where(state.retired, 0.0, state.race_time_s)
    state.gap_to_leader_s = np.where(state.retired, np.inf, known - finite_leader)


def finishing_orders(state: RaceState) -> NDArray[np.int_]:
    """Turn a finished state into per-car finishing positions, ``0`` for a retirement.

    Args:
        state: A state at the chequered flag.

    Returns:
        Array of shape ``(n_sims, n_cars)``: the position among finishers, or 0 for a DNF.
    """
    return np.where(state.retired, 0, state.position).astype(np.int64)


def lap_start_times(state: RaceState) -> NDArray[np.float64]:
    """Each car's cumulative time when it began the current lap.

    Every step that adds time adds it to both ``race_time_s`` and ``lap_time_s``, so the
    difference is where the car was before this lap's steps ran; ordering by it gives the
    running order at the start of the lap.

    Args:
        state: The state part-way through a lap.

    Returns:
        Array of shape ``(n_sims, n_cars)``; infinite for a retired car.
    """
    start = np.where(state.retired, np.inf, state.race_time_s - state.lap_time_s)
    return start.astype(np.float64)


def ahead_in_order(times: NDArray[np.float64]) -> NDArray[np.int_]:
    """The car directly ahead of each car when ordered by ``times``.

    Args:
        times: Array of shape ``(n_sims, n_cars)``; lower is further ahead.

    Returns:
        The column index of the car ahead, or ``-1`` for the leader.
    """
    order = np.argsort(times, axis=1, kind="stable")
    ahead = np.full(order.shape, -1, dtype=np.int64)
    sims = np.arange(times.shape[0], dtype=np.int64).reshape(-1, 1)
    ahead[sims, order[:, 1:]] = order[:, :-1]
    return ahead
