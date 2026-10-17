"""Build the state every race starts from: the grid, the fuel and the first set of tyres (P5-08).

Grid penalties and pit-lane starts are applied to the entries first (:func:`apply_grid_penalties`),
then :func:`starting_state` turns the grid into a time offset: each slot starts ``GRID_SLOT_S``
behind the one ahead and a pit-lane starter leaves ``PIT_LANE_START_S`` after the last slot. From
there the running order follows from the cumulative race time, so the grid keeps mattering after
lap 1 through track position and the overtaking model. Everything here is a pure function of its
inputs, which is what keeps a seeded race reproducible (P5-09).
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from f1sim.core.domain.race import RaceEntry
from f1sim.core.domain.race_state import RaceState, update_order
from f1sim.core.errors import InvariantError

FULL_FUEL_KG = 100.0
"""Fuel on board at lights out, in kilograms, when no better figure is known."""
GRID_SLOT_S = 0.3
"""Time between consecutive grid slots as the field crosses the start line, seconds."""
PIT_LANE_START_S = 5.0
"""Extra delay of a pit-lane starter behind the last grid slot, seconds."""


def apply_grid_penalties(
    entries: Sequence[RaceEntry],
    drops: Mapping[str, int] | None = None,
    pit_lane: Collection[str] = (),
) -> list[RaceEntry]:
    """Turn a qualifying order into the starting grid (P5-08).

    A driver with a drop of ``d`` places targets slot ``grid + d``, capped at the back of the
    grid; the others keep their qualifying order and the penalised drivers are slotted in at
    their targets, the earlier qualifier ahead when two share a target. Pit-lane starters take
    the slots behind every grid starter, in qualifying order, and are flagged so the engine
    delays them.

    Args:
        entries: Entries numbered by qualifying position.
        drops: Driver ID to grid places dropped; a large number means the back of the grid.
        pit_lane: Driver IDs starting from the pit lane.

    Returns:
        New entries, in the input order, with their starting slot and ``pit_lane`` flag set.

    Raises:
        InvariantError: If a drop is negative or names an unknown driver.
    """
    penalties = dict(drops or {})
    known = {entry.driver_id for entry in entries}
    unknown = (set(penalties) | set(pit_lane)) - known
    if unknown:
        raise InvariantError(f"penalty for unknown drivers: {sorted(unknown)}")
    if any(value < 0 for value in penalties.values()):
        raise InvariantError("grid drops must be non-negative")
    by_quali = sorted(entries, key=lambda entry: entry.grid)
    starters = [entry.driver_id for entry in by_quali if entry.driver_id not in pit_lane]
    lineup = _penalised_lineup(starters, penalties)
    lineup += [entry.driver_id for entry in by_quali if entry.driver_id in pit_lane]
    slots = {driver: slot for slot, driver in enumerate(lineup, start=1)}
    return [
        replace(entry, grid=slots[entry.driver_id], pit_lane=entry.driver_id in pit_lane)
        for entry in entries
    ]


def starting_state(
    entries: Sequence[RaceEntry],
    *,
    n_sims: int,
    start_compounds: NDArray[np.int_],
    fuel_kg: float = FULL_FUEL_KG,
    retired: NDArray[np.bool_] | None = None,
) -> RaceState:
    """Build the starting :class:`RaceState` from the grid.

    Args:
        entries: The starters; column ``i`` of every array is ``entries[i]``.
        n_sims: Number of parallel simulations.
        start_compounds: Array of shape ``(n_sims, n_cars)``: the compound each car starts on.
        fuel_kg: Fuel on board at the start, per car.
        retired: Which cars are already out, of shape ``(n_sims, n_cars)``; none when None.

    Returns:
        A state at lap 0 with full tanks, fresh tyres and the grid's time offsets.

    Raises:
        InvariantError: If the shapes disagree with ``entries`` or ``n_sims``.
    """
    n_cars = len(entries)
    if start_compounds.shape != (n_sims, n_cars):
        raise InvariantError(f"start_compounds must have shape ({n_sims}, {n_cars})")
    if retired is not None and retired.shape != (n_sims, n_cars):
        raise InvariantError(f"retired must have shape ({n_sims}, {n_cars})")
    out = np.zeros((n_sims, n_cars), dtype=bool) if retired is None else retired.copy()
    offsets = np.tile(grid_offsets(entries), (n_sims, 1))
    state = RaceState(
        lap=0,
        position=np.tile(_grid_ranks(entries), (n_sims, 1)).astype(np.int64),
        gap_to_leader_s=offsets.copy(),
        compound=start_compounds,
        tyre_age=np.zeros((n_sims, n_cars), dtype=np.int64),
        fuel_kg=np.full((n_sims, n_cars), fuel_kg, dtype=np.float64),
        retired=out,
        race_time_s=np.where(out, np.inf, offsets),
        lap_time_s=np.zeros((n_sims, n_cars), dtype=np.float64),
        pit_this_lap=np.zeros((n_sims, n_cars), dtype=bool),
    )
    update_order(state)
    return state


def grid_offsets(entries: Sequence[RaceEntry]) -> NDArray[np.float64]:
    """Time each car crosses the start line behind pole, from its grid slot.

    Args:
        entries: The starters.

    Returns:
        One offset per entry in seconds; pit-lane starters sit behind the last slot.
    """
    ranks = _grid_ranks(entries).astype(np.float64)
    offsets = (ranks - 1.0) * GRID_SLOT_S
    pit = np.array([entry.pit_lane for entry in entries], dtype=bool)
    return np.where(pit, offsets + PIT_LANE_START_S, offsets).astype(np.float64)


def _grid_ranks(entries: Sequence[RaceEntry]) -> NDArray[np.int_]:
    keys = np.array([(entry.pit_lane, entry.grid) for entry in entries], dtype=np.int64)
    order = np.lexsort((keys[:, 1], keys[:, 0])) if len(entries) else np.zeros(0, np.int64)
    ranks = np.empty(len(entries), dtype=np.int64)
    ranks[order] = np.arange(1, len(entries) + 1)
    return ranks


def _penalised_lineup(starters: Sequence[str], penalties: Mapping[str, int]) -> list[str]:
    size = len(starters)
    quali_rank = {driver: index for index, driver in enumerate(starters, start=1)}
    targets = {
        driver: min(size, quali_rank[driver] + penalties[driver])
        for driver in starters
        if penalties.get(driver, 0) > 0
    }
    lineup = [driver for driver in starters if driver not in targets]
    placed: dict[int, int] = {}
    # Earlier qualifiers first; a later driver with the same target goes in behind them.
    for driver in sorted(targets, key=lambda d: (targets[d], quali_rank[d])):
        target = targets[driver]
        lineup.insert(min(target - 1 + placed.get(target, 0), len(lineup)), driver)
        placed[target] = placed.get(target, 0) + 1
    return lineup
