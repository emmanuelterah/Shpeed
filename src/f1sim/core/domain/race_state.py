"""Vectorised race state for the Monte Carlo engine (P5-01). Arrays have shape (n_sims, n_cars)."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class RaceState:
    """Mutable per-lap state for ``n_sims`` parallel races. LapSteps update it in place."""

    lap: int
    position: NDArray[np.int_]
    gap_to_leader_s: NDArray[np.float64]
    compound: NDArray[np.int_]
    tyre_age: NDArray[np.int_]
    fuel_kg: NDArray[np.float64]
    retired: NDArray[np.bool_]

    @property
    def n_sims(self) -> int:
        return int(self.position.shape[0])

    @property
    def n_cars(self) -> int:
        return int(self.position.shape[1])
