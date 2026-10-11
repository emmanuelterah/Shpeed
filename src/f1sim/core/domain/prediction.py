"""The prediction target (P2-01): a driver-by-position probability matrix plus P(DNF)."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from f1sim.core.errors import InvariantError

_TOL = 1e-6


@dataclass(frozen=True)
class PositionMatrix:
    """Finishing-position probabilities for every driver in one race.

    Row ``i`` is driver ``i``; column ``k`` is the probability of finishing in position
    ``k + 1``. Each row plus that driver's P(DNF) sums to 1. Each position column sums to
    at most 1, because a position can go unfilled when cars retire.

    Attributes:
        drivers: Driver IDs, one per row, unique.
        p_position: Array of shape ``(n_drivers, n_drivers)``.
        p_dnf: Array of shape ``(n_drivers,)``: the probability of not finishing.
    """

    drivers: tuple[str, ...]
    p_position: NDArray[np.float64]
    p_dnf: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Check the matrix invariants.

        Raises:
            InvariantError: If driver IDs repeat, shapes disagree, a probability is negative,
                a row does not sum to 1, or a position column sums to more than 1.
        """
        n = len(self.drivers)
        if len(set(self.drivers)) != n:
            raise InvariantError("driver IDs must be unique")
        if self.p_position.shape != (n, n) or self.p_dnf.shape != (n,):
            raise InvariantError(f"expected shapes ({n}, {n}) and ({n},)")
        if (self.p_position < -_TOL).any() or (self.p_dnf < -_TOL).any():
            raise InvariantError("probabilities must be non-negative")
        rows = self.p_position.sum(axis=1) + self.p_dnf
        if not np.allclose(rows, 1.0, atol=_TOL):
            raise InvariantError("each driver's positions plus P(DNF) must sum to 1")
        if (self.p_position.sum(axis=0) > 1.0 + _TOL).any():
            raise InvariantError("each position can be filled at most once")

    @classmethod
    def from_finishing_orders(
        cls, drivers: tuple[str, ...], orders: NDArray[np.int_]
    ) -> "PositionMatrix":
        """Build a matrix by counting simulated finishing positions.

        Args:
            drivers: Driver IDs, one per column of ``orders``.
            orders: Array of shape ``(n_sims, n_drivers)``. ``orders[s, i]`` is driver
                ``i``'s finishing position (from 1) in simulation ``s``, or 0 for a DNF.

        Returns:
            The empirical position probabilities across all simulations.

        Raises:
            InvariantError: If ``orders`` does not have one column per driver.
        """
        n_sims, n = orders.shape
        if n != len(drivers):
            raise InvariantError("orders must have one column per driver")
        counts = np.zeros((n, n), dtype=np.float64)
        for k in range(1, n + 1):
            counts[:, k - 1] = (orders == k).sum(axis=0)
        p_dnf = (orders == 0).sum(axis=0).astype(np.float64) / n_sims
        return cls(drivers=drivers, p_position=counts / n_sims, p_dnf=p_dnf)

    def p_win(self) -> NDArray[np.float64]:
        """Probability that each driver wins.

        Returns:
            One probability per driver, in ``drivers`` order.
        """
        return self.p_position[:, 0]

    def p_top(self, k: int) -> NDArray[np.float64]:
        """Probability that each driver finishes in the top ``k``.

        Args:
            k: Number of leading positions, for example 3 for a podium or 10 for points.

        Returns:
            One probability per driver, in ``drivers`` order.
        """
        return self.p_position[:, :k].sum(axis=1)
