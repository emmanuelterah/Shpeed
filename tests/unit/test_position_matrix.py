"""PositionMatrix construction and invariants."""

import numpy as np
import pytest

from f1sim.core.domain import PositionMatrix
from f1sim.core.errors import InvariantError


def test_from_finishing_orders_counts_positions_and_dnfs() -> None:
    orders = np.array([[1, 2, 0], [2, 1, 3], [1, 3, 2], [1, 2, 3]])
    m = PositionMatrix.from_finishing_orders(("A", "B", "C"), orders)
    assert m.p_win().tolist() == [0.75, 0.25, 0.0]
    assert m.p_dnf.tolist() == [0.0, 0.0, 0.25]
    assert np.allclose(m.p_top(2), [1.0, 0.75, 0.25])


def test_rejects_rows_that_do_not_sum_to_one() -> None:
    with pytest.raises(InvariantError):
        PositionMatrix(("A", "B"), np.array([[0.5, 0.2], [0.5, 0.8]]), np.zeros(2))


def test_rejects_duplicate_drivers() -> None:
    with pytest.raises(InvariantError):
        PositionMatrix(("A", "A"), np.eye(2), np.zeros(2))
