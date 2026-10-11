"""Property tests: any simulated result yields a valid PositionMatrix."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from f1sim.core.domain import PositionMatrix


@st.composite
def finishing_orders(draw: st.DrawFn) -> np.ndarray:
    """Random finishing orders for 2-22 drivers over 1-50 simulations, about 10% DNFs."""
    n = draw(st.integers(min_value=2, max_value=22))
    sims = draw(st.integers(min_value=1, max_value=50))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    orders = np.stack([rng.permutation(n) + 1 for _ in range(sims)])
    dnf = rng.random(orders.shape) < 0.1
    orders[dnf] = 0
    return orders


@settings(max_examples=60)
@given(finishing_orders())
def test_matrix_from_any_simulation_satisfies_invariants(orders: np.ndarray) -> None:
    drivers = tuple(f"D{i}" for i in range(orders.shape[1]))
    m = PositionMatrix.from_finishing_orders(drivers, orders)
    assert np.allclose(m.p_position.sum(axis=1) + m.p_dnf, 1.0)
    assert (m.p_position.sum(axis=0) <= 1.0 + 1e-9).all()
