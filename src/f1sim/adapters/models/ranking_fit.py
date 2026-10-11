"""Fit Plackett-Luce strengths to weighted rankings (used by P2-10).

A Plackett-Luce ranking is a sequence of choices: the winner is drawn from everyone with
probability proportional to strength, then second place from the rest, and so on. The fit
uses the MM update of Hunter (2004) with a Gamma(1 + c, c) prior on every strength
(Caron and Doucet, 2012), so the prior mode is 1 and items with little data stay near it:

``w_i <- (c + weight of the stages i won) / (c + sum over stages i took part in of
weight / strength left in that stage)``.

All rankings are padded into one matrix, so each iteration is a few array operations.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from f1sim.core.errors import InvariantError


@dataclass(frozen=True)
class _Stages:
    slots: NDArray[np.int_]
    present: NDArray[np.bool_]
    stage: NDArray[np.bool_]
    weights: NDArray[np.float64]


def fit_plackett_luce(
    orders: Sequence[Sequence[int]],
    weights: Sequence[float],
    *,
    n_items: int,
    prior_strength: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-7,
) -> NDArray[np.float64]:
    """Posterior-mode Plackett-Luce strengths for weighted rankings.

    Args:
        orders: Each ranking as item indices, best first.
        weights: One non-negative weight per ranking.
        n_items: Number of items; indices run from 0 to ``n_items - 1``.
        prior_strength: ``c`` in the update; larger pulls strengths harder toward 1.
        max_iter: Most iterations.
        tol: Stop when no strength changes by more than this fraction.

    Returns:
        One strength per item; all ones when there are no rankings.

    Raises:
        InvariantError: If ``orders`` and ``weights`` differ in length, a weight is
            negative or ``prior_strength`` is not positive.
    """
    if len(orders) != len(weights):
        raise InvariantError("need one weight per ranking")
    if prior_strength <= 0 or any(w < 0 for w in weights):
        raise InvariantError("weights must be non-negative and prior_strength positive")
    w = np.ones(n_items, dtype=np.float64)
    if not orders:
        return w
    stages = _stages(orders, weights)
    picked = np.broadcast_to(stages.weights[:, np.newaxis], stages.slots.shape)[stages.stage]
    wins = np.bincount(stages.slots[stages.stage], weights=picked, minlength=n_items)
    for _ in range(max_iter):
        exposure = _exposure(w, stages, n_items)
        updated = (prior_strength + wins) / (prior_strength + exposure)
        change = float(np.max(np.abs(updated - w) / w))
        w = updated
        if change < tol:
            break
    return w


def _stages(orders: Sequence[Sequence[int]], weights: Sequence[float]) -> _Stages:
    width = max(len(o) for o in orders)
    slots = np.full((len(orders), width), -1, dtype=np.int64)
    for row, order in enumerate(orders):
        slots[row, : len(order)] = np.asarray(order, dtype=np.int64)
    present = slots >= 0
    lengths = present.sum(axis=1)
    # A stage is a choice among two or more items; the last place is not a choice.
    stage = present & (np.arange(width)[np.newaxis, :] < (lengths - 1)[:, np.newaxis])
    return _Stages(slots, present, stage, np.asarray(weights, dtype=np.float64))


def _exposure(w: NDArray[np.float64], st: _Stages, n_items: int) -> NDArray[np.float64]:
    """Weighted sum, over every stage each item took part in, of 1 / strength left."""
    held = np.where(st.present, w[np.where(st.present, st.slots, 0)], 0.0)
    remaining = np.cumsum(held[:, ::-1], axis=1)[:, ::-1]
    inverse = np.where(st.stage, 1.0 / np.where(st.stage, remaining, 1.0), 0.0)
    # Item in slot t took part in stages 0..t, so its exposure is the running sum.
    exposure = np.cumsum(inverse, axis=1) * st.weights[:, np.newaxis]
    totals = np.bincount(st.slots[st.present], weights=exposure[st.present], minlength=n_items)
    return totals.astype(np.float64)
