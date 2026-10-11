"""Scoring metrics on hand-computed cases."""

import numpy as np

from f1sim.core.scoring.metrics import brier, ranked_probability_score


def test_brier_perfect_and_worst() -> None:
    assert brier(np.array([1.0, 0.0]), np.array([True, False])) == 0.0
    assert brier(np.array([0.0, 1.0]), np.array([True, False])) == 1.0


def test_rps_is_zero_for_a_certain_correct_forecast() -> None:
    p = np.eye(3)
    assert ranked_probability_score(p, np.array([1, 2, 3])) == 0.0


def test_rps_penalises_distance() -> None:
    p = np.eye(3)
    near = ranked_probability_score(p, np.array([2, 1, 3]))
    far = ranked_probability_score(p, np.array([3, 2, 1]))
    assert 0 < near < far
