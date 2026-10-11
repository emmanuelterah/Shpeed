"""Outcome encoding, per-race scoring and the metrics behind it, on hand-computed cases."""

import math

import numpy as np
import pytest

from f1sim.core.domain import PositionMatrix
from f1sim.core.domain.race import DNF, RaceEntry, encode_outcome
from f1sim.core.errors import InvariantError
from f1sim.core.scoring.metrics import (
    LOG_LOSS_FLOOR,
    calibration_bins,
    log_loss,
    ranked_probability_score,
    spearman,
)
from f1sim.core.scoring.race_score import event_forecasts, score_race

ENTRIES = (RaceEntry("a", "t1", 1), RaceEntry("b", "t1", 2), RaceEntry("c", "t2", 3))


def test_encode_outcome_ranks_among_entries_and_marks_dnfs() -> None:
    # "x" is not an entry; positions are re-ranked among the entries that finished.
    outcome = encode_outcome(ENTRIES, [("c", 1), ("x", 2), ("a", 4)])
    assert outcome.tolist() == [2, DNF, 1]


def test_categories_and_smoothing_keep_invariants() -> None:
    m = PositionMatrix(("a", "b"), np.eye(2), np.zeros(2))
    assert m.categories().shape == (2, 3)
    s = m.smoothed(0.3)
    assert np.isclose(s.p_position[0, 0], 0.7 + 0.1)
    assert np.isclose(s.p_dnf[0], 0.1)
    with pytest.raises(InvariantError):
        m.smoothed(1.5)


def test_certain_correct_forecast_scores_perfectly() -> None:
    m = PositionMatrix(("a", "b", "c"), np.eye(3), np.zeros(3))
    score = score_race(2024, 1, m, np.array([1, 2, 3]))
    assert score.brier_win == score.brier_podium == score.rps == 0.0
    assert score.log_loss == pytest.approx(0.0)
    assert score.spearman == pytest.approx(1.0)


def test_dnf_is_scored_as_the_last_category() -> None:
    m = PositionMatrix(("a", "b"), np.array([[1.0, 0.0], [0.0, 0.5]]), np.array([0.0, 0.5]))
    score = score_race(2024, 1, m, np.array([1, DNF]))
    # b: categories (P1, P2, DNF) = (0, 0.5, 0.5); actual DNF -> log loss -ln 0.5 / 2 drivers.
    assert score.log_loss == pytest.approx(math.log(2) / 2)
    # b's CDF (0, 0.5) against observed (0, 0): squared error 0.25, over 2 thresholds, /2 rows.
    assert score.rps == pytest.approx(0.25 / 2 / 2)
    events = event_forecasts(m, np.array([1, DNF]))
    assert events["podium"].happened.tolist() == [True, False]


def test_log_loss_floors_impossible_outcomes() -> None:
    p = np.array([[1.0, 0.0]])
    assert log_loss(p, np.array([2])) == pytest.approx(-math.log(LOG_LOSS_FLOOR))


def test_rps_of_uniform_forecast_is_positive() -> None:
    p = np.full((2, 2), 0.5)
    assert ranked_probability_score(p, np.array([1, 2])) == pytest.approx(0.25)


def test_spearman_handles_ties_and_degenerate_input() -> None:
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])) == pytest.approx(-1.0)
    assert spearman(np.array([1.0]), np.array([1.0])) == 0.0
    assert spearman(np.array([1.0, 1.0]), np.array([1.0, 2.0])) == 0.0
    assert spearman(np.array([1.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])) > 0


def test_calibration_bins_count_every_forecast() -> None:
    p = np.array([0.05, 0.15, 0.95, 1.0])
    bins = calibration_bins(p, np.array([False, False, True, True]), n_bins=10)
    assert sum(b.count for b in bins) == 4
    assert bins[-1].count == 2
    assert bins[-1].observed == 1.0
