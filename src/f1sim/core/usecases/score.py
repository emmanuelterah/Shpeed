"""Score a published prediction against the official result (P8-07).

The scoring run reads back the record exactly as it was published, lines its drivers up with
the official classification and fills in the ``score`` field, leaving the forecast untouched.
The finishing positions are stored with the score, so the season scoreboard and calibration
chart can be rebuilt from the published records alone. Scoring a scored record is a no-op.
"""

from collections.abc import Sequence
from datetime import datetime

from f1sim.core.domain.race import DNF, encode_outcome
from f1sim.core.domain.record import PredictionRecord, RecordScore
from f1sim.core.scoring.race_score import score_race


def score_record(
    record: PredictionRecord,
    classified: Sequence[tuple[str, int]],
    *,
    scored_at: datetime,
) -> PredictionRecord:
    """Score one prediction record against the official classification.

    Args:
        record: The published record, in its forecast order.
        classified: ``(driver_id, official_position)`` for every classified finisher.
        scored_at: When the score was computed.

    Returns:
        A copy of the record with its ``score`` filled in; an already-scored record is
        returned unchanged.
    """
    if record.score is not None:
        return record
    outcome = encode_outcome(record.entries(), classified)
    score = score_race(record.season, record.round, record.matrix(), outcome)
    return record.with_score(
        RecordScore(
            brier_win=score.brier_win,
            brier_podium=score.brier_podium,
            brier_top10=score.brier_top10,
            log_loss=score.log_loss,
            rps=score.rps,
            scored_at=scored_at,
            positions=tuple(None if int(p) == DNF else int(p) for p in outcome.tolist()),
        )
    )
