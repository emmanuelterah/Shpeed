"""The predictions index: season scoreboard and calibration, rebuilt from records (P8-07, P8-10).

The static site has no server, so everything its Season and Calibration screens show is
precomputed here into ``predictions/index.json`` each time a record is published or scored.
The index is derived entirely from the published records, so it can always be rebuilt and
never disagrees with them. Its shape is ``schemas/predictions-index.v1.schema.json``.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from f1sim.core.domain.race import DNF
from f1sim.core.domain.record import PredictionRecord, RecordScore
from f1sim.core.scoring.metrics import calibration_bins
from f1sim.core.scoring.race_score import EVENTS, event_forecasts

INDEX_SCHEMA_VERSION = 1
"""Version of ``schemas/predictions-index.v1.schema.json``."""

EVENT_LABELS: dict[str, str] = {"win": "Win", "podium": "Podium", "top10": "Points"}
"""Reader-facing names of the calibrated events, matching the race screen's views."""

CALIBRATION_BINS = 10


def build_index(published: Sequence[tuple[str, PredictionRecord]]) -> dict[str, Any]:
    """Summarise every published record for the static site.

    Args:
        published: ``(path, record)`` pairs, the path relative to the predictions folder,
            such as ``"2024/03.json"``. Order does not matter.

    Returns:
        A JSON-ready mapping valid against the predictions-index schema: every race, oldest
        first, with its score; the latest race's path; and calibration over every scored
        race.
    """
    ordered = sorted(published, key=lambda item: (item[1].season, item[1].round))
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "latest": ordered[-1][0] if ordered else None,
        "races": [_race_row(path, record) for path, record in ordered],
        "calibration": calibration([record for _, record in ordered]),
    }


def calibration(records: Sequence[PredictionRecord]) -> dict[str, Any]:
    """Predicted against observed frequency, pooled over every scored record.

    Args:
        records: Published records; unscored ones are ignored.

    Returns:
        The number of scored races and forecasts, and for each event (win, podium, points)
        the non-empty probability bins with their mean forecast and observed frequency.
    """
    scored = [(r, r.score) for r in records if r.score is not None]
    pooled: dict[str, tuple[list[NDArray[np.float64]], list[NDArray[np.bool_]]]] = {
        name: ([], []) for name in EVENTS
    }
    for record, score in scored:
        outcome = _outcome(score)
        for name, forecast in event_forecasts(record.matrix(), outcome).items():
            pooled[name][0].append(forecast.p)
            pooled[name][1].append(forecast.happened)
    events: list[dict[str, Any]] = []
    for name, (ps, hits) in pooled.items():
        bins = (
            calibration_bins(np.concatenate(ps), np.concatenate(hits), CALIBRATION_BINS)
            if ps
            else []
        )
        events.append(
            {
                "event": name,
                "label": EVENT_LABELS[name],
                "bins": [
                    {
                        "low": round(b.low, 6),
                        "high": round(b.high, 6),
                        "mean_forecast": round(b.mean_forecast, 6),
                        "observed": round(b.observed, 6),
                        "count": b.count,
                    }
                    for b in bins
                ],
            }
        )
    return {
        "races_scored": len(scored),
        "forecasts": sum(len(record.drivers) for record, _ in scored),
        "events": events,
    }


def _race_row(path: str, record: PredictionRecord) -> dict[str, Any]:
    mapping = record.to_mapping()
    score: dict[str, Any] | None = mapping["score"]
    row: dict[str, Any] = {
        "season": record.season,
        "round": record.round,
        "path": path,
        "data_cutoff": mapping["data_cutoff"],
        "model_version": record.model_version,
        "n_drivers": len(record.drivers),
        "score": None if score is None else {k: v for k, v in score.items() if k != "positions"},
    }
    if record.event_name is not None:
        row["event_name"] = record.event_name
    return row


def _outcome(score: RecordScore) -> NDArray[np.int_]:
    return np.array([DNF if p is None else p for p in score.positions], dtype=np.int64)
