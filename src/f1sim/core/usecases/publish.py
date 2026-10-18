"""Validate and publish one prediction record under the lights-out lock (P8-03, ADR-0011).

The policy, in order:

1. Nothing published for the race yet: write the record, unless the race has already
   started; a forecast made after lights out is not a forecast.
2. Something published that says the same thing (:meth:`PredictionRecord.same_forecast`):
   keep the published forecast and its provenance, adding the new score if there is one.
   Re-running a round therefore yields the same file.
3. Something published that says something different: before lights out the new forecast
   replaces it (new data such as a grid penalty arrived; git history keeps the old one).
   After lights out the forecast is frozen and :class:`RecordLockedError` is raised.

Every record is validated against the shared schema before it is written (ACID consistency),
and the publisher writes it atomically (atomicity, durability).
"""

from datetime import datetime

from f1sim.core.domain.record import PredictionRecord
from f1sim.core.errors import RecordLockedError
from f1sim.core.ports.publisher import Publisher, PublishResult, RecordReader, RecordValidator


def is_locked(lights_out: datetime | None, now: datetime) -> bool:
    """Whether a race's forecast is frozen.

    Args:
        lights_out: Scheduled race start (naive UTC), or None when the calendar has none.
        now: The current time (naive UTC).

    Returns:
        True once the race has started.
    """
    return lights_out is not None and now >= lights_out


def publish_record(  # pylint: disable=too-many-arguments
    record: PredictionRecord,
    *,
    validator: RecordValidator,
    records: RecordReader,
    publisher: Publisher,
    lights_out: datetime | None,
    now: datetime,
) -> PublishResult:
    """Validate a record and write it, following the lights-out policy.

    Args:
        record: The record to publish: a new forecast, or a published one with its score.
        validator: Checks the record against the shared JSON Schema.
        records: Reads back what is already published.
        publisher: Writes the record atomically and idempotently.
        lights_out: Scheduled race start (naive UTC); None when unknown, in which case a
            published forecast is never replaced.
        now: The current time (naive UTC).

    Returns:
        Where the record was written and whether anything changed.

    Raises:
        RecordLockedError: If the forecast would be published or changed after lights out.
        ValidationError: If the record does not match the shared schema.
    """
    existing = records.load(record.event)
    if existing is not None and existing.same_forecast(record):
        record = existing if record.score is None else existing.with_score(record.score)
    elif existing is not None and (lights_out is None or is_locked(lights_out, now)):
        raise RecordLockedError(
            f"{_race(record)} already has a published forecast and it is locked; "
            "records never change after lights out"
        )
    elif existing is None and is_locked(lights_out, now):
        raise RecordLockedError(f"{_race(record)} has started; too late to publish a forecast")
    mapping = record.to_mapping()
    validator.validate(mapping)
    return publisher.publish(mapping)


def _race(record: PredictionRecord) -> str:
    return f"{record.season} round {record.round}"
