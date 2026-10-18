"""The lights-out publishing policy (P8-03, ADR-0011), against the in-memory double."""

from datetime import datetime, timedelta

import pytest

from f1sim.adapters.records import JsonSchemaValidator
from f1sim.core.domain.record import PredictionRecord
from f1sim.core.errors import RecordLockedError, ValidationError
from f1sim.core.ports.publisher import PublishResult
from f1sim.core.usecases.publish import is_locked, publish_record
from tests.records import LIGHTS_OUT, MemoryRecords, make_record, make_score

BEFORE = LIGHTS_OUT - timedelta(hours=20)
AFTER = LIGHTS_OUT + timedelta(minutes=1)


def _publish(
    store: MemoryRecords,
    record: PredictionRecord | None = None,
    *,
    now: datetime = BEFORE,
    lights_out: datetime | None = LIGHTS_OUT,
) -> PublishResult:
    """Publish ``record`` (the default test record when None) into ``store``."""
    return publish_record(
        record or make_record(),
        validator=JsonSchemaValidator(),
        records=store,
        publisher=store,
        lights_out=lights_out,
        now=now,
    )


def test_first_publish_writes_the_record() -> None:
    store = MemoryRecords()
    assert _publish(store).changed
    assert store.writes == 1


def test_rerunning_the_same_forecast_is_a_no_op() -> None:
    store = MemoryRecords()
    _publish(store)
    assert not _publish(store).changed
    assert store.writes == 1


def test_new_provenance_with_the_same_probabilities_keeps_the_published_file() -> None:
    store = MemoryRecords()
    _publish(store)
    again = _publish(store, make_record(git_commit="1234567", snapshot=8))
    assert not again.changed
    stored = store.load(make_record().event)
    assert stored is not None
    assert stored.git_commit == "abcdef0"
    assert stored.snapshot_id == 7


def test_a_changed_forecast_replaces_the_record_before_lights_out() -> None:
    store = MemoryRecords()
    _publish(store)
    assert _publish(store, make_record(favourite=0.6, snapshot=8)).changed
    stored = store.load(make_record().event)
    assert stored is not None
    assert stored.snapshot_id == 8


def test_a_changed_forecast_is_rejected_after_lights_out() -> None:
    store = MemoryRecords()
    _publish(store)
    with pytest.raises(RecordLockedError):
        _publish(store, make_record(favourite=0.6), now=AFTER)


def test_a_changed_forecast_is_rejected_when_lights_out_is_unknown() -> None:
    store = MemoryRecords()
    _publish(store, lights_out=None)
    with pytest.raises(RecordLockedError):
        _publish(store, make_record(favourite=0.6), lights_out=None)


def test_a_first_forecast_after_lights_out_is_refused() -> None:
    with pytest.raises(RecordLockedError):
        _publish(MemoryRecords(), now=AFTER)


def test_the_score_can_be_added_after_lights_out() -> None:
    store = MemoryRecords()
    _publish(store)
    scored = make_record().with_score(make_score([1, 2, None]))
    assert _publish(store, scored, now=AFTER + timedelta(hours=3)).changed
    stored = store.load(scored.event)
    assert stored is not None
    assert stored.score == scored.score


def test_republishing_the_forecast_never_erases_a_score() -> None:
    store = MemoryRecords()
    scored = make_record().with_score(make_score([1, 2, None]))
    _publish(store, scored)
    assert not _publish(store).changed
    stored = store.load(scored.event)
    assert stored is not None
    assert stored.score is not None


def test_an_invalid_record_is_never_written() -> None:
    store = MemoryRecords()
    record = make_record(git_commit="not a commit")
    with pytest.raises(ValidationError):
        _publish(store, record)
    assert store.writes == 0


@pytest.mark.parametrize(
    ("lights_out", "now", "locked"),
    [(None, AFTER, False), (LIGHTS_OUT, BEFORE, False), (LIGHTS_OUT, LIGHTS_OUT, True)],
)
def test_is_locked(lights_out: datetime | None, now: datetime, locked: bool) -> None:
    assert is_locked(lights_out, now) is locked
