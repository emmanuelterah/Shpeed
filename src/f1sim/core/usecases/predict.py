"""Build a published prediction record for one race (P8-01, P8-02).

A live forecast and a backtest replay are the same code path: plan the race at its cutoff,
fit the model through a ``CutoffReader`` so it cannot see the future, predict, and wrap the
matrix and its provenance in a :class:`~f1sim.core.domain.record.PredictionRecord`.
"""

from f1sim.core.domain.entities import EventKey, SnapshotId
from f1sim.core.domain.record import PredictionRecord
from f1sim.core.errors import InvariantError
from f1sim.core.ports.predictor import ModelSpec
from f1sim.core.ports.store import DataReader
from f1sim.core.usecases.planning import plan_event
from f1sim.core.usecases.readers import CutoffReader, FrameReader


def predict_event(  # pylint: disable=too-many-arguments,too-many-locals
    reader: DataReader,
    *,
    snapshot: SnapshotId,
    model: ModelSpec,
    event: EventKey,
    seed: int,
    version: str,
    git_commit: str,
    n_sims: int,
    event_name: str | None = None,
    p_rain: float | None = None,
) -> PredictionRecord:
    """Fit a model at a race's cutoff and turn its forecast into a record.

    Args:
        reader: The data store.
        snapshot: Snapshot to read.
        model: The model to fit and predict with.
        event: The race weekend to forecast.
        seed: Random seed for the run.
        version: Version string recorded with the forecast.
        git_commit: Commit the run was built from.
        n_sims: Simulated races behind the forecast, recorded for provenance.
        event_name: Human-readable event name, when known.
        p_rain: Rain probability fed to the simulator, when used.

    Returns:
        The prediction record, ready to validate and publish.

    Raises:
        DataError: If the race has no grid to plan from.
        InvariantError: If the model returns drivers out of entry order.
        LeakageError: If the model reads past the cutoff.
    """
    plan = plan_event(reader, snapshot, event)
    tables = tuple(dict.fromkeys(("results", "laps", *model.tables)))
    frames = FrameReader.load(reader, snapshot, tables)
    predictor = model.factory(seed)
    predictor.fit(CutoffReader(frames, plan.cutoff), snapshot=snapshot, cutoff=plan.cutoff)
    matrix = predictor.predict(event, plan.entries)
    expected = tuple(entry.driver_id for entry in plan.entries)
    if matrix.drivers != expected:
        raise InvariantError(f"{predictor.name} returned drivers out of entry order")
    return PredictionRecord.from_matrix(
        matrix,
        plan.entries,
        season=event.season,
        round_=event.round,
        data_cutoff=plan.cutoff,
        snapshot_id=snapshot,
        model_version=version,
        git_commit=git_commit,
        n_sims=n_sims,
        seed=seed,
        event_name=event_name,
        p_rain=p_rain,
    )


def version_string(model_name: str, package_version: str) -> str:
    """Compose the version string stored on a record.

    Args:
        model_name: Registry name of the model, such as ``"simulator"``.
        package_version: The installed package version.

    Returns:
        A version string such as ``"simulator-0.0.0"``.
    """
    return f"{model_name}-{package_version}"
