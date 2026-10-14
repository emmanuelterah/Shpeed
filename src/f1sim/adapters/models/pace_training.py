"""Fit the pace model end to end: observations, NUTS hyperparameters, race-day noise.

``train_pace`` is what ``f1sim pace fit`` runs. It reads the training seasons, fits the
hyperparameters with NUTS (``pace_hyper``), then picks the race-day ranking noise by
backtesting those seasons (``pace_checks``), and returns everything needed to write a
reproducible record.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from f1sim.adapters.models.pace import PaceContext
from f1sim.adapters.models.pace_checks import calibrate_order_noise
from f1sim.adapters.models.pace_data import SessionObs, build_observations
from f1sim.adapters.models.pace_hyper import HyperFit, build_panel, fit_hyper
from f1sim.core.domain.entities import SnapshotId
from f1sim.core.ports.store import DataReader
from f1sim.core.usecases.readers import FAR_FUTURE


@dataclass(frozen=True)
class SamplerSettings:
    """NUTS settings.

    Attributes:
        warmup: Warm-up draws per chain.
        samples: Kept draws per chain.
        chains: Chains; at least two for R-hat.
        seed: Random seed.
        method: ``nuts`` or ``laplace`` (see ``pace_hyper.fit_hyper``).
    """

    warmup: int = 500
    samples: int = 500
    chains: int = 2
    seed: int = 0
    method: str = "nuts"


@dataclass(frozen=True)
class TrainingResult:
    """A fitted pace model.

    Attributes:
        fit: The NUTS fit, with ``order_noise`` set to the chosen value.
        order_noise_rps: Backtest RPS of each race-day noise value tried.
        seasons: Training seasons.
        snapshot: Snapshot the data came from.
    """

    fit: HyperFit
    order_noise_rps: dict[float, float]
    seasons: tuple[int, ...]
    snapshot: SnapshotId

    def record(self, commit: str) -> dict[str, Any]:
        """Describe the fit for ``configs/pace_hyper.json``.

        Args:
            commit: Git commit of the code that ran the fit.

        Returns:
            A JSON-compatible record; ``hyper`` is what ``load_pace_hyper`` reads.
        """
        return {
            "hyper": self.fit.hyper.to_dict(),
            "diagnostics": {k: vars(v) for k, v in self.fit.diagnostics.items()},
            "method": self.fit.method,
            "converged": self.fit.converged,
            "seconds": round(self.fit.seconds, 1),
            "observations": self.fit.observations,
            "seasons": list(self.seasons),
            "snapshot_id": int(self.snapshot),
            "git_commit": commit,
            "order_noise_rps": {str(k): v for k, v in self.order_noise_rps.items()},
        }


def season_observations(
    reader: DataReader, snapshot: SnapshotId, seasons: Sequence[int], context: PaceContext
) -> list[SessionObs]:
    """Build pace observations for some whole seasons.

    Args:
        reader: The data store.
        snapshot: Snapshot to read.
        seasons: Seasons to include.
        context: Reference tables.

    Returns:
        Observations in calendar order.
    """
    laps = reader.read("laps", snapshot=snapshot, as_of=FAR_FUTURE)
    laps = laps.loc[laps["season"].isin(list(seasons))]
    events = reader.read("events", snapshot=snapshot, as_of=FAR_FUTURE)
    return build_observations(laps, events, context.circuits)


def train_pace(
    reader: DataReader,
    *,
    snapshot: SnapshotId,
    seasons: Sequence[int],
    context: PaceContext,
    sampler: SamplerSettings | None = None,
) -> TrainingResult:
    """Fit hyperparameters with NUTS, then race-day noise by backtest.

    Args:
        reader: The data store.
        snapshot: Snapshot to read.
        seasons: Training seasons; the noise is chosen on all but the first, which only
            gives the model history.
        context: Reference tables and starting hyperparameters.
        sampler: NUTS settings.

    Returns:
        The fit and the noise search.
    """
    settings = sampler or SamplerSettings()
    observations = season_observations(reader, snapshot, seasons, context)
    panel = build_panel(observations, context.lineage, context.suppliers)
    fit = fit_hyper(
        panel,
        base=context.hyper,
        method=settings.method,
        sampler=(settings.warmup, settings.samples, settings.chains),
        seed=settings.seed,
    )
    best, scores = calibrate_order_noise(
        reader,
        snapshot=snapshot,
        seasons=list(seasons[1:]) or list(seasons),
        context=context.with_hyper(fit.hyper),
    )
    tuned = replace(fit, hyper=replace(fit.hyper, order_noise=best))
    return TrainingResult(tuned, scores, tuple(seasons), snapshot)
