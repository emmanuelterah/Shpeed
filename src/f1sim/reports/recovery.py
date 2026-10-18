"""Recovery curves after a regulation reset (P7-03).

For one natural experiment (P7-01), the study walks the pace model forward over the seasons
around the reset, once per prior-handling strategy (P7-02), and scores every race. The
recovery curve is RPS by race number after the reset, read against the pre-reset level: the
question is how quickly each strategy gets back to where the old era left it.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from f1sim.adapters.models.registry import ModelOptions, get_model
from f1sim.adapters.models.resets import STRATEGIES, ResetConfig
from f1sim.core.domain.entities import SnapshotId
from f1sim.core.domain.natural_experiment import NaturalExperiment
from f1sim.core.ports.store import DataReader
from f1sim.core.usecases.backtest import walk_forward

MODEL = "pace"
"""Model the study scores; the reset is a pace-model prior, so this is where it shows."""

STRATEGY_LABELS: dict[str, str] = {
    "full": "Full history",
    "decay": "Exponential decay",
    "reset": "Reset priors",
    "testing": "Testing-informed priors",
}
"""Row label for each strategy, in ``STRATEGIES`` order."""

MILESTONES = (1, 2, 3, 5, 10)
"""Race numbers the summary table reports explicitly."""


@dataclass(frozen=True)
class RecoveryPoint:
    """One race's score on a strategy's curve.

    Attributes:
        race: Race number after the reset, 1 for the reset race.
        rps: Mean ranked probability score of that race.
    """

    race: int
    rps: float


@dataclass(frozen=True)
class StrategyCurve:
    """A strategy's recovery, race by race after the reset.

    Attributes:
        strategy: Strategy name, as in ``STRATEGIES``.
        label: Human-readable label for the report.
        pre_reset_rps: Mean RPS over the pre-reset season, or None when it was not scored.
        points: One point per race after the reset, in race order.
    """

    strategy: str
    label: str
    pre_reset_rps: float | None
    points: list[RecoveryPoint] = field(default_factory=list[RecoveryPoint])

    def recovery_race(self, target: float) -> int | None:
        """First race number whose RPS is at or below ``target``.

        Args:
            target: The level to reach, usually the pre-reset RPS.

        Returns:
            The race number, or None if the curve never reaches the target.
        """
        return next((p.race for p in self.points if p.rps <= target), None)


@dataclass(frozen=True)
class RecoveryStudy:
    """Every strategy's curve for one natural experiment.

    Attributes:
        experiment: The experiment's id.
        name: The experiment's human-readable name.
        level: ``lap`` or ``results``.
        reset_season: First season of the new era.
        seasons: Seasons the study covered.
        snapshot: Data snapshot every run read.
        curves: One curve per strategy, in ``STRATEGIES`` order.
    """

    experiment: str
    name: str
    level: str
    reset_season: int
    seasons: tuple[int, ...]
    snapshot: SnapshotId
    curves: list[StrategyCurve] = field(default_factory=list[StrategyCurve])

    def fastest(self) -> str | None:
        """Strategy that returns to its pre-reset level in the fewest races.

        Ties are broken by the lower RPS at the deciding race, then by name.

        Returns:
            The winning strategy's name, or None when no curve reaches its reference.
        """
        scored: list[tuple[int, float, str]] = []
        for curve in self.curves:
            if curve.pre_reset_rps is None:
                continue
            race = curve.recovery_race(curve.pre_reset_rps)
            if race is not None:
                scored.append((race, curve.points[race - 1].rps, curve.strategy))
        return min(scored)[2] if scored else None


def run_recovery(
    reader: DataReader,
    *,
    snapshot: SnapshotId,
    experiment: NaturalExperiment,
    strategies: Sequence[str] = STRATEGIES,
    seed: int = 0,
    options: ModelOptions | None = None,
    workers: int = 4,
) -> RecoveryStudy:
    """Score each reset strategy race by race around one natural experiment (P7-03).

    Args:
        reader: The data store.
        snapshot: Snapshot to read.
        experiment: The regulation reset to study.
        strategies: Strategies to compare, in report order.
        seed: Base seed, shared so every strategy sees the same races.
        options: Base model options; its ``reset`` carries the knobs, and its strategy is
            replaced per run.
        workers: Races forecast in parallel.

    Returns:
        One curve per strategy, plus the pre-reset reference.
    """
    base = options or ModelOptions()
    curves = [
        _curve(
            reader,
            snapshot=snapshot,
            experiment=experiment,
            strategy=name,
            seed=seed,
            base=base,
            workers=workers,
        )
        for name in strategies
    ]
    return RecoveryStudy(
        experiment=experiment.experiment_id,
        name=experiment.name,
        level=experiment.level,
        reset_season=experiment.reset_season,
        seasons=experiment.seasons,
        snapshot=snapshot,
        curves=curves,
    )


def _curve(
    reader: DataReader,
    *,
    snapshot: SnapshotId,
    experiment: NaturalExperiment,
    strategy: str,
    seed: int,
    base: ModelOptions,
    workers: int,
) -> StrategyCurve:
    knobs = base.reset or ResetConfig()
    reset = replace(knobs, strategy=strategy, reset_season=experiment.reset_season)
    result = walk_forward(
        reader,
        snapshot=snapshot,
        seasons=experiment.seasons,
        model=get_model(MODEL, replace(base, reset=reset)),
        seed=seed,
        workers=workers,
    )
    pre = [r.rps for r in result.races if r.season < experiment.reset_season]
    post = [
        r
        for r in result.races
        if r.season == experiment.reset_season and r.round >= experiment.reset_round
    ]
    points = [
        RecoveryPoint(r.round - experiment.reset_round + 1, float(r.rps))
        for r in sorted(post, key=lambda r: r.round)
    ]
    return StrategyCurve(
        strategy=strategy,
        label=STRATEGY_LABELS.get(strategy, strategy),
        pre_reset_rps=float(np.mean(pre)) if pre else None,
        points=points,
    )


def render_recovery(study: RecoveryStudy) -> str:
    """Render a recovery study as Markdown (P7-03).

    Args:
        study: Output of ``run_recovery``.

    Returns:
        The report: the setup, a milestone summary and the full per-race table.
    """
    seasons = ", ".join(str(s) for s in study.seasons)
    fastest = study.fastest()
    verdict = (
        f"Fastest back to the pre-reset level: {STRATEGY_LABELS.get(fastest, fastest)} ({fastest})."
        if fastest
        else "No strategy returned to its pre-reset level within the races scored."
    )
    lines = [
        f"# Regulation-reset recovery: {study.name} ({study.experiment})",
        "",
        f"Seasons {seasons} at the {study.level} level, snapshot {study.snapshot}. "
        f"The reset is season {study.reset_season}; race 1 is its first round.",
        "",
        verdict,
        "",
        "## Milestones",
        "",
        *_table(
            ["Strategy", "Pre-reset RPS", *(f"Race {r}" for r in MILESTONES), "Back to level at"],
            [_milestone_row(curve) for curve in study.curves],
        ),
        "",
        "## Per race",
        "",
        *_table(
            ["Race", *(curve.label for curve in study.curves)],
            _race_rows(study),
        ),
    ]
    return "\n".join(lines) + "\n"


def _milestone_row(curve: StrategyCurve) -> list[str]:
    by_race = {point.race: point.rps for point in curve.points}
    reached = curve.recovery_race(curve.pre_reset_rps) if curve.pre_reset_rps is not None else None
    return [
        curve.label,
        _fmt(curve.pre_reset_rps),
        *(_fmt(by_race.get(r)) for r in MILESTONES),
        str(reached) if reached is not None else "-",
    ]


def _race_rows(study: RecoveryStudy) -> list[list[str]]:
    races = sorted({point.race for curve in study.curves for point in curve.points})
    scores = [{p.race: p.rps for p in curve.points} for curve in study.curves]
    return [[str(r), *(_fmt(points.get(r)) for points in scores)] for r in races]


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(row) + " |" for row in rows]


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"
