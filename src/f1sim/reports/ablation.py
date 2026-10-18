"""Ablation runner and table (P6-09): switch each event component off and score the change.

The Phase 6 gate is that no component is kept if it makes the backtest worse. The runner walks
forward the fully-featured simulator, then once per component with that component disabled (which
also restores the Phase 5 behaviour where one existed: disabling ``reliability`` returns to
pre-start DNFs at the league rate). A component whose absence *lowers* the ranked probability
score was costing accuracy and should not be kept, so the verdict flags it.

The table is published to ``docs/ablation.md``; ``f1sim report ablation --check`` exits non-zero
when a kept component is harming the backtest.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from f1sim.adapters.models.registry import ModelOptions, get_model
from f1sim.core.domain.entities import SnapshotId
from f1sim.core.ports.store import DataReader
from f1sim.core.usecases.backtest import walk_forward

ABLATABLE = ("safety_car", "reliability", "lap1", "penalties", "traffic", "rain")
"""Event components checked one at a time; the order they are reported in."""

LABELS: dict[str, str] = {
    "safety_car": "Safety car and VSC (P6-01)",
    "reliability": "Reliability (P6-03, P6-04)",
    "lap1": "Lap 1 (P6-05)",
    "penalties": "In-race penalties (P6-06)",
    "traffic": "Traffic and defending (P6-08)",
    "rain": "Rain regime (P6-07)",
}
"""Component name to its row label."""

MODEL = "simulator"
HARMFUL_TOLERANCE = 1e-3
"""RPS improvement from disabling a component that still counts as noise rather than a gain."""


@dataclass(frozen=True)
class AblationRow:
    """One component's effect on the backtest.

    Attributes:
        component: Component name, as in ``ABLATABLE``.
        label: Human-readable label for the report.
        rps: Mean RPS with the component disabled.
        delta_rps: ``rps`` minus the fully-featured RPS; negative means disabling helped.
        harmful: Whether disabling the component improved RPS beyond the tolerance.
    """

    component: str
    label: str
    rps: float
    delta_rps: float
    harmful: bool


@dataclass(frozen=True)
class AblationResult:
    """The full table and its baseline.

    Attributes:
        seasons: Seasons the backtest covered.
        snapshot: Data snapshot every run read.
        baseline_rps: Mean RPS of the fully-featured simulator.
        rows: One row per ablated component.
    """

    seasons: tuple[int, ...]
    snapshot: SnapshotId
    baseline_rps: float
    rows: list[AblationRow] = field(default_factory=list[AblationRow])

    def harmful(self) -> tuple[str, ...]:
        """Components that should not be kept, because removing them improves the score.

        Returns:
            The harmful components' names, in report order.
        """
        return tuple(row.component for row in self.rows if row.harmful)


def run_ablation(
    reader: DataReader,
    *,
    snapshot: SnapshotId,
    seasons: Sequence[int],
    seed: int = 0,
    options: ModelOptions | None = None,
    workers: int = 4,
    components: Sequence[str] = ABLATABLE,
) -> AblationResult:
    """Backtest the full simulator, then each component with it switched off (P6-09).

    Args:
        reader: The data store.
        snapshot: Snapshot to read.
        seasons: Seasons to forecast.
        seed: Base seed, shared by every run so the comparison is on identical races.
        options: Simulator settings; the registry defaults when None. Its ``disabled`` is
            overridden per run, and its other settings (runs per race, pace hyperparameters)
            are shared by every run.
        workers: Races forecast in parallel.
        components: Which components to ablate, in report order.

    Returns:
        The baseline score and one row per component.
    """
    base = options or ModelOptions()
    baseline = _score(
        reader, snapshot=snapshot, seasons=seasons, seed=seed, options=base, workers=workers
    )
    rows: list[AblationRow] = []
    for component in components:
        disabled = frozenset({*base.disabled, component})
        rps = _score(
            reader,
            snapshot=snapshot,
            seasons=seasons,
            seed=seed,
            options=replace(base, disabled=disabled),
            workers=workers,
        )
        delta = rps - baseline
        rows.append(
            AblationRow(
                component=component,
                label=LABELS.get(component, component),
                rps=rps,
                delta_rps=delta,
                harmful=delta < -HARMFUL_TOLERANCE,
            )
        )
    return AblationResult(
        seasons=tuple(seasons), snapshot=snapshot, baseline_rps=baseline, rows=rows
    )


def _score(
    reader: DataReader,
    *,
    snapshot: SnapshotId,
    seasons: Sequence[int],
    seed: int,
    options: ModelOptions,
    workers: int,
) -> float:
    result = walk_forward(
        reader,
        snapshot=snapshot,
        seasons=seasons,
        model=get_model(MODEL, options),
        seed=seed,
        workers=workers,
    )
    summary = result.summary()
    return float(summary.get("rps", np.nan))


def render_ablation(result: AblationResult) -> str:
    """Render the ablation table as Markdown.

    Args:
        result: Output of ``run_ablation``.

    Returns:
        The report.
    """
    seasons = ", ".join(str(s) for s in result.seasons) or "none"
    harmful = result.harmful()
    verdict = (
        f"Verdict: drop {', '.join(harmful)} (removing them improves RPS)."
        if harmful
        else "Verdict: every component earns its place."
    )
    lines = [
        "# Event-component ablation (P6-09)",
        "",
        f"Seasons {seasons}, snapshot {result.snapshot}. The fully-featured simulator scores "
        f"RPS {result.baseline_rps:.4f}; each row switches one component off. A negative delta "
        "means the component was making the backtest worse.",
        "",
        verdict,
        "",
        "| Component off | RPS without | Delta RPS | Verdict |",
        "|---|---:|---:|---|",
    ]
    for row in result.rows:
        mark = "harmful" if row.harmful else "kept"
        lines.append(f"| {row.label} | {row.rps:.4f} | {row.delta_rps:+.4f} | {mark} |")
    lines += [
        "",
        f"Deltas within ±{HARMFUL_TOLERANCE:.0e} RPS are inside the run-to-run noise band, so a",
        "component flagged at that edge is a marginal finding rather than a clear loss.",
    ]
    return "\n".join(lines) + "\n"
