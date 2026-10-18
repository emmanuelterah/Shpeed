"""Natural experiments for the regulation-reset study (P7-01).

A regulation reset is a natural experiment: the rules change, and the questions are how much
of the old era still helps and how fast the model becomes accurate again. This entity names
one such reset and the data it needs; the loader lives in the adapters layer, and the study
in ``f1sim.reports.recovery``.
"""

from dataclasses import dataclass

LAP_LEVEL = "lap"
RESULTS_LEVEL = "results"
LEVELS = (LAP_LEVEL, RESULTS_LEVEL)
"""Data levels a natural experiment can be studied at."""


@dataclass(frozen=True)
class NaturalExperiment:
    """One regulation change the study treats as a natural experiment.

    Attributes:
        experiment_id: Stable id, such as ``reg-2022``.
        name: Human-readable name.
        level: ``lap`` when lap timing is available, else ``results``.
        seasons: Seasons the study reads, the pre-reset season included.
        reset_season: First season of the new era.
        reset_round: First round of the new era.
        snapshot: Pinned data snapshot, or None until the data is synced.
        notes: Free text on data availability or method.
    """

    experiment_id: str
    name: str
    level: str
    seasons: tuple[int, ...]
    reset_season: int
    reset_round: int = 1
    snapshot: int | None = None
    notes: str = ""

    @property
    def pinned(self) -> bool:
        """Whether the experiment's data is pinned to a snapshot and so runnable.

        Returns:
            True when a snapshot is recorded.
        """
        return self.snapshot is not None
