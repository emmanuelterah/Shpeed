"""Team lineage (P1-06): one entity per team across rebrands (e.g. Sauber -> Alfa Romeo -> Audi)."""

from collections.abc import Mapping, Sequence

from f1sim.core.errors import InvariantError


class TeamLineage:
    def __init__(self, lineages: Mapping[str, Sequence[str]]) -> None:
        self._of: dict[str, str] = {}
        for lineage, members in lineages.items():
            for team_id in members:
                if team_id in self._of:
                    raise InvariantError(f"{team_id!r} is in two lineages")
                self._of[team_id] = lineage

    def lineage_of(self, team_id: str) -> str:
        """The lineage ID for a constructor; an unlisted constructor is its own lineage."""
        return self._of.get(team_id, team_id)

    def members(self, lineage: str) -> list[str]:
        return [t for t, line in self._of.items() if line == lineage]
