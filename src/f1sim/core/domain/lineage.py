"""Team lineage (P1-06): one entity per team across rebrands, such as Sauber to Audi."""

from collections.abc import Mapping, Sequence

from f1sim.core.errors import InvariantError


class TeamLineage:
    """Maps constructor IDs to the lineage (continuing team) they belong to."""

    def __init__(self, lineages: Mapping[str, Sequence[str]]) -> None:
        """Index every constructor ID by its lineage.

        Args:
            lineages: Lineage ID to the constructor IDs it covers, oldest first.

        Raises:
            InvariantError: If a constructor ID appears in more than one lineage.
        """
        self._of: dict[str, str] = {}
        for lineage, members in lineages.items():
            for team_id in members:
                if team_id in self._of:
                    raise InvariantError(f"{team_id!r} is in two lineages")
                self._of[team_id] = lineage

    def lineage_of(self, team_id: str) -> str:
        """Find the lineage a constructor belongs to.

        Args:
            team_id: A Jolpica constructor ID, such as ``"alfa"``.

        Returns:
            The lineage ID; a constructor that is not listed is its own lineage.
        """
        return self._of.get(team_id, team_id)

    def members(self, lineage: str) -> list[str]:
        """List the constructor IDs in a lineage.

        Args:
            lineage: A lineage ID, such as ``"hinwil"``.

        Returns:
            Its constructor IDs in the order they were listed; empty if unknown.
        """
        return [t for t, line in self._of.items() if line == lineage]
