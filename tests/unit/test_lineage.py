import pytest

from f1sim.adapters.config_files import load_team_lineage
from f1sim.core.domain.lineage import TeamLineage
from f1sim.core.errors import InvariantError

# Constructor IDs used by Jolpica for every team, 2018-2025, plus the 2026 entries
ENTRIES_2018_2026 = {
    "enstone": ["renault", "alpine"],
    "faenza": ["toro_rosso", "alphatauri", "rb"],
    "hinwil": ["sauber", "alfa", "audi"],
    "silverstone": ["force_india", "racing_point", "aston_martin"],
}


def test_rebrands_resolve_to_one_lineage() -> None:
    lineage = load_team_lineage()
    for expected, ids in ENTRIES_2018_2026.items():
        assert {lineage.lineage_of(i) for i in ids} == {expected}


def test_every_2018_2026_constructor_has_a_lineage() -> None:
    lineage = load_team_lineage()
    for team in ["mercedes", "ferrari", "red_bull", "mclaren", "williams", "haas", "cadillac"]:
        assert lineage.lineage_of(team) != ""
    assert lineage.lineage_of("brand_new_team") == "brand_new_team"


def test_a_constructor_cannot_be_in_two_lineages() -> None:
    with pytest.raises(InvariantError):
        TeamLineage({"a": ["x"], "b": ["x"]})
