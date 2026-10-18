"""Contract suite every publisher must pass (Liskov substitution, P8-03).

A publisher stores records so they can be read back exactly, and writing what is already
stored changes nothing. The in-memory double, the local file publisher and the git
publisher all run the same tests.
"""

import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from f1sim.adapters.publishers.git_pages import GitPagesPublisher
from f1sim.adapters.publishers.local_file import LocalFilePublisher
from tests.records import MemoryRecords, make_record, make_score

pytestmark = pytest.mark.contract


def init_repo(path: Path) -> Path:
    """Create an empty git repository with a committer identity.

    Args:
        path: Folder to initialise.

    Returns:
        The same folder.
    """
    path.mkdir(parents=True, exist_ok=True)
    for args in (
        ("init", "--quiet"),
        ("config", "user.email", "ci@example.com"),
        ("config", "user.name", "Shpeed CI"),
        ("config", "commit.gpgsign", "false"),
    ):
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)
    return path


@pytest.fixture(params=["memory", "local_file", "git_pages"])
def publisher(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Any]:
    """Each publisher implementation in turn; every test below runs against all of them."""
    if request.param == "memory":
        yield MemoryRecords()
    elif request.param == "local_file":
        yield LocalFilePublisher(tmp_path / "predictions")
    else:
        repo = init_repo(tmp_path / "repo")
        yield GitPagesPublisher(repo / "predictions", repo=repo)


def test_nothing_is_published_at_first(publisher: Any) -> None:
    assert publisher.load(make_record().event) is None


def test_a_published_record_reads_back_exactly(publisher: Any) -> None:
    record = make_record()
    result = publisher.publish(record.to_mapping())
    assert result.changed
    assert result.location
    assert publisher.load(record.event) == record


def test_publishing_the_same_record_again_changes_nothing(publisher: Any) -> None:
    record = make_record()
    publisher.publish(record.to_mapping())
    assert not publisher.publish(record.to_mapping()).changed


def test_a_new_version_of_a_record_replaces_the_old(publisher: Any) -> None:
    record = make_record()
    publisher.publish(record.to_mapping())
    scored = record.with_score(make_score([2, 1, None]))
    assert publisher.publish(scored.to_mapping()).changed
    assert publisher.load(record.event) == scored


def test_races_are_kept_apart(publisher: Any) -> None:
    first, second = make_record(round_=1), make_record(round_=2, favourite=0.7)
    publisher.publish(first.to_mapping())
    publisher.publish(second.to_mapping())
    assert publisher.load(first.event) == first
    assert publisher.load(second.event) == second
