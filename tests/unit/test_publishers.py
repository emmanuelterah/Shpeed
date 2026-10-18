"""Publisher adapters: atomic files, the rebuilt index and one git commit per publish (P8-03)."""

import json
import subprocess
from pathlib import Path

import pytest

from f1sim.adapters.publishers.git_pages import GitPagesPublisher, commit_message
from f1sim.adapters.publishers.index import (
    published_records,
    read_json,
    rebuild_index,
    record_path,
    write_atomic,
)
from f1sim.adapters.publishers.local_file import LocalFilePublisher
from f1sim.adapters.records import validate_index, validate_record
from f1sim.core.domain.entities import EventKey
from f1sim.core.errors import DataError
from tests.contract.test_publisher_contract import init_repo
from tests.records import make_record, make_score


def _log(repo: Path) -> list[str]:
    """Commit subjects, newest first."""
    done = subprocess.run(
        ["git", "log", "--format=%s"], cwd=repo, check=False, capture_output=True, text=True
    )
    return done.stdout.split("\n")[:-1] if done.returncode == 0 else []


def _files_in_last_commit(repo: Path) -> set[str]:
    """Paths changed by HEAD."""
    done = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return set(done.stdout.split())


def test_publish_writes_a_valid_record_and_index(tmp_path: Path) -> None:
    result = LocalFilePublisher(tmp_path).publish(make_record().to_mapping())
    assert Path(result.location) == record_path(tmp_path, 2024, 3)
    validate_record(read_json(Path(result.location)))
    index = read_json(tmp_path / "index.json")
    validate_index(index)
    assert index["latest"] == "2024/03.json"
    assert index["races"][0]["path"] == "2024/03.json"


def test_files_end_with_a_newline_and_use_two_space_indents(tmp_path: Path) -> None:
    LocalFilePublisher(tmp_path).publish(make_record().to_mapping())
    text = record_path(tmp_path, 2024, 3).read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert '\n  "schema_version": 1' in text


def test_scoring_updates_the_index(tmp_path: Path) -> None:
    publisher = LocalFilePublisher(tmp_path)
    publisher.publish(make_record().to_mapping())
    scored = make_record().with_score(make_score([1, 2, 3], rps=0.3))
    publisher.publish(scored.to_mapping())
    index = read_json(tmp_path / "index.json")
    assert index["races"][0]["score"]["rps"] == 0.3
    assert index["calibration"]["races_scored"] == 1


def test_an_unreadable_record_is_left_out_of_the_index(tmp_path: Path) -> None:
    (tmp_path / "2024").mkdir()
    (tmp_path / "2024" / "01.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "2024" / "02.json").write_text('{"season": 2024}', encoding="utf-8")
    LocalFilePublisher(tmp_path).publish(make_record().to_mapping())
    assert [path for path, _ in published_records(tmp_path)] == ["2024/03.json"]
    path, changed = rebuild_index(tmp_path)
    assert not changed
    assert [r["round"] for r in read_json(path)["races"]] == [3]


def test_a_damaged_record_cannot_be_loaded_as_if_missing(tmp_path: Path) -> None:
    (tmp_path / "2024").mkdir()
    (tmp_path / "2024" / "03.json").write_text('{"season": 2024}', encoding="utf-8")
    with pytest.raises(DataError):
        LocalFilePublisher(tmp_path).load(EventKey(2024, 3))


def test_write_atomic_leaves_no_temporary_files(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "file.json"
    write_atomic(target, '{"ok": true}\n')
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
    assert not list(tmp_path.rglob(".tmp-*"))


def test_write_atomic_keeps_the_old_file_when_the_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "file.json"
    write_atomic(target, "old\n")

    def broken(src: str, dst: Path) -> None:
        """Fail the way a full disk or a locked file would."""
        raise OSError(f"cannot replace {dst} with {src}")

    monkeypatch.setattr("f1sim.adapters.publishers.index.os.replace", broken)
    with pytest.raises(OSError, match="cannot replace"):
        write_atomic(target, "new\n")
    assert target.read_text(encoding="utf-8") == "old\n"
    assert not list(tmp_path.glob(".tmp-*"))


def test_git_publisher_commits_record_and_index_together(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    publisher = GitPagesPublisher(repo / "predictions", repo=repo)
    assert publisher.publish(make_record().to_mapping()).changed
    assert _log(repo) == ["chore(predictions): publish 2024 round 3 forecast (P8-06)"]
    assert _files_in_last_commit(repo) == {"predictions/2024/03.json", "predictions/index.json"}


def test_git_publisher_makes_no_commit_when_nothing_changed(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    publisher = GitPagesPublisher(repo / "predictions", repo=repo)
    publisher.publish(make_record().to_mapping())
    assert not publisher.publish(make_record().to_mapping()).changed
    assert len(_log(repo)) == 1


def test_a_retry_commits_files_an_earlier_run_wrote_but_did_not_commit(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    GitPagesPublisher(repo / "predictions", commit=False, repo=repo).publish(
        make_record().to_mapping()
    )
    assert _log(repo) == []
    retry = GitPagesPublisher(repo / "predictions", repo=repo).publish(make_record().to_mapping())
    assert retry.changed
    assert len(_log(repo)) == 1


def test_scoring_is_its_own_commit(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    publisher = GitPagesPublisher(repo / "predictions", repo=repo)
    publisher.publish(make_record().to_mapping())
    publisher.publish(make_record().with_score(make_score([1, 2, 3])).to_mapping())
    assert _log(repo)[0] == "chore(predictions): score 2024 round 3 (P8-07)"


def test_other_staged_files_are_not_swept_into_the_commit(tmp_path: Path) -> None:
    repo = init_repo(tmp_path)
    (repo / "notes.txt").write_text("draft\n", encoding="utf-8")
    subprocess.run(["git", "add", "notes.txt"], cwd=repo, check=True)
    GitPagesPublisher(repo / "predictions", repo=repo).publish(make_record().to_mapping())
    assert "notes.txt" not in _files_in_last_commit(repo)


def test_a_git_failure_is_an_error_not_a_warning(tmp_path: Path) -> None:
    publisher = GitPagesPublisher(tmp_path / "predictions", repo=tmp_path)
    with pytest.raises(DataError, match="could not commit"):
        publisher.publish(make_record().to_mapping())


def test_commit_message_names_the_race() -> None:
    assert "2024 round 3" in commit_message(make_record().to_mapping())
