"""Writes records to predictions/ and commits them in one git commit (P8-03, P8-14).

The record and the rebuilt index land in a single commit, so a published forecast and the
scoreboard that lists it can never disagree, and git history is the durable log of every
forecast (ACID atomicity, durability). The commit step always runs, even when the files were
already up to date: if an earlier run wrote the files but failed to commit, the retry
commits them. When nothing is staged, nothing is committed. Pushing is left to the caller
(the race-weekend workflow), which owns the credentials.
"""

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from f1sim.adapters.publishers.index import INDEX_NAME, record_path
from f1sim.adapters.publishers.local_file import LocalFilePublisher
from f1sim.core.domain.entities import EventKey
from f1sim.core.domain.record import PredictionRecord
from f1sim.core.errors import DataError
from f1sim.core.ports.publisher import PublishResult

_GIT_TIMEOUT_S = 60
_NOTHING_STAGED = 0


class GitPagesPublisher:
    """Publishes records under a folder and commits the change."""

    def __init__(self, root: Path, *, commit: bool = True, repo: Path | None = None) -> None:
        """Create a publisher.

        Args:
            root: The predictions folder.
            commit: Whether to stage and commit the written files.
            repo: Git working tree to commit in; the current directory when None.
        """
        self._files = LocalFilePublisher(root)
        self._commit = commit
        self._repo = repo if repo is not None else Path()
        self._root = root

    def load(self, event: EventKey) -> PredictionRecord | None:
        """Read one race's published record.

        Args:
            event: The race weekend.

        Returns:
            The record, or None when nothing is published for that race.

        Raises:
            DataError: If the file exists but is not a valid record.
        """
        return self._files.load(event)

    def publish(self, record: Mapping[str, Any]) -> PublishResult:
        """Write one record, refresh the index, and commit both in one commit.

        Args:
            record: A record valid against the shared schema.

        Returns:
            Where the record lives and whether anything changed on disk or in git.

        Raises:
            DataError: If git fails to stage or commit the files.
        """
        result = self._files.publish(record)
        committed = self._commit and self._commit_files(record)
        return PublishResult(result.location, changed=result.changed or committed)

    def _commit_files(self, record: Mapping[str, Any]) -> bool:
        path = record_path(self._root, int(record["season"]), int(record["round"]))
        paths = [_relative(p, self._repo) for p in (path, self._root / INDEX_NAME)]
        try:
            _git(self._repo, "add", "--", *paths)
            if _nothing_staged(self._repo, paths):
                return False
            _git(self._repo, "commit", "--quiet", "-m", commit_message(record), "--", *paths)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise DataError(f"could not commit the published prediction: {detail}") from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise DataError(f"could not commit the published prediction: {exc}") from exc
        return True


def commit_message(record: Mapping[str, Any]) -> str:
    """The Conventional Commit message for a published or scored record.

    Args:
        record: The record being committed.

    Returns:
        A one-line commit message naming the race and the feature ID.
    """
    season, round_ = record["season"], record["round"]
    if record.get("score") is None:
        return f"chore(predictions): publish {season} round {round_} forecast (P8-06)"
    return f"chore(predictions): score {season} round {round_} (P8-07)"


def _relative(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(path)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        timeout=_GIT_TIMEOUT_S,
    )


def _nothing_staged(repo: Path, paths: list[str]) -> bool:
    done = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", *paths],
        cwd=repo,
        capture_output=True,
        check=False,
        timeout=_GIT_TIMEOUT_S,
    )
    return done.returncode == _NOTHING_STAGED
