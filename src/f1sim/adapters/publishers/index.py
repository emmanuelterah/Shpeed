"""Atomic file writes and the predictions index the static site reads (P8-03, P8-10).

Every file is written to a temporary sibling, flushed to disk and renamed into place, so a
reader never sees a half-written record (ACID atomicity and durability). The index is
rebuilt from the records on disk after every publish and validated against its own schema,
so the scoreboard and calibration can never disagree with the records they summarise.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, cast

from f1sim.adapters.records import validate_index
from f1sim.core.domain.record import PredictionRecord
from f1sim.core.errors import DataError, InvariantError
from f1sim.core.usecases.scoreboard import build_index

LOGGER = logging.getLogger(__name__)

INDEX_NAME = "index.json"

_RECORD_GLOB = "[0-9][0-9][0-9][0-9]/[0-9]*.json"


def record_path(root: Path, season: int, round_: int) -> Path:
    """Path of one race's record.

    Args:
        root: The predictions folder.
        season: Championship year.
        round_: Round number within the season.

    Returns:
        ``<root>/<season>/<round>.json``, with the round zero-padded.
    """
    return root / str(season) / f"{round_:02d}.json"


def to_json(document: Any) -> str:
    """Serialise a document the way every published file is written.

    Args:
        document: A JSON-ready value.

    Returns:
        Two-space-indented JSON with a trailing newline, so files diff cleanly in git.
    """
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    """Read one published JSON object.

    Args:
        path: The file.

    Returns:
        The parsed object.

    Raises:
        DataError: If the file is unreadable or not a JSON object.
    """
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise DataError(f"{path} is not a JSON object")
    return cast(dict[str, Any], data)


def write_if_changed(path: Path, text: str) -> bool:
    """Write a file atomically unless it already holds exactly this text.

    Args:
        path: Destination file; its parent is created if missing.
        text: Content to write.

    Returns:
        True when the file was written, False when it was already up to date.
    """
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    write_atomic(path, text)
    return True


def write_atomic(path: Path, text: str) -> None:
    """Write text to a file atomically: temp file, fsync, then rename.

    Args:
        path: Destination file; its parent is created if missing.
        text: Content to write.

    Raises:
        BaseException: Whatever interrupted the write, re-raised after the temporary file is
            removed, so no partial file is left behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def published_records(root: Path) -> list[tuple[str, PredictionRecord]]:
    """Read every record under the predictions folder.

    A file that cannot be read as a record is logged and left out, so one damaged file
    cannot take the whole site down.

    Args:
        root: The predictions folder.

    Returns:
        ``(path relative to root, record)`` pairs.
    """
    found: list[tuple[str, PredictionRecord]] = []
    for path in sorted(root.glob(_RECORD_GLOB)):
        try:
            record = PredictionRecord.from_mapping(read_json(path))
        except (DataError, InvariantError) as exc:
            LOGGER.warning("skipping unreadable prediction record %s: %s", path, exc)
            continue
        found.append((path.relative_to(root).as_posix(), record))
    return found


def rebuild_index(root: Path) -> tuple[Path, bool]:
    """Rewrite the predictions index from the records on disk.

    Args:
        root: The predictions folder.

    Returns:
        The index file and whether it changed.

    Raises:
        ValidationError: If the index does not match its schema (a bug, never data).
    """
    index = build_index(published_records(root))
    validate_index(index)
    path = root / INDEX_NAME
    return path, write_if_changed(path, to_json(index))
