"""Versioned fitted-state files (P3-10).

States live at ``<root>/<model>/<version>/<season>-<round>-s<snapshot>.json``. Each save
writes a temporary file in the same folder, flushes it to disk and renames it over the
target, so readers see either the old file or the new one, never half of one.
"""

import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from f1sim.core.domain.entities import EventKey, SnapshotId
from f1sim.core.errors import DataError
from f1sim.core.ports.state import StateKey

_NAME = re.compile(r"^(\d{4})-(\d{2})-s(\d+)\.json$")
_SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


class FileStateStore:
    """A ``FittedStateStore`` on the local disk."""

    def __init__(self, root: Path) -> None:
        """Use a folder for states; it is created on the first save.

        Args:
            root: Base folder, normally ``<data_dir>/state``.
        """
        self._root = root

    def path(self, key: StateKey) -> Path:
        """Where a state is saved.

        Args:
            key: The state.

        Returns:
            The file path.

        Raises:
            DataError: If the model or version contains characters unsafe in a path.
        """
        folder = self._folder(key.model, key.version)
        return folder / f"{key.event.season:04d}-{key.event.round:02d}-s{int(key.snapshot)}.json"

    def save(self, key: StateKey, state: Mapping[str, Any]) -> None:
        """Save a state atomically.

        Args:
            key: What the state is.
            state: JSON-compatible state.
        """
        target = self.path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temp = tempfile.mkstemp(dir=target.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                json.dump(dict(state), out)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temp, target)
        finally:
            # Only left behind if the write or rename failed.
            Path(temp).unlink(missing_ok=True)

    def load_latest(
        self, model: str, version: str, event: EventKey, max_snapshot: SnapshotId
    ) -> tuple[SnapshotId, dict[str, Any]] | None:
        """Load the newest state for one weekend from a snapshot no newer than given.

        Args:
            model: Model name.
            version: Model version.
            event: The weekend the state must end at.
            max_snapshot: Newest snapshot allowed.

        Returns:
            ``(snapshot, state)``, or None when nothing matches or the file is unreadable.
        """
        folder = self._folder(model, version)
        files: list[Path] = sorted(folder.glob("*.json")) if folder.is_dir() else []
        candidates: list[tuple[int, Path]] = []
        for file in files:
            match = _NAME.match(file.name)
            same_event = match and (int(match[1]), int(match[2])) == (event.season, event.round)
            if match and same_event and int(match[3]) <= int(max_snapshot):
                candidates.append((int(match[3]), file))
        if not candidates:
            return None
        snapshot, path = max(candidates)
        try:
            data: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        return SnapshotId(snapshot), cast(dict[str, Any], data)

    def _folder(self, model: str, version: str) -> Path:
        if not (_SAFE.match(model) and _SAFE.match(version)):
            raise DataError(f"unsafe model or version name: {model!r}, {version!r}")
        return self._root / model / version
