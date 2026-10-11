"""Single-writer lock (P1-12). A DuckDB-file catalog allows one writer at a time."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from f1sim.core.errors import StoreBusyError


@contextmanager
def writer_lock(root: Path, *, timeout_s: float = 0.0) -> Generator[None]:
    """Hold the store's writer lock, or raise ``StoreBusyError`` if another process has it."""
    lock = FileLock(str(root / ".writer.lock"), timeout=timeout_s)
    try:
        lock.acquire()
    except Timeout as exc:
        raise StoreBusyError(
            f"Another sync is writing to {root}. Wait for it to finish, then try again."
        ) from exc
    try:
        yield
    finally:
        lock.release()
