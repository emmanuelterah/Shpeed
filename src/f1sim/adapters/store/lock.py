"""Single-writer lock (P1-12): a DuckDB-file catalog allows one writer at a time."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from f1sim.core.errors import StoreBusyError


@contextmanager
def writer_lock(root: Path, *, timeout_s: float = 0.0) -> Generator[None]:
    """Hold the store's writer lock for the duration of a ``with`` block.

    Args:
        root: Store directory; the lock file is ``<root>/.writer.lock``.
        timeout_s: How long to wait for the lock before giving up.

    Yields:
        Nothing; the lock is held while the block runs and released afterwards.

    Raises:
        StoreBusyError: If another process holds the lock past ``timeout_s``.
    """
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
