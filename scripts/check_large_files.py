"""Pre-commit hook that blocks files over 500 kB, since raw data is never committed (ADR-0001)."""

import sys
from pathlib import Path

LIMIT = 500_000


def main(paths: list[str]) -> int:
    """Report every file over the size limit.

    Args:
        paths: Files staged for commit.

    Returns:
        0 when every file is small enough, 1 otherwise.
    """
    too_big = [p for p in paths if Path(p).is_file() and Path(p).stat().st_size > LIMIT]
    for p in too_big:
        print(f"{p} is over {LIMIT // 1000} kB; raw data belongs in data/ (git-ignored).")
    return 1 if too_big else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
