"""Fail if any line of code is longer than 100 characters.

Ruff and Prettier wrap most code, but neither shortens long strings, comments or lines that
end in pragma comments, and nothing else checks YAML, TOML, Dockerfiles or shell scripts.
This check covers every tracked code file in the repository with the same limit.

Usage:
    python scripts/check_line_length.py            # every tracked code file
    python scripts/check_line_length.py FILE ...   # only these files (pre-commit)
"""

import subprocess
import sys
from pathlib import Path

MAX_LENGTH = 100

CODE_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".ts",
        ".tsx",
        ".js",
        ".mjs",
        ".cjs",
        ".css",
        ".toml",
        ".yaml",
        ".yml",
        ".conf",
        ".sh",
        ".ini",
        ".cfg",
    }
)
CODE_NAMES = frozenset({"Dockerfile", ".editorconfig", ".gitignore", ".dockerignore"})

# Machine-written files that people never edit by hand
EXCLUDED = frozenset({"uv.lock", "frontend/pnpm-lock.yaml"})


def is_code(path: Path) -> bool:
    """Return whether ``path`` is a hand-written code file that this check covers.

    Args:
        path: File path relative to the repository root.

    Returns:
        True for code and configuration files, False for prose, data and lockfiles.
    """
    if path.as_posix() in EXCLUDED:
        return False
    return (
        path.suffix in CODE_SUFFIXES or path.name in CODE_NAMES or path.name.endswith(".Dockerfile")
    )


def tracked_files() -> list[Path]:
    """List every file git tracks or would track, in sorted order.

    Returns:
        Paths relative to the repository root.
    """
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(Path(line) for line in listing.stdout.splitlines() if line)


def long_lines(path: Path) -> list[tuple[int, int]]:
    """Find lines in ``path`` longer than ``MAX_LENGTH`` characters.

    Args:
        path: File to scan, read as UTF-8.

    Returns:
        ``(line_number, length)`` for each offending line, numbered from 1.
    """
    text = path.read_text(encoding="utf-8")
    return [
        (number, len(line))
        for number, line in enumerate(text.splitlines(), start=1)
        if len(line) > MAX_LENGTH
    ]


def main(arguments: list[str]) -> int:
    """Report every over-long line and return a process exit code.

    Args:
        arguments: Files to check; empty means every tracked code file.

    Returns:
        0 when every line fits, 1 otherwise.
    """
    candidates = [Path(a) for a in arguments] if arguments else tracked_files()
    failures = 0
    for path in candidates:
        if not path.is_file() or not is_code(path):
            continue
        for number, length in long_lines(path):
            print(f"{path}:{number}: line is {length} characters (limit {MAX_LENGTH})")
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
