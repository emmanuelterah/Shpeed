"""Every module imports cleanly (catches circular imports and syntax errors in stubs)."""

import importlib
import pkgutil

import pytest

import f1sim


def _modules() -> list[str]:
    skip = ("f1sim.api", "f1sim.__main__")
    return [
        m.name
        for m in pkgutil.walk_packages(f1sim.__path__, "f1sim.")
        if not m.name.startswith(skip)
    ]


@pytest.mark.parametrize("name", _modules())
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
