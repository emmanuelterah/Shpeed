"""Parsing for the CLI's free-form options (P2-04, P6-09).

Typer passes seasons, model lists and component lists as raw strings. These helpers turn them
into the tuples, sets and lists the commands expect, raising a ``BadParameter`` that Typer
turns into a clean usage error. Keeping them out of ``cli.py`` keeps that module short.
"""

import typer

from f1sim.adapters.models.registry import model_names
from f1sim.adapters.models.simulator import COMPONENTS


def parse_seasons(text: str) -> tuple[int, ...]:
    """Parse a season list such as ``2022-2025`` or ``2021,2023``.

    Args:
        text: Comma-separated seasons or inclusive ranges.

    Returns:
        The seasons in ascending order, without duplicates.

    Raises:
        typer.BadParameter: If the text is not a season list.
    """
    seasons: set[int] = set()
    try:
        for part in (p.strip() for p in text.split(",") if p.strip()):
            first, _, last = part.partition("-")
            seasons.update(range(int(first), int(last or first) + 1))
    except ValueError as exc:
        raise typer.BadParameter(f"{text!r} is not a season list like 2022-2025") from exc
    if not seasons:
        raise typer.BadParameter("give at least one season")
    return tuple(sorted(seasons))


def parse_model_list(text: str) -> list[str]:
    """Parse a model option into registry names.

    Args:
        text: A registered model name, a comma-separated list, or ``all``.

    Returns:
        The model names, in the order given.

    Raises:
        typer.BadParameter: If any name is not registered, or the list is empty.
    """
    names = model_names() if text == "all" else [m.strip() for m in text.split(",") if m.strip()]
    unknown = [n for n in names if n not in model_names()]
    if unknown or not names:
        choices = ", ".join(model_names())
        raise typer.BadParameter(f"unknown model {', '.join(unknown)}; choose from {choices}")
    return names


def parse_components(text: str) -> frozenset[str]:
    """Parse the ``--ablate`` list into the set of event components to switch off (P6-09).

    Args:
        text: Comma-separated component names, or empty for none.

    Returns:
        The components to disable.

    Raises:
        typer.BadParameter: If a name is not a known event component.
    """
    names = {name.strip() for name in text.split(",") if name.strip()}
    unknown = names - set(COMPONENTS)
    if unknown:
        choices = ", ".join(COMPONENTS)
        bad = ", ".join(sorted(unknown))
        raise typer.BadParameter(f"unknown component {bad}; choose from {choices}")
    return frozenset(names)
