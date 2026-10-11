"""Command-line interface and composition root.

This is the only module that knows concrete adapter classes: it reads settings,
builds the adapters, and injects them into the core use cases (dependency inversion).
"""

import typer

from f1sim.observability import configure_logging
from f1sim.settings import Settings

app = typer.Typer(help="Shpeed F1 race simulator", no_args_is_help=True)
data_app = typer.Typer(help="Fetch, validate and manage the local data store", no_args_is_help=True)
report_app = typer.Typer(help="Backtest and scoring reports", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(report_app, name="report")


@app.callback()
def main() -> None:
    """Configure logging for every command."""
    configure_logging(Settings().log_level)


@data_app.command("sync")
def data_sync(season: int = typer.Option(..., help="Season to sync, e.g. 2025")) -> None:
    """Fetch, validate and commit one season as a new snapshot (P1-17)."""
    raise NotImplementedError(f"P1-17: data sync for {season}")


@data_app.command("verify")
def data_verify() -> None:
    """Check every data file the catalog references (P1-14)."""
    raise NotImplementedError("P1-14")


@app.command()
def predict(season: int = typer.Option(...), round_: int = typer.Option(..., "--round")) -> None:
    """Predict one race and write its prediction record (P8-01)."""
    raise NotImplementedError(f"P8-01: predict {season} round {round_}")


@app.command()
def backtest(
    model: str = typer.Option("plackett_luce"),
    seasons: str = typer.Option("2022-2025"),
) -> None:
    """Walk-forward backtest of a registered model (P2-04)."""
    raise NotImplementedError(f"P2-04: backtest {model} over {seasons}")


@report_app.command("score")
def report_score(
    season: int = typer.Option(...), round_: int = typer.Option(..., "--round")
) -> None:
    """Score a published prediction against the official result (P8-07)."""
    raise NotImplementedError(f"P8-07: score {season} round {round_}")
