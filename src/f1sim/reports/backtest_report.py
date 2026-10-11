"""Backtest report (P2-12): leaderboard, calibration and per-race scores, as Markdown."""

from collections.abc import Sequence

from f1sim.core.usecases.backtest import BacktestResult

LABELS: dict[str, str] = {
    "rps": "RPS",
    "log_loss": "Log loss",
    "brier_win": "Brier win",
    "brier_podium": "Brier podium",
    "brier_top10": "Brier top 10",
    "spearman": "Spearman",
}
"""Column headings, in the order they appear; RPS is the headline metric."""


def render_backtest(results: Sequence[BacktestResult]) -> str:
    """Render one or more backtests of the same seasons as a Markdown report.

    Args:
        results: Backtest results, typically one per model.

    Returns:
        The report: a leaderboard sorted by RPS, then a section per model.
    """
    lines = ["# Backtest report", ""]
    if not results:
        return "\n".join([*lines, "No backtests were run.", ""])
    first = results[0]
    seasons = ", ".join(str(s) for s in first.seasons)
    lines += [
        f"Seasons {seasons}, snapshot {first.snapshot}. Walk-forward: each race is fitted "
        "only on data available at the end of qualifying (ADR-0005).",
        "",
        "Lower is better for every metric except Spearman.",
        "",
        *leaderboard(results),
    ]
    for result in results:
        lines += ["", *_model_section(result)]
    return "\n".join([*lines, ""])


def leaderboard(results: Sequence[BacktestResult]) -> list[str]:
    """Render the model leaderboard, best RPS first.

    Args:
        results: Backtest results, typically one per model.

    Returns:
        Markdown lines: a heading and a table with every summary metric.
    """
    ranked = sorted(results, key=lambda r: r.summary().get("rps", float("inf")))
    header = ["Model", "Races", *LABELS.values()]
    rows = [
        [r.model, str(len(r.races)), *(_fmt(r.summary().get(m)) for m in LABELS)] for r in ranked
    ]
    return ["## Leaderboard", "", *_table(header, rows)]


def _model_section(result: BacktestResult) -> list[str]:
    lines = [f"## {result.model}", ""]
    if result.skipped:
        lines += ["Skipped races:", ""]
        lines += [f"- {s.event.season} round {s.event.round}: {s.reason}" for s in result.skipped]
        lines.append("")
    for event, bins in result.calibration.items():
        lines += [f"### Calibration: {event}", ""]
        rows = [
            [f"{b.low:.1f}-{b.high:.1f}", _fmt(b.mean_forecast), _fmt(b.observed), str(b.count)]
            for b in bins
            if b.count
        ]
        lines += [*_table(["Forecast bin", "Mean forecast", "Observed", "Drivers"], rows), ""]
    lines += ["### Per race", ""]
    rows = [
        [
            str(r.season),
            str(r.round),
            str(r.n_drivers),
            *(_fmt(float(getattr(r, m))) for m in LABELS),
        ]
        for r in result.races
    ]
    return [*lines, *_table(["Season", "Round", "Drivers", *LABELS.values()], rows)]


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(row) + " |" for row in rows]


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"
