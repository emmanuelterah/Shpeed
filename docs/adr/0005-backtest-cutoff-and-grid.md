# 0005. Backtest cutoff and starting grid

- Status: Accepted
- Date: 2026-10-11
- Roadmap: P2-04, P2-05

## Context

A walk-forward backtest is only honest if every forecast uses exactly what was known when it would have been made live. That needs two rules: when the forecast is made (the cutoff), and which grid the model is told about.

## Decision

- **Cutoff:** the end of the last pre-race session of the weekend, which is qualifying, or the sprint when it runs after qualifying. In practice it is the latest `available_at` of that weekend's `Q` and `S` results. Every row in the store carries `available_at`, so "known at the cutoff" is one comparison.
- **Leakage guard:** models only read through `CutoffReader`, which raises `LeakageError` for a read past the cutoff, and also if the store ever returns a row from after it. A leak fails the backtest loudly instead of inflating its score.
- **Grid:** the qualifying classification, numbered from 1, with unclassified drivers at the back. In 2021 and 2022 the sprint result set the grid, so those weekends use the sprint order. A weekend with no qualifying uses the sprint.
- **Entries:** the drivers in that grid session. Forecasts have one row per entry, in grid order, and are scored against the race classification. Drivers not classified count as DNF.
- **Weekends with no pre-race results** are skipped and listed in the report, not silently dropped.

## Alternatives considered

- **Cutoff at lights out, using the official starting grid.** More accurate (grid penalties and pit-lane starts are applied), but the official grid is published shortly before the race, and an exact publication time is not in our sources. Using it risks training on information published after a live forecast would have gone out.
- **Cutoff before qualifying.** Cleaner separation, but throws away the strongest single signal available to a real forecaster on Saturday night.

## Consequences

- Grid penalties applied after qualifying are not known to the model, so drivers with penalties look better than they are. The effect is the same for every model, so comparisons stay fair.
- The same rule will apply to live predictions (P2-02), so backtest scores estimate live performance.
