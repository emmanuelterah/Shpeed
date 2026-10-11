# Shpeed web

The public UI for Shpeed predictions: race forecasts, the season scoreboard and calibration. A static React + TypeScript app built with Vite and deployed to GitHub Pages (ADR-0004). Design rules are in [DESIGN.md](DESIGN.md).

```bash
pnpm install
pnpm dev          # local dev server; shows clearly-labelled sample data until real predictions exist
pnpm lint && pnpm typecheck && pnpm test
pnpm gen:types    # regenerate src/types/prediction.ts from ../schemas
pnpm test:e2e     # Playwright smoke tests on desktop Safari and iPhone viewports
```

## Structure

```
src/
  app/          routes and the app shell
  design/       tokens.css (colour, type, spacing, materials, motion) and base.css
  components/   design-system primitives, one folder each with its CSS module and test
  features/     one folder per screen: race, season, calibration, about
  lib/          data loading and formatting
  types/        generated from the shared JSON Schema
  fixtures/     sample data for development and tests (never shown as real)
  test/         test setup
e2e/            Playwright tests
```
