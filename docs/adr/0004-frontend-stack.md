# 0004. Frontend stack and design language

- Status: Accepted
- Date: 2026-10-11
- Roadmap: P0-18, P8-09 to P8-15

## Context

Predictions, the season scoreboard and calibration need a public UI. It is served from GitHub Pages, reads the committed prediction JSON, and should follow Apple's Human Interface Guidelines in look and behaviour.

## Decision

- **React 18 + TypeScript (strict) + Vite**, built as a static single-page app for GitHub Pages. No server rendering needed: all data is static JSON.
- **pnpm** for packages; **Vitest + Testing Library** for unit tests; **Playwright** for end-to-end smoke tests.
- **Types generated from `schemas/prediction-record.v1.schema.json`**, the same schema the backend validates against, so the two sides cannot drift.
- **Own design system** in `frontend/src/design/` built on HIG principles (see `frontend/DESIGN.md`), no component library, so the look stays native-feeling and light.

## Alternatives considered

- **Next.js / Astro:** strong static-site options; more framework than a JSON-reading dashboard needs.
- **SvelteKit:** smaller bundles; less familiar ecosystem for contributors.

## Consequences

- Apple's SF Pro font files and SF Symbols are licensed for Apple platforms only, so the web app uses the system font stack (which resolves to SF on Apple devices) and an open-source icon set.
- No team logos or official F1 marks in the UI (non-affiliation, P9-06).
