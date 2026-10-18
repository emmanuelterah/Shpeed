# How fast does the pace model recover after a regulation reset? (P7-05)

The project's headline question has a short answer: **the model does not need to forget.** On
the 2022 ground-effect reset it is back to the level the old era finished on within about a
third of the season, and the simplest strategy — keeping the full history — gets there as fast
as, or faster than, deliberately discarding or fading it. A hard reset to wide priors costs
accuracy at the reset race and never clearly buys it back.

Numbers: `docs/research/regulation-reset.md` (produced by `f1sim report recovery`). Runnable
walkthrough: `notebooks/01-regulation-reset.ipynb`.

## Method

The reset is a natural experiment (P7-01). The pace model (ADR-0006) is walked forward across
the seasons around it, one race at a time, and every race is scored with the model that saw
only the data before it. For each way of handling the history (P7-02):

- **Full history** — keep everything; the Phase 3 baseline.
- **Exponential decay** — fade the kept information a little every weekend (half-life 10 races).
- **Reset priors** — at the reset season, set the car effects back to zero mean and a wide
  prior (spread 1.5% of lap time); driver effects carry over.
- **Testing-informed priors** — a reset guided by the reset season's opening practice, which
  stands in for the pre-season testing the store cannot see.

A **recovery curve** is the ranked probability score (RPS, lower is better) of each race after
the reset, read against the **pre-reset level**: the mean RPS of the season before it. The
reference here is 2021's 0.1251; race 1 is the 2022 Bahrain Grand Prix.

## Results (2021 → 2022, snapshot 10)

| Strategy | Pre-reset | Race 1 | Race 2 | Race 3 | Race 5 | Race 10 | Back to level |
|---|---|---|---|---|---|---|---|
| Full history | 0.1251 | 0.1474 | 0.1420 | 0.1778 | 0.1439 | 0.1764 | race 7 |
| Exponential decay | 0.1253 | 0.1488 | 0.1425 | 0.1792 | 0.1447 | 0.1770 | race 7 |
| Reset priors | 0.1251 | 0.1712 | 0.1481 | 0.1743 | 0.1491 | 0.1804 | race 7 |
| Testing-informed priors | 0.1251 | 0.2032 | 0.1515 | 0.1731 | 0.1513 | 0.1837 | race 7 |

What the table says:

1. **Everything recovers by race 7** (the 2022 Monaco Grand Prix, late May) — about a third of
   a 22-race season. The reset race itself (Bahrain) is the hardest for every strategy.
2. **Keeping the history is not a handicap.** Full history and exponential decay are within
   0.002 of each other at every race, and differ by under 0.0015 for most of them; they are
   effectively the same strategy on this data, and where they differ it is never systematic.
3. **A hard reset hurts exactly where you would expect.** At the reset race it is 0.171 (reset
   priors) and 0.203 (testing-informed) against 0.147 for full history — the model has thrown
   away what it knew about the cars and has nothing trustworthy to replace it with yet.
4. **Resetting changes *where* the model is bad, not whether it recovers.** Once the new cars
   have raced a few weekends, reset priors scores *better* than full history at 9 of the 22
   races (3, 4, 8, 9, 11, 14, 16, 17 and 21): dropping the stale prior helps it move onto the
   new era faster at some weekends, at the cost of being clearly worse early.

## The honest caveats

- The whole study is **one season of 22 races**, scored by a deterministic model, on one
  snapshot. The numbers reproduce exactly, but the race-to-race spread in RPS (0.09 to 0.19)
  is an order of magnitude wider than the gaps between strategies (0.002 to 0.02). These are
  **directional findings, not a ranking with error bars.**
- 2014 and 2026 are declared in the manifest but not runnable: 2014 has no lap timing (results
  level only) and 2026 has no data. `f1sim report recovery` says so rather than failing, so the
  study extends to them the moment their snapshots exist.
- The synthetic world in the notebook agrees with the direction — full history is again the
  fastest to return to level, and every strategy is back within two races — but its reset is
  cleaner than a real one, so it exercises the code rather than settling the question.

## What this means for the model

Do not add a reset step to the production path. The default stays **full history**: on the
evidence available it is the simplest strategy and loses nothing. If a future regulation makes
history look genuinely misleading, **exponential decay** is the safer lever than a hard reset —
it costs a hair of early accuracy and never throws information away — and both remain config
options (`ModelOptions.reset`) rather than engine behaviour (ADR-0010).
