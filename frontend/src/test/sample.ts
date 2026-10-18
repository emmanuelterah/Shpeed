import type { PredictionsIndex } from "@/types/predictions-index";

/** A small index: two scored races in 2027, one awaiting its result, one race in 2026. */
export function sampleIndex(overrides: Partial<PredictionsIndex> = {}): PredictionsIndex {
  const score = (rps: number) => ({
    brier_win: 0.04,
    brier_podium: 0.1,
    brier_top10: 0.15,
    log_loss: 2.5,
    rps,
    scored_at: "2027-03-15T18:00:00Z",
  });
  const race = (season: number, round: number, rps: number | null) => ({
    season,
    round,
    path: `${season}/${String(round).padStart(2, "0")}.json`,
    event_name: `Round ${round} Grand Prix`,
    data_cutoff: `${season}-03-13T16:00:00Z`,
    model_version: "simulator-0.9.0",
    n_drivers: 20,
    score: rps === null ? null : score(rps),
  });
  return {
    schema_version: 1,
    latest: "2027/03.json",
    races: [race(2026, 24, 0.1), race(2027, 1, 0.1), race(2027, 2, 0.2), race(2027, 3, null)],
    calibration: {
      races_scored: 3,
      forecasts: 60,
      events: [
        {
          event: "win",
          label: "Win",
          bins: [{ low: 0, high: 0.1, mean_forecast: 0.03, observed: 0.04, count: 55 }],
        },
        {
          event: "podium",
          label: "Podium",
          bins: [
            { low: 0, high: 0.1, mean_forecast: 0.05, observed: 0.06, count: 40 },
            { low: 0.5, high: 0.6, mean_forecast: 0.55, observed: 0.5, count: 20 },
          ],
        },
        { event: "top10", label: "Points", bins: [] },
      ],
    },
    ...overrides,
  };
}
