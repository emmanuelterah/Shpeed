/* Generated from schemas/predictions-index.v1.schema.json by pnpm gen:types. Do not edit. */

/**
 * predictions/index.json: published races, scoreboard and calibration (P8-10)
 */
export interface PredictionsIndex {
  schema_version: 1;
  /**
   * Path of the newest record, relative to predictions/
   */
  latest: string | null;
  /**
   * Oldest first
   */
  races: RaceSummary[];
  calibration: Calibration;
}
export interface RaceSummary {
  season: number;
  round: number;
  /**
   * Record path relative to predictions/, such as 2024/03.json
   */
  path: string;
  event_name?: string;
  data_cutoff: string;
  model_version: string;
  n_drivers: number;
  score: null | RaceScore;
}
export interface RaceScore {
  brier_win: number;
  brier_podium: number;
  brier_top10: number;
  log_loss: number;
  /**
   * Ranked probability score; lower is better
   */
  rps: number;
  scored_at: string;
}
export interface Calibration {
  races_scored: number;
  /**
   * Driver forecasts behind each event's bins
   */
  forecasts: number;
  events: CalibrationEvent[];
}
export interface CalibrationEvent {
  event: "win" | "podium" | "top10";
  label: string;
  /**
   * Non-empty probability bins, lowest first
   */
  bins: CalibrationBin[];
}
export interface CalibrationBin {
  low: number;
  high: number;
  mean_forecast: number;
  observed: number;
  count: number;
}
