/* Generated from schemas/prediction-record.v1.schema.json by pnpm gen:types. Do not edit. */

/**
 * One published race prediction with provenance. Immutable once the race starts.
 */
export interface PredictionRecord {
  schema_version: 1;
  season: number;
  round: number;
  event_name?: string;
  /**
   * Latest data the model was allowed to see (end of qualifying)
   */
  data_cutoff: string;
  /**
   * DuckLake snapshot the run read (ADR-0002)
   */
  snapshot_id: number;
  model_version: string;
  git_commit: string;
  n_sims: number;
  seed: number;
  p_rain?: number;
  drivers: {
    driver_id: string;
    display_name?: string;
    team_id?: string;
    grid?: number;
    /**
     * P(finish in position k) for k = 1..n; with p_dnf sums to 1
     */
    p_position: number[];
    p_dnf: number;
  }[];
  score?: {
    brier_win?: number;
    brier_podium?: number;
    log_loss?: number;
    rps?: number;
    scored_at?: string;
  } | null;
}
