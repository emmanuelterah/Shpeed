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
  /**
   * Build commit; -dirty if uncommitted changes existed; 'unknown' outside a checkout
   */
  git_commit: string;
  n_sims: number;
  seed: number;
  p_rain?: number;
  /**
   * @minItems 1
   */
  drivers: [
    {
      driver_id: string;
      display_name?: string;
      team_id?: string;
      grid?: number;
      /**
       * P(finish in position k) for k = 1..n; with p_dnf sums to 1
       */
      p_position: number[];
      p_dnf: number;
    },
    ...{
      driver_id: string;
      display_name?: string;
      team_id?: string;
      grid?: number;
      /**
       * P(finish in position k) for k = 1..n; with p_dnf sums to 1
       */
      p_position: number[];
      p_dnf: number;
    }[]
  ];
  /**
   * Filled in by the scoring run after the race (P8-07); null until then
   */
  score?: {
    brier_win: number;
    brier_podium: number;
    brier_top10: number;
    log_loss: number;
    rps: number;
    scored_at: string;
    /**
     * Finishing position among the forecast drivers, in drivers order; null = DNF
     */
    positions: (number | null)[];
  } | null;
}
