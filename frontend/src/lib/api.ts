import type { PredictionRecord } from "@/types/prediction";
import type { PredictionsIndex } from "@/types/predictions-index";
import sample from "@/fixtures/sample-prediction.json";
import sampleIndex from "@/fixtures/sample-index.json";

export interface Loaded<T> {
  data: T;
  isSample: boolean;
}
export type LoadedPrediction = Loaded<PredictionRecord>;
export type LoadedIndex = Loaded<PredictionsIndex>;

// With VITE_API_URL set (Docker: "/api"), data comes from the backend API.
// Without it (GitHub Pages), data comes from the static predictions/ files.
const apiUrl = import.meta.env.VITE_API_URL?.replace(/\/$/, "");
const base = import.meta.env.BASE_URL;

/**
 * Load one race's prediction, or the latest when no race is given.
 * In development, falls back to clearly labelled sample data.
 */
export function loadPrediction(race?: {
  season: number;
  round: number;
}): Promise<LoadedPrediction | null> {
  return withSample(async () => {
    if (apiUrl) {
      const path = race ? `${race.season}/${race.round}` : "latest";
      return fetchJson<PredictionRecord>(`${apiUrl}/predictions/${path}`);
    }
    const path = race ? recordPath(race.season, race.round) : (await loadIndexFile()).latest;
    if (!path) throw new Error("nothing published yet");
    return fetchJson<PredictionRecord>(`${base}predictions/${path}`);
  }, sample as PredictionRecord);
}

/** Load the predictions index: every race, the season scoreboard and calibration. */
export function loadIndex(): Promise<LoadedIndex | null> {
  return withSample(
    () => (apiUrl ? fetchJson<PredictionsIndex>(`${apiUrl}/predictions/index`) : loadIndexFile()),
    sampleIndex as PredictionsIndex,
  );
}

/** Path of a record relative to predictions/, matching the publisher's layout. */
export function recordPath(season: number, round: number): string {
  return `${season}/${String(round).padStart(2, "0")}.json`;
}

async function withSample<T>(load: () => Promise<T>, fallback: T): Promise<Loaded<T> | null> {
  try {
    return { data: await load(), isSample: false };
  } catch {
    if (import.meta.env.DEV) return { data: fallback, isSample: true };
    return null;
  }
}

function loadIndexFile(): Promise<PredictionsIndex> {
  return fetchJson<PredictionsIndex>(`${base}predictions/index.json`);
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return (await res.json()) as T;
}
