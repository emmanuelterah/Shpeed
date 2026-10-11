import type { PredictionRecord } from "@/types/prediction";
import sample from "@/fixtures/sample-prediction.json";

export interface LoadedPrediction {
  record: PredictionRecord;
  isSample: boolean;
}

// With VITE_API_URL set (Docker: "/api"), data comes from the backend API.
// Without it (GitHub Pages), data comes from the static predictions/ files.
const apiUrl = import.meta.env.VITE_API_URL?.replace(/\/$/, "");
const base = import.meta.env.BASE_URL;

/**
 * Load the latest published prediction.
 * In development, falls back to clearly labelled sample data.
 */
export async function loadLatestPrediction(): Promise<LoadedPrediction | null> {
  try {
    const record = apiUrl
      ? await fetchJson<PredictionRecord>(`${apiUrl}/predictions/latest`)
      : await loadFromStaticFiles();
    return { record, isSample: false };
  } catch {
    if (import.meta.env.DEV) return { record: sample as PredictionRecord, isSample: true };
    return null;
  }
}

async function loadFromStaticFiles(): Promise<PredictionRecord> {
  const index = await fetchJson<{ latest: string }>(`${base}predictions/index.json`);
  return fetchJson<PredictionRecord>(`${base}predictions/${index.latest}`);
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return (await res.json()) as T;
}
