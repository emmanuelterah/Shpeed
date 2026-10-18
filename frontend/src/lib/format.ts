const percent = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 0 });
const percentFine = new Intl.NumberFormat(undefined, {
  style: "percent",
  maximumFractionDigits: 1,
});

/** Whole percent, with one decimal for small non-zero values so "0%" never hides a real chance. */
export function formatPercent(p: number): string {
  if (p > 0 && p < 0.01) return "<1%";
  if (p > 0.01 && p < 0.1) return percentFine.format(p);
  return percent.format(p);
}

export function sumFirst(values: readonly number[], k: number): number {
  return values.slice(0, k).reduce((a, b) => a + b, 0);
}

const score = new Intl.NumberFormat(undefined, {
  minimumFractionDigits: 3,
  maximumFractionDigits: 3,
});
const day = new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short", year: "numeric" });

/** A proper score such as RPS or Brier, to three decimals. */
export function formatScore(value: number): string {
  return score.format(value);
}

/** The calendar day of an ISO timestamp, in the reader's locale. */
export function formatDay(iso: string): string {
  return day.format(new Date(iso));
}

export function mean(values: readonly number[]): number | null {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
}
