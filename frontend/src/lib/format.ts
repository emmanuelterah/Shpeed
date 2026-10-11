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
