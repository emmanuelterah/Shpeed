import { useId } from "react";
import type { CalibrationBin } from "@/types/predictions-index";
import { formatPercent } from "@/lib/format";
import styles from "./CalibrationChart.module.css";

interface Props {
  /** What the probabilities are of, such as "Win". */
  label: string;
  bins: readonly CalibrationBin[];
}

const SIZE = 320;
const PAD = { top: 12, right: 16, bottom: 40, left: 60 };
const PLOT = SIZE - PAD.left - PAD.right;
const TICKS = [0, 0.25, 0.5, 0.75, 1];

const x = (p: number) => PAD.left + p * PLOT;
const y = (p: number) => PAD.top + (1 - p) * PLOT;

/**
 * Reliability diagram: forecast probability against how often the event happened.
 * Points on the diagonal are perfectly calibrated; larger dots hold more forecasts.
 * The same numbers are always shown in a table, so the chart is never the only source.
 */
export function CalibrationChart({ label, bins }: Props) {
  const id = useId();
  const most = Math.max(1, ...bins.map((b) => b.count));
  const radius = (count: number) => 4 + 5 * Math.sqrt(count / most);
  const path = bins
    .map((b, i) => `${i ? "L" : "M"}${x(b.mean_forecast)},${y(b.observed)}`)
    .join(" ");
  return (
    <figure className={styles.figure}>
      <svg
        className={styles.chart}
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        role="img"
        aria-labelledby={`${id}-title`}
        aria-describedby={`${id}-desc`}
      >
        <title id={`${id}-title`}>{`${label} calibration`}</title>
        <desc id={`${id}-desc`}>
          {`Forecast probability against observed frequency in ${bins.length} bins. ` +
            "The table below lists every value."}
        </desc>
        <g className={styles.grid} aria-hidden>
          {TICKS.map((t) => (
            <g key={t}>
              <line x1={x(0)} x2={x(1)} y1={y(t)} y2={y(t)} />
              <line x1={x(t)} x2={x(t)} y1={y(0)} y2={y(1)} />
              <text className={styles.tick} x={x(0) - 8} y={y(t)} dy="0.32em" textAnchor="end">
                {formatPercent(t)}
              </text>
              <text className={styles.tick} x={x(t)} y={y(0) + 18} textAnchor="middle">
                {formatPercent(t)}
              </text>
            </g>
          ))}
        </g>
        <line className={styles.diagonal} x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} aria-hidden />
        <text className={styles.axis} x={x(0.5)} y={SIZE - 4} textAnchor="middle" aria-hidden>
          Forecast probability
        </text>
        <text
          className={styles.axis}
          transform={`translate(10 ${y(0.5)}) rotate(-90)`}
          textAnchor="middle"
          aria-hidden
        >
          Observed frequency
        </text>
        <path className={styles.line} d={path} aria-hidden />
        {bins.map((b) => (
          <circle
            key={b.low}
            className={styles.dot}
            cx={x(b.mean_forecast)}
            cy={y(b.observed)}
            r={radius(b.count)}
          >
            <title>
              {`Forecast ${formatPercent(b.mean_forecast)}, happened ` +
                `${formatPercent(b.observed)} of ${b.count} times`}
            </title>
          </circle>
        ))}
      </svg>
      <figcaption className={styles.caption}>
        Dots on the dashed line are perfectly calibrated. Above it, the model was too cautious;
        below it, too confident.
      </figcaption>
    </figure>
  );
}
