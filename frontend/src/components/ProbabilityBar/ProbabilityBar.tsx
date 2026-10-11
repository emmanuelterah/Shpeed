import { formatPercent } from "@/lib/format";
import styles from "./ProbabilityBar.module.css";

interface Props {
  value: number;
  label: string;
}

/** A probability as a filled track. Exposed to assistive tech as a meter with a text value. */
export function ProbabilityBar({ value, label }: Props) {
  const clamped = Math.min(1, Math.max(0, value));
  return (
    <div
      className={styles.track}
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(clamped * 100)}
      aria-valuetext={formatPercent(clamped)}
    >
      <div className={styles.fill} style={{ width: `${clamped * 100}%` }} />
    </div>
  );
}
