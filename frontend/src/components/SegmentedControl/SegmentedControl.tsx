import { useRef, type KeyboardEvent } from "react";
import styles from "./SegmentedControl.module.css";

interface Props<T extends string> {
  label: string;
  options: readonly { value: T; label: string }[];
  value: T;
  onChange: (value: T) => void;
}

const step: Record<string, number> = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };

/**
 * HIG segmented control: switches between views of the same data.
 * Follows the ARIA radio group pattern: one Tab stop, arrow keys move and select.
 */
export function SegmentedControl<T extends string>({ label, options, value, onChange }: Props<T>) {
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);
  const current = Math.max(
    0,
    options.findIndex((o) => o.value === value),
  );

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const delta = step[event.key];
    let next: number | undefined;
    if (delta !== undefined) next = (current + delta + options.length) % options.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = options.length - 1;
    const target = next === undefined ? undefined : options[next];
    if (next === undefined || !target) return;
    event.preventDefault();
    onChange(target.value);
    buttons.current[next]?.focus();
  }

  return (
    <div className={styles.control} role="radiogroup" aria-label={label} onKeyDown={onKeyDown}>
      {options.map((o, i) => (
        <button
          key={o.value}
          ref={(el) => {
            buttons.current[i] = el;
          }}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          tabIndex={i === current ? 0 : -1}
          className={styles.segment}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
