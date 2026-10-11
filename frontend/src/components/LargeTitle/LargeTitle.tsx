import type { ReactNode } from "react";
import styles from "./LargeTitle.module.css";

interface Props {
  title: string;
  subtitle?: ReactNode;
  accessory?: ReactNode;
}

/** iOS-style large title header. */
export function LargeTitle({ title, subtitle, accessory }: Props) {
  return (
    <header className={styles.header}>
      <div className={styles.text}>
        <h1 className={styles.title}>{title}</h1>
        {subtitle && <p className={styles.subtitle}>{subtitle}</p>}
      </div>
      {accessory}
    </header>
  );
}
