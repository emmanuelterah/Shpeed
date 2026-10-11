import type { ReactNode } from "react";
import styles from "./Badge.module.css";

export function Badge({
  tone = "neutral",
  children,
}: {
  tone?: "neutral" | "caution";
  children: ReactNode;
}) {
  return <span className={`${styles.badge} ${styles[tone]}`}>{children}</span>;
}
