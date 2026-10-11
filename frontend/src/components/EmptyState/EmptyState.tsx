import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import styles from "./EmptyState.module.css";

interface Props {
  icon: LucideIcon;
  title: string;
  children: ReactNode;
}

export function EmptyState({ icon: Icon, title, children }: Props) {
  return (
    <div className={styles.empty}>
      <Icon size={44} strokeWidth={1.5} aria-hidden className={styles.icon} />
      <h2 className={styles.title}>{title}</h2>
      <p className={styles.body}>{children}</p>
    </div>
  );
}
