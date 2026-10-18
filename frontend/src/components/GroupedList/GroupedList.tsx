import { ChevronRight } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import styles from "./GroupedList.module.css";

interface ListProps {
  header?: string;
  footer?: ReactNode;
  children: ReactNode;
}

/** Inset grouped list, the standard container for rows of data. */
export function GroupedList({ header, footer, children }: ListProps) {
  return (
    <section className={styles.section}>
      {header && <h2 className={styles.header}>{header}</h2>}
      <ul className={styles.list}>{children}</ul>
      {footer && <p className={styles.footer}>{footer}</p>}
    </section>
  );
}

interface RowProps {
  leading?: ReactNode;
  title: ReactNode;
  subtitle?: ReactNode;
  trailing?: ReactNode;
  /** Makes the whole row a navigation link, shown with a disclosure chevron. */
  to?: string;
  children?: ReactNode;
}

export function ListRow({ leading, title, subtitle, trailing, to, children }: RowProps) {
  const content = (
    <>
      {leading && <span className={styles.leading}>{leading}</span>}
      <span className={styles.body}>
        <span className={styles.titleLine}>
          <span className={styles.title}>{title}</span>
          {trailing && <span className={styles.trailing}>{trailing}</span>}
        </span>
        {subtitle && <span className={styles.subtitle}>{subtitle}</span>}
        {children}
      </span>
    </>
  );
  if (!to) return <li className={styles.row}>{content}</li>;
  return (
    <li className={styles.linkItem}>
      <Link to={to} className={styles.row}>
        {content}
        <ChevronRight size={18} strokeWidth={2} aria-hidden className={styles.chevron} />
      </Link>
    </li>
  );
}
