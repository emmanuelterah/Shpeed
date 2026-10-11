import { Flag, Gauge, Info, Trophy } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import styles from "./AppShell.module.css";

const destinations = [
  { to: "/race", label: "Race", icon: Flag },
  { to: "/season", label: "Season", icon: Trophy },
  { to: "/calibration", label: "Calibration", icon: Gauge },
  { to: "/about", label: "About", icon: Info },
] as const;

/** Sidebar on wide screens, translucent tab bar on phones (HIG navigation patterns). */
export function AppShell() {
  return (
    <div className={styles.shell}>
      <nav className={styles.sidebar} aria-label="Primary">
        <div className={styles.brand}>Shpeed</div>
        {destinations.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} className={styles.sideItem}>
            <Icon size={18} strokeWidth={2} aria-hidden />
            {label}
          </NavLink>
        ))}
      </nav>
      <main className={styles.content}>
        <Outlet />
      </main>
      <nav className={styles.tabBar} aria-label="Primary">
        {destinations.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} className={styles.tabItem}>
            <Icon size={24} strokeWidth={1.75} aria-hidden />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
