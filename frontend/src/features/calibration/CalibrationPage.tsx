import { Gauge } from "lucide-react";
import { useState } from "react";
import { Badge } from "@/components/Badge/Badge";
import { CalibrationChart } from "@/components/CalibrationChart/CalibrationChart";
import { EmptyState } from "@/components/EmptyState/EmptyState";
import { GroupedList } from "@/components/GroupedList/GroupedList";
import { LargeTitle } from "@/components/LargeTitle/LargeTitle";
import { SegmentedControl } from "@/components/SegmentedControl/SegmentedControl";
import { loadIndex } from "@/lib/api";
import { formatPercent } from "@/lib/format";
import { useLoad } from "@/lib/useLoad";
import type { CalibrationEvent } from "@/types/predictions-index";
import styles from "./CalibrationPage.module.css";

type Event = CalibrationEvent["event"];
const views = [
  { value: "win", label: "Win" },
  { value: "podium", label: "Podium" },
  { value: "top10", label: "Points" },
] as const;

export function CalibrationPage() {
  const loaded = useLoad(loadIndex);
  const [event, setEvent] = useState<Event>("win");

  if (loaded === undefined) return <LargeTitle title="Calibration" subtitle="Loading…" />;
  const calibration = loaded?.data.calibration;
  if (!loaded || !calibration || calibration.races_scored === 0)
    return (
      <>
        <LargeTitle title="Calibration" />
        <EmptyState icon={Gauge} title="Nothing scored yet">
          When the model says 30%, it should happen about 30% of the time. This chart appears after
          the first race is scored.
        </EmptyState>
      </>
    );

  const shown = calibration.events.find((e) => e.event === event);
  const bins = shown?.bins ?? [];
  const label = views.find((v) => v.value === event)?.label ?? event;
  const races = calibration.races_scored;
  return (
    <>
      <LargeTitle
        title="Calibration"
        subtitle={`${calibration.forecasts.toLocaleString()} driver forecasts · ${races} ${
          races === 1 ? "race" : "races"
        }`}
        accessory={loaded.isSample ? <Badge tone="caution">Sample data</Badge> : undefined}
      />
      <div className={styles.controls}>
        <SegmentedControl label="Event shown" options={views} value={event} onChange={setEvent} />
      </div>
      <GroupedList
        header={`Chance to ${label === "Points" ? "score points" : label.toLowerCase()}`}
      >
        <li className={styles.chartCell}>
          <CalibrationChart label={label} bins={bins} />
        </li>
      </GroupedList>
      <section className={styles.tableSection} aria-labelledby="calibration-table">
        <h2 id="calibration-table" className={styles.tableHeader}>
          The same numbers
        </h2>
        {/* Focusable so keyboard users can scroll the table sideways on narrow screens */}
        <div
          className={styles.tableCell}
          role="region"
          aria-labelledby="calibration-table"
          tabIndex={0}
        >
          <table className={styles.table}>
            <caption className={styles.srOnly}>
              {`${label} forecasts grouped by probability, with how often the event happened`}
            </caption>
            <thead>
              <tr>
                <th scope="col">Forecast range</th>
                <th scope="col">Average forecast</th>
                <th scope="col">Happened</th>
                <th scope="col">Forecasts</th>
              </tr>
            </thead>
            <tbody>
              {bins.map((b) => (
                <tr key={b.low}>
                  <th scope="row">{`${formatPercent(b.low)}–${formatPercent(b.high)}`}</th>
                  <td>{formatPercent(b.mean_forecast)}</td>
                  <td>{formatPercent(b.observed)}</td>
                  <td>{b.count.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
