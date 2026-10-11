import { Flag } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/Badge/Badge";
import { EmptyState } from "@/components/EmptyState/EmptyState";
import { GroupedList, ListRow } from "@/components/GroupedList/GroupedList";
import { LargeTitle } from "@/components/LargeTitle/LargeTitle";
import { ProbabilityBar } from "@/components/ProbabilityBar/ProbabilityBar";
import { SegmentedControl } from "@/components/SegmentedControl/SegmentedControl";
import { loadLatestPrediction, type LoadedPrediction } from "@/lib/api";
import { formatPercent, sumFirst } from "@/lib/format";

type View = "win" | "podium" | "points";
const views = [
  { value: "win", label: "Win" },
  { value: "podium", label: "Podium" },
  { value: "points", label: "Points" },
] as const;
const topK: Record<View, number> = { win: 1, podium: 3, points: 10 };
const listHeader: Record<View, string> = {
  win: "Chance to win",
  podium: "Chance to finish on the podium",
  points: "Chance to score points",
};

export function RacePage() {
  const [state, setState] = useState<LoadedPrediction | null | undefined>(undefined);
  const [view, setView] = useState<View>("win");

  useEffect(() => {
    let live = true;
    loadLatestPrediction().then((p) => live && setState(p));
    return () => {
      live = false;
    };
  }, []);

  const rows = useMemo(() => {
    if (!state) return [];
    return state.record.drivers
      .map((d) => ({ ...d, p: sumFirst(d.p_position, topK[view]) }))
      .sort((a, b) => b.p - a.p);
  }, [state, view]);

  if (state === undefined) return <LargeTitle title="Race" subtitle="Loading…" />;
  if (state === null)
    return (
      <>
        <LargeTitle title="Race" />
        <EmptyState icon={Flag} title="No prediction yet">
          Predictions publish automatically after qualifying.
        </EmptyState>
      </>
    );

  const { record, isSample } = state;
  const subtitle = [
    record.season,
    `Round ${record.round}`,
    `${record.n_sims.toLocaleString()} simulations`,
  ].join(" · ");
  const cutoff = new Date(record.data_cutoff).toLocaleString();
  return (
    <>
      <LargeTitle
        title={record.event_name ?? `Round ${record.round}`}
        subtitle={subtitle}
        accessory={isSample ? <Badge tone="caution">Sample data</Badge> : undefined}
      />
      <div
        style={{
          maxWidth: "var(--content-max)",
          margin: "0 auto var(--space-4)",
          padding: "0 var(--space-4)",
        }}
      >
        <SegmentedControl
          label="Probability shown"
          options={views}
          value={view}
          onChange={setView}
        />
      </div>
      <GroupedList
        header={listHeader[view]}
        footer={`Data up to ${cutoff} · model ${record.model_version}`}
      >
        {rows.map((d, i) => (
          <ListRow
            key={d.driver_id}
            leading={i + 1}
            title={d.display_name ?? d.driver_id}
            subtitle={[d.team_id, d.grid ? `Starts P${d.grid}` : null].filter(Boolean).join(" · ")}
            trailing={formatPercent(d.p)}
          >
            <ProbabilityBar
              value={d.p}
              label={`${d.display_name ?? d.driver_id}, ${view} probability`}
            />
          </ListRow>
        ))}
      </GroupedList>
    </>
  );
}
