import { Flag } from "lucide-react";
import { useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { Badge } from "@/components/Badge/Badge";
import { EmptyState } from "@/components/EmptyState/EmptyState";
import { GroupedList, ListRow } from "@/components/GroupedList/GroupedList";
import { LargeTitle } from "@/components/LargeTitle/LargeTitle";
import { ProbabilityBar } from "@/components/ProbabilityBar/ProbabilityBar";
import { SegmentedControl } from "@/components/SegmentedControl/SegmentedControl";
import { loadPrediction } from "@/lib/api";
import { formatPercent, formatScore, sumFirst } from "@/lib/format";
import { useLoad } from "@/lib/useLoad";
import type { PredictionRecord } from "@/types/prediction";

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

/** The latest forecast, or the race named in the URL (/race/:season/:round). */
export function RacePage() {
  const params = useParams();
  const race =
    params.season && params.round
      ? { season: Number(params.season), round: Number(params.round) }
      : undefined;
  const state = useLoad(() => loadPrediction(race), race ? `${race.season}/${race.round}` : "");
  const [view, setView] = useState<View>("win");

  const rows = useMemo(() => {
    if (!state) return [];
    return state.data.drivers
      .map((d) => ({ ...d, p: sumFirst(d.p_position, topK[view]) }))
      .sort((a, b) => b.p - a.p);
  }, [state, view]);

  if (state === undefined) return <LargeTitle title="Race" subtitle="Loading…" />;
  if (state === null)
    return (
      <>
        <LargeTitle title="Race" />
        <EmptyState icon={Flag} title={race ? "No prediction for this race" : "No prediction yet"}>
          Predictions publish automatically after qualifying.
        </EmptyState>
      </>
    );

  const { data: record, isSample } = state;
  const subtitle = [
    record.season,
    `Round ${record.round}`,
    `${record.n_sims.toLocaleString()} simulations`,
  ].join(" · ");

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
      <GroupedList header={listHeader[view]} footer={provenance(record)}>
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

/** Everything needed to reproduce the forecast, and its score once the race is run. */
function provenance(record: PredictionRecord): string {
  const parts = [
    `Data up to ${new Date(record.data_cutoff).toLocaleString()}`,
    `model ${record.model_version}`,
    `commit ${record.git_commit.slice(0, 7)}`,
    `snapshot ${record.snapshot_id}`,
    `seed ${record.seed}`,
  ];
  if (record.p_rain !== undefined) parts.push(`rain ${formatPercent(record.p_rain)}`);
  if (record.score) parts.push(`scored: RPS ${formatScore(record.score.rps)}`);
  return parts.join(" · ");
}
