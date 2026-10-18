import { Trophy } from "lucide-react";
import { useState } from "react";
import { Badge } from "@/components/Badge/Badge";
import { EmptyState } from "@/components/EmptyState/EmptyState";
import { GroupedList, ListRow } from "@/components/GroupedList/GroupedList";
import { LargeTitle } from "@/components/LargeTitle/LargeTitle";
import { SegmentedControl } from "@/components/SegmentedControl/SegmentedControl";
import { loadIndex } from "@/lib/api";
import { formatDay, formatScore, mean } from "@/lib/format";
import { useLoad } from "@/lib/useLoad";
import type { RaceSummary } from "@/types/predictions-index";

const MAX_SEASON_TABS = 4;

/** Season scoreboard: every published race and how its forecast scored (P8-10). */
export function SeasonPage() {
  const loaded = useLoad(loadIndex);
  const [picked, setPicked] = useState<string | undefined>(undefined);

  if (loaded === undefined) return <LargeTitle title="Season" subtitle="Loading…" />;
  const races = loaded?.data.races ?? [];
  if (!loaded || races.length === 0)
    return (
      <>
        <LargeTitle title="Season" />
        <EmptyState icon={Trophy} title="No races yet">
          The scoreboard fills in as each race is forecast after qualifying and scored against the
          official result.
        </EmptyState>
      </>
    );

  const seasons = [...new Set(races.map((r) => String(r.season)))].sort().reverse();
  const season = picked && seasons.includes(picked) ? picked : (seasons[0] ?? "");
  const shown = races.filter((r) => String(r.season) === season).reverse();
  const scored = shown.flatMap((r) => (r.score ? [r.score] : []));
  const average = (pick: (s: (typeof scored)[number]) => number) => {
    const value = mean(scored.map(pick));
    return value === null ? "—" : formatScore(value);
  };

  return (
    <>
      <LargeTitle
        title="Season"
        subtitle={`${season} · ${scored.length} of ${shown.length} races scored`}
        accessory={loaded.isSample ? <Badge tone="caution">Sample data</Badge> : undefined}
      />
      {seasons.length > 1 && seasons.length <= MAX_SEASON_TABS && (
        <div
          style={{
            maxWidth: "var(--content-max)",
            margin: "0 auto var(--space-4)",
            padding: "0 var(--space-4)",
          }}
        >
          <SegmentedControl
            label="Season"
            options={seasons.map((s) => ({ value: s, label: s }))}
            value={season}
            onChange={setPicked}
          />
        </div>
      )}
      <GroupedList
        header="Season average"
        footer="Lower is better for every score. RPS rewards near misses; Brier scores the win."
      >
        <ListRow title="Ranked probability score" trailing={average((s) => s.rps)} />
        <ListRow title="Log loss" trailing={average((s) => s.log_loss)} />
        <ListRow title="Brier score, win" trailing={average((s) => s.brier_win)} />
      </GroupedList>
      <GroupedList header="Races" footer="Forecasts lock at lights out and are scored after.">
        {shown.map((race) => (
          <ListRow
            key={race.path}
            leading={race.round}
            title={race.event_name ?? `Round ${race.round}`}
            subtitle={subtitle(race)}
            trailing={race.score ? `RPS ${formatScore(race.score.rps)}` : undefined}
            to={`/race/${race.season}/${race.round}`}
          />
        ))}
      </GroupedList>
    </>
  );
}

function subtitle(race: RaceSummary): string {
  const when = formatDay(race.data_cutoff);
  return race.score ? `${when} · scored` : `${when} · awaiting result`;
}
