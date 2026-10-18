import { GroupedList, ListRow } from "@/components/GroupedList/GroupedList";
import { LargeTitle } from "@/components/LargeTitle/LargeTitle";

const method = [
  {
    title: "Lap-by-lap simulation",
    subtitle:
      "Each race is run 10,000 times with sampled pace, tyre wear, strategy, " +
      "safety cars and retirements.",
  },
  {
    title: "Cutoff after qualifying",
    subtitle: "Forecasts use only data from before the race and are locked at lights out.",
  },
  {
    title: "Scored in public",
    subtitle: "Every forecast is scored against the official result with proper scoring rules.",
  },
] as const;

export function AboutPage() {
  return (
    <>
      <LargeTitle title="About" subtitle="How Shpeed makes its forecasts" />
      <GroupedList header="Method">
        {method.map((item) => (
          <ListRow key={item.title} title={item.title} subtitle={item.subtitle} />
        ))}
      </GroupedList>
      <GroupedList footer="Shpeed is an unofficial project and is not affiliated with Formula 1.">
        <ListRow title="Source code" trailing="MIT" />
      </GroupedList>
    </>
  );
}
