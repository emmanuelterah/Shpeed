import { Trophy } from "lucide-react";
import { EmptyState } from "@/components/EmptyState/EmptyState";
import { LargeTitle } from "@/components/LargeTitle/LargeTitle";

export function SeasonPage() {
  return (
    <>
      <LargeTitle title="Season" />
      <EmptyState icon={Trophy} title="Nothing scored yet">
        The scoreboard fills in as each race is scored against the official result (roadmap P8-10).
      </EmptyState>
    </>
  );
}
