import { Gauge } from "lucide-react";
import { EmptyState } from "@/components/EmptyState/EmptyState";
import { LargeTitle } from "@/components/LargeTitle/LargeTitle";

export function CalibrationPage() {
  return (
    <>
      <LargeTitle title="Calibration" />
      <EmptyState icon={Gauge} title="Nothing scored yet">
        When the model says 30%, it should happen about 30% of the time. This chart appears after
        the first scored races (roadmap P8-10).
      </EmptyState>
    </>
  );
}
