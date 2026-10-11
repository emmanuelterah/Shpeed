import { render, screen } from "@testing-library/react";
import { ProbabilityBar } from "./ProbabilityBar";

describe("ProbabilityBar", () => {
  it("exposes the value as an accessible meter", () => {
    render(<ProbabilityBar value={0.314} label="Win probability" />);
    const meter = screen.getByRole("meter", { name: "Win probability" });
    expect(meter).toHaveAttribute("aria-valuenow", "31");
    expect(meter).toHaveAttribute("aria-valuetext", "31%");
  });

  it("clamps out-of-range values", () => {
    render(<ProbabilityBar value={1.4} label="p" />);
    expect(screen.getByRole("meter")).toHaveAttribute("aria-valuenow", "100");
  });
});
