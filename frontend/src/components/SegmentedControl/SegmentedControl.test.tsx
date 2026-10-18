import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { SegmentedControl } from "./SegmentedControl";

const options = [
  { value: "win", label: "Win" },
  { value: "podium", label: "Podium" },
  { value: "points", label: "Points" },
] as const;

function Harness() {
  const [value, setValue] = useState<(typeof options)[number]["value"]>("win");
  return <SegmentedControl label="View" options={options} value={value} onChange={setValue} />;
}

describe("SegmentedControl", () => {
  it("is a single tab stop on the selected segment", () => {
    render(<Harness />);
    const radios = screen.getAllByRole("radio");
    expect(radios.map((r) => r.tabIndex)).toEqual([0, -1, -1]);
    expect(screen.getByRole("radio", { name: "Win" })).toBeChecked();
  });

  it("moves and selects with the arrow keys, wrapping at the ends", () => {
    render(<Harness />);
    const group = screen.getByRole("radiogroup", { name: "View" });
    fireEvent.keyDown(group, { key: "ArrowRight" });
    expect(screen.getByRole("radio", { name: "Podium" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Podium" })).toHaveFocus();
    fireEvent.keyDown(group, { key: "ArrowLeft" });
    fireEvent.keyDown(group, { key: "ArrowLeft" });
    expect(screen.getByRole("radio", { name: "Points" })).toBeChecked();
    fireEvent.keyDown(group, { key: "Home" });
    expect(screen.getByRole("radio", { name: "Win" })).toBeChecked();
  });

  it("selects on click", () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole("radio", { name: "Points" }));
    expect(screen.getByRole("radio", { name: "Points" })).toBeChecked();
  });
});
