import { fireEvent, render, screen, within } from "@testing-library/react";
import { loadIndex } from "@/lib/api";
import { sampleIndex } from "@/test/sample";
import { CalibrationPage } from "./CalibrationPage";

vi.mock("@/lib/api", () => ({ loadIndex: vi.fn() }));
const mocked = vi.mocked(loadIndex);

describe("CalibrationPage", () => {
  it("pairs the chart with an equivalent table", async () => {
    mocked.mockResolvedValue({ data: sampleIndex(), isSample: false });
    render(<CalibrationPage />);
    expect(await screen.findByRole("img", { name: /Win calibration/ })).toBeInTheDocument();
    const rows = within(screen.getByRole("table")).getAllByRole("row");
    expect(rows).toHaveLength(2);
    expect(rows[1]).toHaveTextContent("55");
  });

  it("switches events with the segmented control", async () => {
    mocked.mockResolvedValue({ data: sampleIndex(), isSample: false });
    render(<CalibrationPage />);
    fireEvent.click(await screen.findByRole("radio", { name: "Podium" }));
    expect(screen.getByRole("img", { name: /Podium calibration/ })).toBeInTheDocument();
    expect(within(screen.getByRole("table")).getAllByRole("row")).toHaveLength(3);
  });

  it("shows an empty state until a race is scored", async () => {
    const empty = sampleIndex();
    empty.calibration = { ...empty.calibration, races_scored: 0 };
    mocked.mockResolvedValue({ data: empty, isSample: false });
    render(<CalibrationPage />);
    expect(await screen.findByText("Nothing scored yet")).toBeInTheDocument();
  });

  it("labels sample data", async () => {
    mocked.mockResolvedValue({ data: sampleIndex(), isSample: true });
    render(<CalibrationPage />);
    expect(await screen.findByText("Sample data")).toBeInTheDocument();
  });
});
