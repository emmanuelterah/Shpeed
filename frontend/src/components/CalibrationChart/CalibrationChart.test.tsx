import { render, screen } from "@testing-library/react";
import { CalibrationChart } from "./CalibrationChart";

const bins = [
  { low: 0, high: 0.1, mean_forecast: 0.04, observed: 0.03, count: 120 },
  { low: 0.3, high: 0.4, mean_forecast: 0.35, observed: 0.5, count: 8 },
];

describe("CalibrationChart", () => {
  it("is an image named after the event, pointing readers to the table", () => {
    render(<CalibrationChart label="Win" bins={bins} />);
    const chart = screen.getByRole("img", { name: /Win calibration/ });
    expect(chart).toHaveAccessibleDescription(/table below/);
  });

  it("draws one dot per bin with a hover description", () => {
    const { container } = render(<CalibrationChart label="Win" bins={bins} />);
    const dots = container.querySelectorAll("circle");
    expect(dots).toHaveLength(2);
    expect(dots[1]?.querySelector("title")?.textContent).toBe(
      "Forecast 35%, happened 50% of 8 times",
    );
  });
});
