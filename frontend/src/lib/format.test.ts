import { formatPercent, sumFirst } from "./format";

describe("formatPercent", () => {
  it("rounds normal values to whole percent", () => {
    expect(formatPercent(0.314)).toBe("31%");
  });
  it("never shows a real chance as 0%", () => {
    expect(formatPercent(0.004)).toBe("<1%");
  });
  it("shows zero as 0%", () => {
    expect(formatPercent(0)).toBe("0%");
  });
});

describe("sumFirst", () => {
  it("sums the first k positions", () => {
    expect(sumFirst([0.5, 0.25, 0.125], 2)).toBe(0.75);
  });
});
