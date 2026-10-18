import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { loadIndex } from "@/lib/api";
import { sampleIndex } from "@/test/sample";
import { SeasonPage } from "./SeasonPage";

vi.mock("@/lib/api", () => ({ loadIndex: vi.fn() }));
const mocked = vi.mocked(loadIndex);

function renderPage() {
  return render(
    <MemoryRouter>
      <SeasonPage />
    </MemoryRouter>,
  );
}

describe("SeasonPage", () => {
  it("lists the newest season's races, newest first, linking to each forecast", async () => {
    mocked.mockResolvedValue({ data: sampleIndex(), isSample: false });
    renderPage();
    const races = await screen.findByRole("heading", { name: "Races" });
    const links = within(races.parentElement as HTMLElement).getAllByRole("link");
    expect(links.map((l) => l.getAttribute("href"))).toEqual([
      "/race/2027/3",
      "/race/2027/2",
      "/race/2027/1",
    ]);
    expect(screen.getByText("2027 · 2 of 3 races scored")).toBeInTheDocument();
    expect(screen.getByText(/awaiting result/)).toBeInTheDocument();
  });

  it("averages the scores of the scored races only", async () => {
    mocked.mockResolvedValue({ data: sampleIndex(), isSample: false });
    renderPage();
    const row = (await screen.findByText("Ranked probability score")).closest("li");
    expect(row).toHaveTextContent((0.15).toFixed(3));
  });

  it("offers a season picker when more than one season is published", async () => {
    mocked.mockResolvedValue({ data: sampleIndex(), isSample: false });
    renderPage();
    expect(await screen.findByRole("radiogroup", { name: "Season" })).toBeInTheDocument();
  });

  it("shows an empty state before anything is published", async () => {
    mocked.mockResolvedValue(null);
    renderPage();
    expect(await screen.findByText("No races yet")).toBeInTheDocument();
  });
});
