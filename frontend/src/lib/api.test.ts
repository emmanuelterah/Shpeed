import { loadIndex, loadPrediction, recordPath } from "./api";
import { sampleIndex } from "@/test/sample";

function respond(routes: Record<string, unknown>) {
  return vi.fn(async (url: string) => {
    const body = routes[url];
    return body === undefined
      ? new Response("missing", { status: 404 })
      : new Response(JSON.stringify(body), { status: 200 });
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("api (static files, as on GitHub Pages)", () => {
  it("pads round numbers like the publisher", () => {
    expect(recordPath(2027, 3)).toBe("2027/03.json");
  });

  it("finds the latest record through the index", async () => {
    vi.stubGlobal(
      "fetch",
      respond({
        "/predictions/index.json": sampleIndex(),
        "/predictions/2027/03.json": { season: 2027, round: 3 },
      }),
    );
    const loaded = await loadPrediction();
    expect(loaded?.data.round).toBe(3);
    expect(loaded?.isSample).toBe(false);
  });

  it("loads a named race directly", async () => {
    vi.stubGlobal("fetch", respond({ "/predictions/2027/02.json": { season: 2027, round: 2 } }));
    expect((await loadPrediction({ season: 2027, round: 2 }))?.data.round).toBe(2);
  });

  it("loads the index", async () => {
    vi.stubGlobal("fetch", respond({ "/predictions/index.json": sampleIndex() }));
    expect((await loadIndex())?.data.races).toHaveLength(4);
  });

  it("falls back to labelled sample data in development", async () => {
    vi.stubGlobal("fetch", respond({}));
    expect((await loadIndex())?.isSample).toBe(true);
  });
});
