// Smoke and accessibility checks on desktop Safari and iPhone viewports (P8-15, P8-16).
// Published records are served from e2e/fixtures/predictions, written by the real publisher.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

const fixtures = resolve(process.cwd(), "e2e/fixtures");

test.beforeEach(async ({ page }) => {
  await page.route("**/predictions/**", (route) => {
    const path = new URL(route.request().url()).pathname.replace(/^.*\/predictions\//, "");
    try {
      const body = readFileSync(resolve(fixtures, "predictions", path), "utf8");
      return route.fulfill({ body, contentType: "application/json" });
    } catch {
      return route.fulfill({ status: 404, body: "not found" });
    }
  });
});

const screens = [
  { path: "/#/race", heading: "Desert Grand Prix" },
  { path: "/#/race/2027/1", heading: "Opening Grand Prix" },
  { path: "/#/season", heading: "Season" },
  { path: "/#/calibration", heading: "Calibration" },
  { path: "/#/about", heading: "About" },
];

async function open(page: Page, path: string, heading: string) {
  await page.goto(path);
  await expect(page.getByRole("heading", { level: 1, name: heading })).toBeVisible();
}

test("app shell renders with primary navigation", async ({ page }) => {
  await open(page, "/", "Desert Grand Prix");
  await expect(page.getByRole("navigation", { name: "Primary" }).first()).toBeVisible();
});

test("the season scoreboard links to each race's forecast", async ({ page }) => {
  await open(page, "/#/season", "Season");
  await expect(page.getByText("2027 · 2 of 3 races scored")).toBeVisible();
  await page.getByRole("link", { name: /Harbour Grand Prix/ }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Harbour Grand Prix" })).toBeVisible();
  await expect(page.getByText(/scored: RPS/)).toBeVisible();
});

test("calibration has a chart and an equivalent table", async ({ page }) => {
  await open(page, "/#/calibration", "Calibration");
  await expect(page.getByRole("img", { name: /Win calibration/ })).toBeVisible();
  await expect(page.getByRole("table")).toBeVisible();
});

test("the views switch with the keyboard alone", async ({ page }) => {
  await open(page, "/#/race", "Desert Grand Prix");
  const win = page.getByRole("radio", { name: "Win" });
  await win.focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("radio", { name: "Podium" })).toBeChecked();
  await expect(page.getByRole("radio", { name: "Podium" })).toBeFocused();
  await expect(page.getByText("Chance to finish on the podium")).toBeVisible();
});

for (const scheme of ["light", "dark"] as const) {
  for (const { path, heading } of screens) {
    test(`${path} has no WCAG AA violations in ${scheme} mode`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme, reducedMotion: "reduce" });
      await open(page, path, heading);
      const results = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
        .analyze();
      expect(results.violations.map((v) => `${v.id}: ${v.help}`)).toEqual([]);
    });
  }
}
