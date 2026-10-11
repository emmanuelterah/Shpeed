import { expect, test } from "@playwright/test";

test("app shell renders with primary navigation", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("navigation", { name: "Primary" }).first()).toBeVisible();
});
