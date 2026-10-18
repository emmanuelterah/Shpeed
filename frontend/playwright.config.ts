import { defineConfig, devices } from "@playwright/test";

const ci = Boolean(process.env.CI);

export default defineConfig({
  testDir: "./e2e",
  forbidOnly: ci,
  retries: ci ? 1 : 0,
  reporter: ci ? [["list"], ["html", { open: "never" }]] : "list",
  use: { baseURL: "http://localhost:4173", trace: "retain-on-failure" },
  webServer: { command: "pnpm build && pnpm preview", port: 4173, reuseExistingServer: !ci },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Safari"] } },
    { name: "iphone", use: { ...devices["iPhone 15"] } },
  ],
});
