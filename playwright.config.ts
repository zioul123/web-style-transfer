import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  testIgnore: ["**/benchmarks/**"],
  globalTeardown: "./tests/helpers/coverageTeardown.ts",
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 4173",
    env: {
      PLAYWRIGHT_TEST: "1",
    },
    port: 4173,
    reuseExistingServer: true,
  },
  use: {
    baseURL: "http://127.0.0.1:4173",
    launchOptions: {
      args: [
        "--use-angle=swiftshader",
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan,UseSkiaRenderer",
      ],
    },
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
