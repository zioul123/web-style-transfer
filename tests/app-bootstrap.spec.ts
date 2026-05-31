import { expect, test } from "./helpers/coverage";
import { gotoStableApp } from "./helpers/appPage";

test("app boots and responds with worker status", async ({ page }) => {
  await gotoStableApp(page);

  await expect(
    page.getByRole("heading", { name: /WebGPU Style Transfer/i }),
  ).toBeVisible();
  await expect
    .poll(async () => await page.locator("main").textContent(), {
      timeout: 15000,
    })
    .toMatch(
      /Worker|Loaded VGG19 weights pack|Failed to load VGG19 weights pack/i,
    );
  await expect(page.getByText(/WebGPU/i).first()).toBeVisible();
});
