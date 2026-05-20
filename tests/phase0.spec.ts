import { expect, test } from "@playwright/test";
import { gotoStableApp } from "./helpers/appPage";

test("phase 0 app boots and responds with worker status", async ({ page }) => {
  await gotoStableApp(page);

  await expect(
    page.getByRole("heading", { name: /WebGPU Style Transfer/i }),
  ).toBeVisible();
  await expect
    .poll(async () => await page.locator("main p").first().textContent(), {
      timeout: 15000,
    })
    .toMatch(/Worker/i);
  await expect(page.getByText(/WebGPU/i).first()).toBeVisible();
});
