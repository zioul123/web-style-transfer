import type { Page } from "@playwright/test";

const settleWindowMs = 300;
const maxLoadAttempts = 3;

export const gotoStableApp = async (page: Page): Promise<void> => {
  for (let attempt = 0; attempt < maxLoadAttempts; attempt += 1) {
    await page.goto("/", { waitUntil: "load" });
    await page.locator("main").waitFor({ state: "visible" });

    const navigated = await page
      .waitForEvent("framenavigated", { timeout: settleWindowMs })
      .then((): true => true)
      .catch((): false => false);

    if (!navigated) return;
  }

  await page.locator("main").waitFor({ state: "visible" });
};
