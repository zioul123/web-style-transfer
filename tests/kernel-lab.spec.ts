import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";

const gotoStableBenchmark = async (page: Page): Promise<void> => {
  await page.goto("/benchmark?kernelLabSmoke=1", { waitUntil: "load" });
  await page.locator("main").waitFor({ state: "visible" });
};

test("kernel lab smoke runs baseline and optimized storage rows", async ({ page }) => {
  test.setTimeout(300000);
  await gotoStableBenchmark(page);
  const hasFixtures = await page.evaluate(async () => {
    const fixture = await fetch("/vgg19-phase3-full-pass/vgg19_phase3_full_pass_fixture.json");
    const manifestResponse = await fetch("/vgg19-models/fp32/manifest.json");
    if (!fixture.ok || !manifestResponse.ok) return false;
    const manifest = (await manifestResponse.json()) as { shards?: Array<{ name: string }> };
    const firstShard = manifest.shards?.[0];
    if (firstShard === undefined) return false;
    const shard = await fetch(`/vgg19-models/fp32/${firstShard.name}`);
    return shard.ok;
  });
  test.skip(
    !hasFixtures,
    "Missing phase3/model fixtures. Run python-reference/export_vgg19_phase3_full_pass.py and provide fp32 model pack first.",
  );

  await page.getByRole("button", { name: "Kernel lab" }).click();
  await page.getByRole("button", { name: "Run benchmark" }).click();
  await expect(page.getByText("Status: Done")).toBeVisible({ timeout: 300000 });
  await expect(page.getByText("Kernel lab (cumulative variants, default pooled)")).toBeVisible();
  await expect(page.getByText("baseline (default pooled fp32)")).toBeVisible();
  await expect(page.getByText("cached+persistent+fp16-storage")).toBeVisible();
  const kernelTable = page.locator("table", {
    hasText: "baseline (default pooled fp32)",
  });
  await expect(kernelTable.locator("tbody tr")).toHaveCount(2);
  await expect(page.getByText("cached+persistent+fp16-storage:")).toHaveCount(0);
});
