import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";

const gotoStableBenchmark = async (page: Page): Promise<void> => {
  await page.goto("/benchmark?kernelLabSmoke=1", { waitUntil: "load" });
  await page.locator("main").waitFor({ state: "visible" });
};

test("kernel lab smoke runs baseline and optimized storage rows", async ({
  page,
}) => {
  test.setTimeout(300000);
  await gotoStableBenchmark(page);
  const availablePack = await page.evaluate(async () => {
    const isRecord = (value: unknown): value is Record<string, unknown> =>
      typeof value === "object" && value !== null;
    const isShard = (value: unknown): value is { name: string } =>
      isRecord(value) && typeof value.name === "string";
    const isManifest = (
      value: unknown,
    ): value is { shards: Array<{ name: string }> } =>
      isRecord(value) &&
      Array.isArray(value.shards) &&
      value.shards.every(isShard);
    const loadJson = async (url: string): Promise<unknown | null> => {
      const response = await fetch(url);
      if (!response.ok) return null;
      const text = await response.text();
      try {
        return JSON.parse(text) as unknown;
      } catch {
        return null;
      }
    };
    const hasPack = async (pack: string): Promise<boolean> => {
      const manifest = await loadJson(`/vgg19-models/${pack}/manifest.json`);
      if (!isManifest(manifest)) return false;
      const firstShard = manifest.shards[0];
      if (firstShard === undefined) return false;
      const shard = await fetch(`/vgg19-models/${pack}/${firstShard.name}`);
      return shard.ok;
    };

    const fixture = await loadJson(
      "/vgg19-phase3-full-pass/vgg19_phase3_full_pass_fixture.json",
    );
    if (fixture === null) return null;
    if (await hasPack("fp32")) return "fp32";
    if (await hasPack("int8-per-channel")) return "int8-per-channel";
    return null;
  });
  test.skip(
    availablePack === null,
    "Missing phase3 fixture or usable model pack. Provide the phase3 fixture and fp32 or int8-per-channel model pack first.",
  );
  if (availablePack === null) return;

  const baselineName = `baseline (default pooled ${availablePack})`;
  await page.getByRole("button", { name: "Kernel lab" }).click();
  await page.getByRole("button", { name: "Run benchmark" }).click();
  await expect(page.getByText("Status: Done")).toBeVisible({ timeout: 300000 });
  await expect(
    page.getByText("Kernel lab (cumulative variants, default pooled)"),
  ).toBeVisible();
  await expect(page.getByText(baselineName)).toBeVisible();
  await expect(page.getByText("cached+persistent+fp16-storage")).toBeVisible();
  const kernelTable = page.locator("table", {
    hasText: baselineName,
  });
  await expect(kernelTable.locator("tbody tr")).toHaveCount(2);
  await expect(page.getByText("cached+persistent+fp16-storage:")).toHaveCount(
    0,
  );
});
