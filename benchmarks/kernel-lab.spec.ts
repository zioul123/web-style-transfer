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
    const { checkModelPackAvailability, fetchJsonOrNull } =
      await import("/tests/helpers/fixtures.ts");
    const fixture = await fetchJsonOrNull<unknown>(
      "/vgg19-phase3-full-pass/vgg19_phase3_full_pass_fixture.json",
    );
    if (fixture === null) return null;

    const fp32Availability = await checkModelPackAvailability(
      "/vgg19-models",
      "fp32",
    );
    if (fp32Availability.ok) return "fp32";

    const int8Availability = await checkModelPackAvailability(
      "/vgg19-models",
      "int8-per-channel",
    );
    if (int8Availability.ok) return "int8-per-channel";

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
