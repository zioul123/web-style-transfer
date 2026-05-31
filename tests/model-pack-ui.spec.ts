import { expect, test } from "./helpers/coverage";
import { gotoStableApp } from "./helpers/appPage";

test("model pack selector reloads manifest-backed weights and supports fp32", async ({
  page,
}) => {
  await page.addInitScript(() => {
    const originalFetch = window.fetch.bind(window);
    const requests: string[] = [];
    (window as Window & { __packFetches?: string[] }).__packFetches = requests;
    window.fetch = async (...args: Parameters<typeof fetch>) => {
      const input = args[0];
      const url =
        typeof input === "string"
          ? input
          : input instanceof Request
            ? input.url
            : String(input);
      if (url.includes("/vgg19-models/")) requests.push(url);
      return originalFetch(...args);
    };
  });
  await gotoStableApp(page);

  await page.getByRole("button", { name: "Show options" }).click();
  const modelPackSelect = page.getByLabel("Model pack");
  await expect(modelPackSelect).toBeVisible();
  const initialPack = await modelPackSelect.inputValue();
  await expect
    .poll(async () => {
      const urls = await page.evaluate(() => {
        const state = window as Window & { __packFetches?: string[] };
        return state.__packFetches ?? [];
      });
      return urls.some((url) =>
        url.includes(`/vgg19-models/${initialPack}/manifest.json`),
      );
    })
    .toBeTruthy();

  await modelPackSelect.selectOption("fp32");
  await expect(modelPackSelect).toHaveValue("fp32");
  await expect
    .poll(async () => {
      const urls = await page.evaluate(() => {
        const state = window as Window & { __packFetches?: string[] };
        return state.__packFetches ?? [];
      });
      return urls.some((url) =>
        url.includes("/vgg19-models/fp32/manifest.json"),
      );
    })
    .toBeTruthy();
});
