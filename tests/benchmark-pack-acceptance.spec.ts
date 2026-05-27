import { expect, test } from "@playwright/test";
import { gotoStableApp } from "./helpers/appPage";

test("benchmark pack acceptance thresholds produce expected pass/fail", async ({ page }) => {
  await gotoStableApp(page);
  const result = await page.evaluate(async () => {
    const { buildPackAcceptanceRows } = await import(
      "/src/features/style-transfer/benchmark/packAcceptance.ts"
    );

    const baseline = [
      { pack: "fp32" as const, elapsedMs: 100, avgStepMs: 25, finalLoss: 0.2000, downloadSizeBytes: 1000 },
      { pack: "fp16" as const, elapsedMs: 80, avgStepMs: 20, finalLoss: 0.2005, downloadSizeBytes: 600 },
      { pack: "int8-per-channel" as const, elapsedMs: 75, avgStepMs: 18, finalLoss: 0.2450, downloadSizeBytes: 400 },
    ];

    const failing = [
      { pack: "fp32" as const, elapsedMs: 100, avgStepMs: 25, finalLoss: 0.2000, downloadSizeBytes: 1000 },
      { pack: "fp16" as const, elapsedMs: 80, avgStepMs: 20, finalLoss: 0.2020, downloadSizeBytes: 600 },
      { pack: "int8-per-channel" as const, elapsedMs: 75, avgStepMs: 18, finalLoss: 0.2700, downloadSizeBytes: 400 },
    ];

    return {
      passRows: buildPackAcceptanceRows(baseline),
      failRows: buildPackAcceptanceRows(failing),
      missingBaselineRows: buildPackAcceptanceRows(baseline.filter((row) => row.pack !== "fp32")),
    };
  });

  expect(result.passRows.find((row) => row.pack === "fp32")?.verdict).toBe("pass");
  expect(result.passRows.find((row) => row.pack === "fp16")?.verdict).toBe("pass");
  expect(result.passRows.find((row) => row.pack === "int8-per-channel")?.verdict).toBe("pass");

  expect(result.failRows.find((row) => row.pack === "fp16")?.verdict).toBe("fail");
  expect(result.failRows.find((row) => row.pack === "int8-per-channel")?.verdict).toBe("fail");

  expect(result.missingBaselineRows).toHaveLength(0);
});
