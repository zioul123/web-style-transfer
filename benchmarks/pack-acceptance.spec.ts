import { expect, test } from "@playwright/test";
import { gotoStableApp } from "../tests/helpers/appPage";

test("benchmark pack acceptance thresholds produce expected pass/fail", async ({
  page,
}) => {
  await gotoStableApp(page);
  const result = await page.evaluate(async () => {
    const { buildPackAcceptanceRows } =
      await import("/src/features/style-transfer/benchmark/packAcceptance.ts");

    const baseline = [
      {
        pack: "fp32" as const,
        elapsedMs: 100,
        avgStepMs: 25,
        finalLoss: 0.2,
        downloadSizeBytes: 1000,
      },
      {
        pack: "fp16" as const,
        elapsedMs: 80,
        avgStepMs: 20,
        finalLoss: 0.2005,
        downloadSizeBytes: 600,
      },
      {
        pack: "int8-per-channel" as const,
        elapsedMs: 75,
        avgStepMs: 18,
        finalLoss: 0.245,
        downloadSizeBytes: 400,
      },
      {
        pack: "int8log-per-channel" as const,
        elapsedMs: 76,
        avgStepMs: 18.5,
        finalLoss: 0.272,
        downloadSizeBytes: 390,
      },
    ];

    const failing = [
      {
        pack: "fp32" as const,
        elapsedMs: 100,
        avgStepMs: 25,
        finalLoss: 0.2,
        downloadSizeBytes: 1000,
      },
      {
        pack: "fp16" as const,
        elapsedMs: 80,
        avgStepMs: 20,
        finalLoss: 0.202,
        downloadSizeBytes: 600,
      },
      {
        pack: "int8-per-channel" as const,
        elapsedMs: 75,
        avgStepMs: 18,
        finalLoss: 0.27,
        downloadSizeBytes: 400,
      },
      {
        pack: "int8log-per-channel" as const,
        elapsedMs: 76,
        avgStepMs: 18.5,
        finalLoss: 0.28,
        downloadSizeBytes: 390,
      },
    ];

    return {
      passRows: buildPackAcceptanceRows(baseline),
      failRows: buildPackAcceptanceRows(failing),
      missingBaselineRows: buildPackAcceptanceRows(
        baseline.filter((row) => row.pack !== "fp32"),
      ),
    };
  });

  expect(result.passRows.find((row) => row.pack === "fp32")?.verdict).toBe(
    "pass",
  );
  expect(result.passRows.find((row) => row.pack === "fp16")?.verdict).toBe(
    "pass",
  );
  expect(
    result.passRows.find((row) => row.pack === "int8-per-channel")?.verdict,
  ).toBe("pass");
  expect(
    result.passRows.find((row) => row.pack === "int8log-per-channel")?.verdict,
  ).toBe("pass");

  expect(result.failRows.find((row) => row.pack === "fp16")?.verdict).toBe(
    "fail",
  );
  expect(
    result.failRows.find((row) => row.pack === "int8-per-channel")?.verdict,
  ).toBe("fail");
  expect(
    result.failRows.find((row) => row.pack === "int8log-per-channel")?.verdict,
  ).toBe("fail");

  expect(result.missingBaselineRows).toHaveLength(0);
});
