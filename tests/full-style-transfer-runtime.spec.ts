import { expect, test } from "@playwright/test";
import { gotoStableApp } from "./helpers/appPage";

test("full style transfer endpoint returns losses", async ({ page }) => {
  test.setTimeout(300000);
  await gotoStableApp(page);
  const result = await page.evaluate(async () => {
    const { loadFullPassArtifacts } =
      await import("/tests/helpers/fullPassArtifacts.ts");
    const artifactsResult = await loadFullPassArtifacts();
    if (!artifactsResult.ok) return artifactsResult;
    const { fixture, weights } = artifactsResult.artifacts;
    const { askRunStyleTransfer, withBrowserStyleTransferWorkerClient } =
      await import("/tests/helpers/browserWorkerClient.ts");
    const runResult = await withBrowserStyleTransferWorkerClient(
      async (workerClient) => {
        await workerClient.initWebGpu("phase5-full-init");
        return askRunStyleTransfer(workerClient, {
          type: "run-style-transfer",
          id: "phase5-full-run",
          inputShape: fixture.inputShape,
          inputImageValues: fixture.inputImageValues,
          contentImageValues: fixture.contentImageValues,
          styleImageValues: fixture.styleImageValues,
          mean: fixture.mean,
          std: fixture.std,
          styleLayerIndices: fixture.styleLayerIndices,
          contentLayerIndex: fixture.contentLayerIndex,
          weights,
          contentWeight: 1,
          styleWeight: 1,
          learningRate: 0.01,
          steps: 5,
          optimizer: "sgd",
        });
      },
    );
    if (!runResult.ok) return runResult;
    return { ok: true as const, losses: runResult.response.losses };
  });
  test.skip(
    !result.ok && result.reason === "missing-fixtures",
    result.ok ? "" : result.message,
  );
  if (!result.ok && result.reason !== "missing-fixtures")
    throw new Error(
      result.reason === "worker-failed" ? result.message : result.reason,
    );
  if (!result.ok) return;
  expect(result.losses.length).toBe(5);
  expect(result.losses.every((loss) => Number.isFinite(loss))).toBeTruthy();
});

test("full style transfer supports rectangular style shape", async ({
  page,
}) => {
  test.setTimeout(300000);
  await gotoStableApp(page);
  const result = await page.evaluate(async () => {
    const { loadFullPassArtifacts } =
      await import("/tests/helpers/fullPassArtifacts.ts");
    const artifactsResult = await loadFullPassArtifacts();
    if (!artifactsResult.ok) return artifactsResult;
    const { weights } = artifactsResult.artifacts;
    const makeValues = (count: number, phase: number): number[] =>
      Array.from(
        { length: count },
        (_, index: number): number =>
          0.5 + 0.25 * Math.sin(index * 0.017 + phase),
      );
    const inputShape = [1, 3, 16, 16] as const;
    const styleShape = [1, 3, 16, 24] as const;
    const inputImageValues = makeValues(3 * 16 * 16, 0);
    const contentImageValues = Array.from(inputImageValues);
    const styleImageValues = makeValues(3 * 16 * 24, 0.7);
    const { askRunStyleTransfer, withBrowserStyleTransferWorkerClient } =
      await import("/tests/helpers/browserWorkerClient.ts");
    const runResult = await withBrowserStyleTransferWorkerClient(
      async (workerClient) => {
        await workerClient.initWebGpu("phase5-rect-style-init");
        return askRunStyleTransfer(workerClient, {
          type: "run-style-transfer",
          id: "phase5-rect-style-run",
          optimizer: "lbfgs",
          inputShape,
          contentShape: inputShape,
          styleShape,
          inputImageValues,
          contentImageValues,
          styleImageValues,
          mean: [0.485, 0.456, 0.406],
          std: [0.229, 0.224, 0.225],
          styleLayerIndices: [1, 6, 11, 20, 29],
          contentLayerIndex: 22,
          weights,
          contentWeight: 1,
          styleWeight: 1,
          learningRate: 1e-5,
          steps: 1,
        });
      },
    );
    if (!runResult.ok) return runResult;
    return {
      ok: true as const,
      losses: runResult.response.losses,
      finalValuesLength: runResult.response.finalValues.length,
      expectedLength: inputImageValues.length,
    };
  });
  test.skip(
    !result.ok && result.reason === "missing-fixtures",
    result.ok ? "" : result.message,
  );
  if (!result.ok && result.reason !== "missing-fixtures")
    throw new Error(
      result.reason === "worker-failed" ? result.message : result.reason,
    );
  if (!result.ok) return;
  expect(result.losses.length).toBe(1);
  expect(result.losses.every((loss) => Number.isFinite(loss))).toBeTruthy();
  expect(result.finalValuesLength).toBe(result.expectedLength);
});

test("style transfer supports adam and lbfgs", async ({ page }) => {
  test.setTimeout(300000);
  await gotoStableApp(page);
  const result = await page.evaluate(async () => {
    const { loadFullPassArtifacts } =
      await import("/tests/helpers/fullPassArtifacts.ts");
    const artifactsResult = await loadFullPassArtifacts();
    if (!artifactsResult.ok) return artifactsResult;
    const { fixture, weights } = artifactsResult.artifacts;
    const { askRunStyleTransfer, withBrowserStyleTransferWorkerClient } =
      await import("/tests/helpers/browserWorkerClient.ts");
    const common = {
      type: "run-style-transfer" as const,
      inputShape: fixture.inputShape,
      inputImageValues: fixture.inputImageValues,
      contentImageValues: fixture.contentImageValues,
      styleImageValues: fixture.styleImageValues,
      mean: fixture.mean,
      std: fixture.std,
      styleLayerIndices: fixture.styleLayerIndices,
      contentLayerIndex: fixture.contentLayerIndex,
      weights,
      contentWeight: 1,
      styleWeight: 1,
      learningRate: 1e-5,
    };
    const cases: readonly {
      optimizer: "adam" | "lbfgs";
      steps: 1 | 3;
      id: string;
    }[] = [
      { optimizer: "adam", steps: 1, id: "adam-1" },
      { optimizer: "adam", steps: 3, id: "adam-3" },
      { optimizer: "lbfgs", steps: 3, id: "lbfgs-3" },
    ];
    const results: {
      id: string;
      finalLoss: number;
      finalValuesLength: number;
    }[] = [];
    const runFailure = await withBrowserStyleTransferWorkerClient(
      async (workerClient) => {
        await workerClient.initWebGpu("phase5-optimizers-init");
        for (const testCase of cases) {
          const run = await askRunStyleTransfer(workerClient, {
            ...common,
            id: `phase5-optimizer-${testCase.id}`,
            optimizer: testCase.optimizer,
            steps: testCase.steps,
          });
          if (!run.ok) return run;
          results.push({
            id: testCase.id,
            finalLoss:
              run.response.losses[run.response.losses.length - 1] ?? Number.NaN,
            finalValuesLength: run.response.finalValues.length,
          });
        }
        return null;
      },
    );
    if (runFailure !== null) return runFailure;
    return {
      ok: true as const,
      results,
      expectedLength: fixture.inputImageValues.length,
    };
  });
  test.skip(
    !result.ok && result.reason === "missing-fixtures",
    result.ok ? "" : result.message,
  );
  if (!result.ok && result.reason !== "missing-fixtures") {
    throw new Error(
      `${result.reason}${"message" in result && result.message !== undefined ? `: ${result.message}` : ""}`,
    );
  }
  if (!result.ok) return;
  for (const optimizerResult of result.results) {
    expect(
      Number.isFinite(optimizerResult.finalLoss),
      optimizerResult.id,
    ).toBeTruthy();
    expect(optimizerResult.finalValuesLength, optimizerResult.id).toBe(
      result.expectedLength,
    );
  }
});
