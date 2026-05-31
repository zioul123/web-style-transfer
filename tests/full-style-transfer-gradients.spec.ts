import { expect, test } from "./helpers/coverage";
import { gotoStableApp } from "./helpers/appPage";

test("run-style-transfer first-step gradient matches pytorch oracle", async ({
  page,
}) => {
  test.setTimeout(300000);
  await gotoStableApp(page);
  const result = await page.evaluate(async () => {
    const { loadFullPassArtifacts } =
      await import("/tests/helpers/fullPassArtifacts.ts");
    const artifactsResult = await loadFullPassArtifacts();
    if (!artifactsResult.ok) return artifactsResult;
    const {
      fixture,
      weights,
      firstStepGradientMeanTolerance,
      firstStepGradientMaxTolerance,
    } = artifactsResult.artifacts;
    if (fixture.expectedGradients?.total === undefined)
      return {
        ok: false as const,
        reason: "missing-gradient-fixture" as const,
      };
    const { askRunStyleTransfer, withBrowserStyleTransferWorkerClient } =
      await import("/tests/helpers/browserWorkerClient.ts");
    const learningRate = 1e-5;
    const runResult = await withBrowserStyleTransferWorkerClient(
      async (workerClient) => {
        await workerClient.initWebGpu("phase5-grad-init");
        return askRunStyleTransfer(workerClient, {
          type: "run-style-transfer",
          id: "phase5-grad-run",
          inputShape: fixture.inputShape,
          inputImageValues: fixture.inputImageValues,
          contentImageValues: fixture.contentImageValues,
          styleImageValues: fixture.styleImageValues,
          mean: fixture.mean,
          std: fixture.std,
          styleLayerIndices: fixture.styleLayerIndices,
          contentLayerIndex: fixture.contentLayerIndex,
          weights,
          optimizer: "sgd",
          contentWeight: 1,
          styleWeight: 1,
          learningRate,
          steps: 1,
        });
      },
    );
    if (!runResult.ok) return runResult;
    const gradOracle = fixture.expectedGradients.total as number[];
    const gradObserved = runResult.response.finalValues.map(
      (value: number, index: number) =>
        (fixture.inputImageValues[index] - value) / learningRate,
    );
    let maxDiff = 0;
    let meanDiff = 0;
    for (let i = 0; i < gradObserved.length; i += 1) {
      const diff = Math.abs(gradObserved[i] - gradOracle[i]);
      maxDiff = Math.max(maxDiff, diff);
      meanDiff += diff;
    }
    meanDiff /= gradObserved.length;
    return {
      ok: true as const,
      maxDiff,
      meanDiff,
      meanTolerance: firstStepGradientMeanTolerance,
      maxTolerance: firstStepGradientMaxTolerance,
    };
  });
  test.skip(
    !result.ok && result.reason === "missing-fixtures",
    result.ok ? "" : result.message,
  );
  test.skip(
    !result.ok && result.reason === "missing-gradient-fixture",
    "Fixture is missing gradients. Re-run python-reference/export_vgg19_phase3_full_pass.py.",
  );
  if (
    !result.ok &&
    result.reason !== "missing-fixtures" &&
    result.reason !== "missing-gradient-fixture"
  )
    throw new Error(
      result.reason === "worker-failed" ? result.message : result.reason,
    );
  if (!result.ok) return;
  expect(result.maxDiff).toBeLessThan(result.maxTolerance);
  expect(result.meanDiff).toBeLessThan(result.meanTolerance);
});

test("weighted first-step gradient composes content and style gradients", async ({
  page,
}) => {
  test.setTimeout(300000);
  await gotoStableApp(page);
  const result = await page.evaluate(async () => {
    const { loadFullPassArtifacts } =
      await import("/tests/helpers/fullPassArtifacts.ts");
    const artifactsResult = await loadFullPassArtifacts();
    if (!artifactsResult.ok) return artifactsResult;
    const {
      fixture,
      weights,
      weightedGradientMeanTolerance,
      weightedGradientMaxTolerance,
    } = artifactsResult.artifacts;
    if (
      fixture.expectedGradients?.content === undefined ||
      fixture.expectedGradients.styleByLayer === undefined
    )
      return {
        ok: false as const,
        reason: "missing-gradient-fixture" as const,
      };
    const { askRunStyleTransfer, withBrowserStyleTransferWorkerClient } =
      await import("/tests/helpers/browserWorkerClient.ts");
    const learningRate = 1e-5;
    const contentWeight = 2.5;
    const styleWeight = 7;
    const runResult = await withBrowserStyleTransferWorkerClient(
      async (workerClient) => {
        await workerClient.initWebGpu("phase5-weighted-grad-init");
        return askRunStyleTransfer(workerClient, {
          type: "run-style-transfer",
          id: "phase5-weighted-grad-run",
          inputShape: fixture.inputShape,
          inputImageValues: fixture.inputImageValues,
          contentImageValues: fixture.contentImageValues,
          styleImageValues: fixture.styleImageValues,
          mean: fixture.mean,
          std: fixture.std,
          styleLayerIndices: fixture.styleLayerIndices,
          contentLayerIndex: fixture.contentLayerIndex,
          weights,
          optimizer: "sgd",
          contentWeight,
          styleWeight,
          learningRate,
          steps: 1,
        });
      },
    );
    if (!runResult.ok) return runResult;
    const contentGrad = fixture.expectedGradients.content;
    const styleByLayer = fixture.expectedGradients.styleByLayer;
    const gradObserved = runResult.response.finalValues.map(
      (value: number, index: number) =>
        (fixture.inputImageValues[index] - value) / learningRate,
    );
    let maxDiff = 0;
    let meanDiff = 0;
    for (let i = 0; i < gradObserved.length; i += 1) {
      let expected = contentGrad[i] * contentWeight;
      for (const layerIndex of fixture.styleLayerIndices) {
        expected += styleByLayer[`relu${layerIndex}`][i] * styleWeight;
      }
      const diff = Math.abs(gradObserved[i] - expected);
      maxDiff = Math.max(maxDiff, diff);
      meanDiff += diff;
    }
    meanDiff /= gradObserved.length;
    return {
      ok: true as const,
      maxDiff,
      meanDiff,
      meanTolerance: weightedGradientMeanTolerance,
      maxTolerance: weightedGradientMaxTolerance,
    };
  });
  test.skip(
    !result.ok && result.reason === "missing-fixtures",
    result.ok ? "" : result.message,
  );
  test.skip(
    !result.ok && result.reason === "missing-gradient-fixture",
    "Fixture is missing gradients. Re-run python-reference/export_vgg19_phase3_full_pass.py.",
  );
  if (
    !result.ok &&
    result.reason !== "missing-fixtures" &&
    result.reason !== "missing-gradient-fixture"
  )
    throw new Error(
      result.reason === "worker-failed" ? result.message : result.reason,
    );
  if (!result.ok) return;
  expect(result.maxDiff).toBeLessThan(result.maxTolerance);
  expect(result.meanDiff).toBeLessThan(result.meanTolerance);
});
