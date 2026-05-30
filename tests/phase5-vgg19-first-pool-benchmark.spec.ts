import { expect, test } from "@playwright/test";
import type { WorkerRequest } from "../src/types";
import { gotoStableApp } from "./helpers/appPage";
import { firstPoolFixtureMissingMessage } from "./helpers/fixtures";

// Use this for benchmarking.
test.skip("phase 5 benchmark: fused first-pool path is not slower than unfused by large margin", async ({
  page,
}) => {
  test.setTimeout(300000);
  await gotoStableApp(page);
  const result = await page.evaluate(async () => {
    const { loadVggFirstPoolArtifacts } =
      await import("/tests/helpers/fixtures.ts");
    const { createBrowserStyleTransferWorkerClient } =
      await import("/tests/helpers/browserWorkerClient.ts");
    const artifact = await loadVggFirstPoolArtifacts();
    if (!artifact.ok) {
      return { ok: false as const, reason: "missing-fixtures" as const };
    }
    const { weights, caseData } = artifact;
    const workerClient = createBrowserStyleTransferWorkerClient();

    try {
      await workerClient.initWebGpu("phase5-bench-init");
      const makePayload = (
        id: string,
        fusedConvRelu: boolean,
        fusedUpdate: boolean,
      ): WorkerRequest => ({
        type: "run-first-pool-optimizer",
        id,
        inputShape: caseData.inputShape,
        mean: caseData.mean,
        std: caseData.std,
        contentImageValues: caseData.inputValues,
        styleImageValues: [...caseData.inputValues].reverse(),
        initialInputValues: caseData.inputValues,
        conv1Weight: {
          shape: weights.conv1WeightShape,
          values: weights.conv1WeightValues,
        },
        conv1Bias: weights.conv1BiasValues,
        conv2Weight: {
          shape: weights.conv2WeightShape,
          values: weights.conv2WeightValues,
        },
        conv2Bias: weights.conv2BiasValues,
        conv3Weight: {
          shape: weights.conv3WeightShape,
          values: weights.conv3WeightValues,
        },
        conv3Bias: weights.conv3BiasValues,
        contentWeight: 1,
        styleWeightConv1: 30,
        styleWeightConv3: 120,
        learningRate: 0.15,
        steps: 6,
        useFusedConvRelu: fusedConvRelu,
        useFusedUpdateClamp: fusedUpdate,
        collectBenchmarkStats: true,
      });
      const baseline = await workerClient.ask(
        makePayload("phase5-bench-baseline", false, false),
      );
      const fused = await workerClient.ask(
        makePayload("phase5-bench-fused", true, true),
      );
      return { ok: true as const, baseline, fused };
    } finally {
      workerClient.dispose();
    }
  });
  test.skip(
    !result.ok && result.reason === "missing-fixtures",
    firstPoolFixtureMissingMessage,
  );
  if (!result.ok) throw new Error("benchmark setup failed");
  if (
    result.baseline.type !== "run-first-pool-optimizer-result" ||
    !result.baseline.ok
  ) {
    throw new Error("baseline failed");
  }
  if (
    result.fused.type !== "run-first-pool-optimizer-result" ||
    !result.fused.ok
  ) {
    throw new Error("fused failed");
  }
  expect(result.baseline.stats).toBeDefined();
  expect(result.fused.stats).toBeDefined();
  if (result.baseline.stats === undefined || result.fused.stats === undefined) {
    throw new Error("missing stats");
  }
  // Allow noise on SwiftShader; ensure no major regression.
  expect(result.fused.stats.elapsedMs).toBeLessThan(
    result.baseline.stats.elapsedMs * 1.5,
  );
  console.log("FIRST_POOL_BENCHMARK", {
    baseline: result.baseline.stats,
    fused: result.fused.stats,
  });
});
