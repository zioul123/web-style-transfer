import { expect, test } from "./helpers/coverage";
import type { WorkerRequest, WorkerResponse } from "../src/types";
import { gotoStableApp } from "./helpers/appPage";
import { firstPoolFixtureMissingMessage } from "./helpers/fixtures";
import { expectWebGpuInitOk } from "./helpers/workerClient";

type FirstPoolOptimizerSuccessResult = {
  ok: true;
  init: WorkerResponse;
  losses: number[];
  finalValuesLength?: number;
  expectedLength?: number;
};

type FirstPoolOptimizerFailureResult =
  | { ok: false; reason: "missing-fixtures"; message: string }
  | { ok: false; reason: "unexpected-response"; responseType: string }
  | { ok: false; reason: "worker-failed"; message: string };

type FirstPoolOptimizerResult =
  | FirstPoolOptimizerSuccessResult
  | FirstPoolOptimizerFailureResult;

const throwIfUnexpectedFailure = (result: FirstPoolOptimizerResult): void => {
  if (result.ok || result.reason === "missing-fixtures") return;
  if (result.reason === "worker-failed") {
    throw new Error(`${result.reason}: ${result.message}`);
  }
  throw new Error(`${result.reason} (responseType=${result.responseType})`);
};

type FirstPoolOptimizerBrowserOptions = {
  requestId: string;
  steps: number;
  spatialSize: 16 | 32;
};

const runFirstPoolOptimizerInBrowser = async ({
  requestId,
  steps,
  spatialSize,
}: FirstPoolOptimizerBrowserOptions): Promise<FirstPoolOptimizerResult> => {
  const { loadVggFirstPoolArtifacts } =
    await import("/tests/helpers/fixtures.ts");
  const { createBrowserStyleTransferWorkerClient } =
    await import("/tests/helpers/browserWorkerClient.ts");

  const upscaleNchwNearest = (
    values: readonly number[],
    inputShape: [number, number, number, number],
    nextHeight: number,
    nextWidth: number,
  ): { shape: [number, number, number, number]; values: number[] } => {
    const [batch, channels, height, width] = inputShape;
    const scaled = new Array<number>(batch * channels * nextHeight * nextWidth);
    for (let b = 0; b < batch; b += 1) {
      for (let c = 0; c < channels; c += 1) {
        for (let y = 0; y < nextHeight; y += 1) {
          const srcY = Math.min(
            height - 1,
            Math.floor((y * height) / nextHeight),
          );
          for (let x = 0; x < nextWidth; x += 1) {
            const srcX = Math.min(
              width - 1,
              Math.floor((x * width) / nextWidth),
            );
            const srcIndex =
              ((b * channels + c) * height + srcY) * width + srcX;
            const dstIndex =
              ((b * channels + c) * nextHeight + y) * nextWidth + x;
            scaled[dstIndex] = values[srcIndex];
          }
        }
      }
    }
    return { shape: [batch, channels, nextHeight, nextWidth], values: scaled };
  };

  const artifact = await loadVggFirstPoolArtifacts();
  if (!artifact.ok) {
    return {
      ok: false,
      reason: "missing-fixtures",
      message: artifact.message,
    };
  }

  const { weights, caseData } = artifact;
  const upscaledInput =
    spatialSize === 16
      ? { shape: caseData.inputShape, values: caseData.inputValues }
      : upscaleNchwNearest(
          caseData.inputValues,
          caseData.inputShape,
          spatialSize,
          spatialSize,
        );
  const optimizerInput = {
    inputShape: upscaledInput.shape,
    values: upscaledInput.values,
  };
  const workerClient = createBrowserStyleTransferWorkerClient();

  try {
    const init = await workerClient.initWebGpu(`${requestId}-init`);
    const request: WorkerRequest = {
      type: "run-first-pool-optimizer",
      id: requestId,
      inputShape: optimizerInput.inputShape,
      mean: caseData.mean,
      std: caseData.std,
      contentImageValues: optimizerInput.values,
      styleImageValues: [...optimizerInput.values].reverse(),
      initialInputValues: optimizerInput.values,
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
      steps,
    };
    const optimized = await workerClient.ask(request);
    if (optimized.type !== "run-first-pool-optimizer-result") {
      return {
        ok: false,
        reason: "unexpected-response",
        responseType: optimized.type,
      };
    }
    if (!optimized.ok) {
      return {
        ok: false,
        reason: "worker-failed",
        message: optimized.message,
      };
    }
    return {
      ok: true,
      init,
      losses: optimized.losses,
      finalValuesLength: optimized.finalValues.length,
      expectedLength: optimizerInput.values.length,
    };
  } finally {
    workerClient.dispose();
  }
};

test("first-pool optimizer loop returns finite nonnegative losses", async ({
  page,
}) => {
  test.setTimeout(300000);
  await gotoStableApp(page);

  const result: FirstPoolOptimizerResult = await page.evaluate(
    runFirstPoolOptimizerInBrowser,
    {
      requestId: "phase5-run-first-pool-optimizer",
      steps: 6,
      spatialSize: 16,
    },
  );

  test.skip(
    !result.ok && result.reason === "missing-fixtures",
    firstPoolFixtureMissingMessage,
  );
  throwIfUnexpectedFailure(result);
  if (!result.ok) return;

  expectWebGpuInitOk(result.init);
  expect(result.losses.length).toBeGreaterThan(1);
  expect(Number.isFinite(result.losses[0])).toBeTruthy();
  expect(Number.isFinite(result.losses.at(-1) ?? Number.NaN)).toBeTruthy();
  expect(result.losses[0]).toBeGreaterThanOrEqual(0);
  expect(result.losses.at(-1) ?? -1).toBeGreaterThanOrEqual(0);
});

test("first-pool optimizer supports non-16x16 dynamic spatial shape", async ({
  page,
}) => {
  test.setTimeout(300000);
  await gotoStableApp(page);

  const result: FirstPoolOptimizerResult = await page.evaluate(
    runFirstPoolOptimizerInBrowser,
    {
      requestId: "phase5-run-first-pool-optimizer-32x32",
      steps: 4,
      spatialSize: 32,
    },
  );

  test.skip(
    !result.ok && result.reason === "missing-fixtures",
    firstPoolFixtureMissingMessage,
  );
  throwIfUnexpectedFailure(result);
  if (!result.ok) return;

  expectWebGpuInitOk(result.init);
  expect(result.losses.length).toBe(4);
  expect(Number.isFinite(result.losses[0])).toBeTruthy();
  expect(Number.isFinite(result.losses.at(-1) ?? Number.NaN)).toBeTruthy();
  expect(result.finalValuesLength).toBe(result.expectedLength);
});
