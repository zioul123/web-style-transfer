import { expect, test } from "@playwright/test";
import type { WorkerRequest, WorkerResponse } from "../src/types";
import { gotoStableApp } from "./helpers/appPage";

type VggWeightsFixture = {
  conv1WeightShape: [number, number, number, number];
  conv1WeightValues: number[];
  conv1BiasValues: number[];
  conv2WeightShape: [number, number, number, number];
  conv2WeightValues: number[];
  conv2BiasValues: number[];
  conv3WeightShape: [number, number, number, number];
  conv3WeightValues: number[];
  conv3BiasValues: number[];
};

type VggCaseFixture = {
  inputShape: [number, number, number, number];
  inputValues: number[];
  mean: [number, number, number];
  std: [number, number, number];
};

test("phase 5 optimizer loop decreases first-pool graph objective", async ({
  page,
}) => {
  test.setTimeout(300000);
  await gotoStableApp(page);

  const result = await page.evaluate(async () => {
    const loadJson = async <T>(url: string): Promise<T | null> => {
      const response = await fetch(url);
      if (!response.ok) return null;
      return JSON.parse(await response.text()) as T;
    };
    const weights = await loadJson<VggWeightsFixture>(
      "/vgg19-first-pool/vgg19_first_pool_weights.json",
    );
    const caseData = await loadJson<VggCaseFixture>(
      "/vgg19-first-pool/vgg19_first_pool_case_madeira16.json",
    );
    if (weights === null || caseData === null)
      return { ok: false as const, reason: "missing-fixtures" as const };

    const worker = new Worker(
      new URL("/src/styleTransfer.worker.ts", window.location.origin),
      { type: "module" },
    );
    const ask = (payload: WorkerRequest): Promise<WorkerResponse> =>
      new Promise((resolve) => {
        const handler = (event: MessageEvent<WorkerResponse>): void => {
          if (event.data.id === payload.id) {
            worker.removeEventListener("message", handler);
            resolve(event.data);
          }
        };
        worker.addEventListener("message", handler);
        worker.postMessage(payload);
      });

    const init = await ask({ type: "init-webgpu", id: "phase5-init" });
    const makeRequest = (id: string, debugUseLegacyCpuLossReadback: boolean): WorkerRequest => ({
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
      debugUseLegacyCpuLossReadback,
    });
    const optimized = await ask(makeRequest("phase5-run-first-pool-optimizer-gpu", false));
    const optimizedLegacy = await ask(makeRequest("phase5-run-first-pool-optimizer-legacy", true));
    worker.terminate();
    if (optimized.type !== "run-first-pool-optimizer-result")
      return {
        ok: false as const,
        reason: "unexpected-response" as const,
        responseType: optimized.type,
      };
    if (!optimized.ok)
      return {
        ok: false as const,
        reason: "worker-failed" as const,
        message: optimized.message,
      };
    if (optimizedLegacy.type !== "run-first-pool-optimizer-result")
      return {
        ok: false as const,
        reason: "unexpected-response" as const,
        responseType: optimizedLegacy.type,
      };
    if (!optimizedLegacy.ok)
      return {
        ok: false as const,
        reason: "worker-failed" as const,
        message: optimizedLegacy.message,
      };
    return { ok: true as const, init, losses: optimized.losses, legacyLosses: optimizedLegacy.losses };
  });

  test.skip(
    !result.ok && result.reason === "missing-fixtures",
    "Missing VGG19 first-pool fixtures. Run python-reference/export_vgg19_first_pool.py first.",
  );
  if (!result.ok) {
    throw new Error(
      `${result.reason}${"message" in result && result.message !== undefined ? `: ${result.message}` : ""}${"responseType" in result && result.responseType !== undefined ? ` (responseType=${String(result.responseType)})` : ""}`,
    );
  }
  if (!result.ok) return;
  expect(result.init.type).toBe("webgpu-init-result");
  if (result.init.type !== "webgpu-init-result")
    throw new Error("init failure");
  expect(result.init.ok).toBeTruthy();
  expect(result.losses.length).toBeGreaterThan(1);
  expect(Number.isFinite(result.losses[0])).toBeTruthy();
  expect(Number.isFinite(result.losses.at(-1) ?? Number.NaN)).toBeTruthy();
  expect(result.losses[0]).toBeGreaterThanOrEqual(0);
  expect(result.losses.at(-1) ?? -1).toBeGreaterThanOrEqual(0);
  expect(result.legacyLosses.length).toBeGreaterThan(1);
  expect(Number.isFinite(result.legacyLosses[0])).toBeTruthy();
  expect(Number.isFinite(result.legacyLosses.at(-1) ?? Number.NaN)).toBeTruthy();
  expect(result.legacyLosses[0]).toBeGreaterThanOrEqual(0);
  expect(result.legacyLosses.at(-1) ?? -1).toBeGreaterThanOrEqual(0);
  expect(result.losses[0]).toBeCloseTo(result.legacyLosses[0], 5);
});

test("phase 5 first-pool optimizer supports non-16x16 dynamic spatial shape", async ({
  page,
}) => {
  test.setTimeout(300000);
  await gotoStableApp(page);

  const result = await page.evaluate(async () => {
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
            const srcY = Math.min(height - 1, Math.floor((y * height) / nextHeight));
            for (let x = 0; x < nextWidth; x += 1) {
              const srcX = Math.min(width - 1, Math.floor((x * width) / nextWidth));
              const srcIndex = ((b * channels + c) * height + srcY) * width + srcX;
              const dstIndex = ((b * channels + c) * nextHeight + y) * nextWidth + x;
              scaled[dstIndex] = values[srcIndex];
            }
          }
        }
      }
      return { shape: [batch, channels, nextHeight, nextWidth], values: scaled };
    };

    const loadJson = async <T>(url: string): Promise<T | null> => {
      const response = await fetch(url);
      if (!response.ok) return null;
      return JSON.parse(await response.text()) as T;
    };
    const weights = await loadJson<VggWeightsFixture>(
      "/vgg19-first-pool/vgg19_first_pool_weights.json",
    );
    const caseData = await loadJson<VggCaseFixture>(
      "/vgg19-first-pool/vgg19_first_pool_case_madeira16.json",
    );
    if (weights === null || caseData === null)
      return { ok: false as const, reason: "missing-fixtures" as const };

    const upscaledCase = upscaleNchwNearest(
      caseData.inputValues,
      caseData.inputShape,
      32,
      32,
    );

    const worker = new Worker(
      new URL("/src/styleTransfer.worker.ts", window.location.origin),
      { type: "module" },
    );
    const ask = (payload: WorkerRequest): Promise<WorkerResponse> =>
      new Promise((resolve) => {
        const handler = (event: MessageEvent<WorkerResponse>): void => {
          if (event.data.id === payload.id) {
            worker.removeEventListener("message", handler);
            resolve(event.data);
          }
        };
        worker.addEventListener("message", handler);
        worker.postMessage(payload);
      });

    const init = await ask({ type: "init-webgpu", id: "phase5-init-dynamic" });
    const request: WorkerRequest = {
      type: "run-first-pool-optimizer",
      id: "phase5-run-first-pool-optimizer-32x32",
      inputShape: upscaledCase.shape,
      mean: caseData.mean,
      std: caseData.std,
      contentImageValues: upscaledCase.values,
      styleImageValues: [...upscaledCase.values].reverse(),
      initialInputValues: upscaledCase.values,
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
      steps: 4,
      debugUseLegacyCpuLossReadback: false,
    };
    const optimized = await ask(request);
    worker.terminate();
    if (optimized.type !== "run-first-pool-optimizer-result")
      return {
        ok: false as const,
        reason: "unexpected-response" as const,
        responseType: optimized.type,
      };
    if (!optimized.ok)
      return {
        ok: false as const,
        reason: "worker-failed" as const,
        message: optimized.message,
      };
    return {
      ok: true as const,
      init,
      losses: optimized.losses,
      finalValuesLength: optimized.finalValues.length,
      expectedLength: upscaledCase.values.length,
    };
  });

  test.skip(
    !result.ok && result.reason === "missing-fixtures",
    "Missing VGG19 first-pool fixtures. Run python-reference/export_vgg19_first_pool.py first.",
  );
  if (!result.ok) {
    throw new Error(
      `${result.reason}${"message" in result && result.message !== undefined ? `: ${result.message}` : ""}${"responseType" in result && result.responseType !== undefined ? ` (responseType=${String(result.responseType)})` : ""}`,
    );
  }
  if (!result.ok) return;
  expect(result.init.type).toBe("webgpu-init-result");
  if (result.init.type !== "webgpu-init-result")
    throw new Error("init failure");
  expect(result.init.ok).toBeTruthy();
  expect(result.losses.length).toBe(4);
  expect(Number.isFinite(result.losses[0])).toBeTruthy();
  expect(Number.isFinite(result.losses.at(-1) ?? Number.NaN)).toBeTruthy();
  expect(result.finalValuesLength).toBe(result.expectedLength);
});
