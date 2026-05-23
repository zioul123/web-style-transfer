import { expect, test } from "@playwright/test";
import type { WorkerRequest, WorkerResponse } from "../src/types";
import { gotoStableApp } from "./helpers/appPage";

test("phase 5 benchmark: fused first-pool path is not slower than unfused by large margin", async ({ page }) => {
  test.setTimeout(300000);
  await gotoStableApp(page);
  const result = await page.evaluate(async () => {
    const loadJson = async <T>(url: string): Promise<T | null> => {
      const response = await fetch(url);
      if (!response.ok) return null;
      return JSON.parse(await response.text()) as T;
    };
    const weights = await loadJson<any>("/vgg19-first-pool/vgg19_first_pool_weights.json");
    const caseData = await loadJson<any>("/vgg19-first-pool/vgg19_first_pool_case_madeira16.json");
    if (weights === null || caseData === null) return { ok: false as const, reason: "missing-fixtures" as const };
    const worker = new Worker(new URL("/src/styleTransfer.worker.ts", window.location.origin), { type: "module" });
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
    await ask({ type: "init-webgpu", id: "phase5-bench-init" });
    const makePayload = (id: string, fusedConvRelu: boolean, fusedUpdate: boolean): WorkerRequest => ({
      type: "run-first-pool-optimizer",
      id,
      inputShape: caseData.inputShape,
      mean: caseData.mean,
      std: caseData.std,
      contentImageValues: caseData.inputValues,
      styleImageValues: [...caseData.inputValues].reverse(),
      initialInputValues: caseData.inputValues,
      conv1Weight: { shape: weights.conv1WeightShape, values: weights.conv1WeightValues },
      conv1Bias: weights.conv1BiasValues,
      conv2Weight: { shape: weights.conv2WeightShape, values: weights.conv2WeightValues },
      conv2Bias: weights.conv2BiasValues,
      conv3Weight: { shape: weights.conv3WeightShape, values: weights.conv3WeightValues },
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
    const baseline = await ask(makePayload("phase5-bench-baseline", false, false));
    const fused = await ask(makePayload("phase5-bench-fused", true, true));
    worker.terminate();
    return { ok: true as const, baseline, fused };
  });
  test.skip(!result.ok && result.reason === "missing-fixtures", "Missing fixtures for benchmark.");
  if (!result.ok) throw new Error("benchmark setup failed");
  if (result.baseline.type !== "run-first-pool-optimizer-result" || !result.baseline.ok) throw new Error("baseline failed");
  if (result.fused.type !== "run-first-pool-optimizer-result" || !result.fused.ok) throw new Error("fused failed");
  expect(result.baseline.stats).toBeDefined();
  expect(result.fused.stats).toBeDefined();
  if (result.baseline.stats === undefined || result.fused.stats === undefined) throw new Error("missing stats");
  // Allow noise on SwiftShader; ensure no major regression.
  expect(result.fused.stats.elapsedMs).toBeLessThan(result.baseline.stats.elapsedMs * 1.5);
  // eslint-disable-next-line no-console
  console.log("FIRST_POOL_BENCHMARK", { baseline: result.baseline.stats, fused: result.fused.stats });
});

