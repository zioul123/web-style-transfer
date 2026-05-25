import { expect, test } from "@playwright/test";
import type { WorkerRequest, WorkerResponse } from "../src/types";
import { gotoStableApp } from "./helpers/appPage";

type Phase3FullPassFixture = {
  inputShape: [number, number, number, number];
  inputImageValues: number[];
  contentImageValues: number[];
  styleImageValues: number[];
  mean: [number, number, number];
  std: [number, number, number];
  styleLayerIndices: number[];
  contentLayerIndex: number;
  expectedGradients?: {
    content?: number[];
    styleByLayer?: Record<string, number[]>;
    total?: number[];
  };
};

test("phase 6 UI defaults match python reference controls", async ({
  page,
}) => {
  await gotoStableApp(page);
  await expect(page.getByLabel("Optimizer")).toHaveValue("lbfgs");
  await expect(page.getByLabel("Content resolution")).toHaveValue("128x128");
  await expect(page.getByLabel("Style resolution")).toHaveValue("128x192");
  await expect(page.getByLabel("Style weight")).toHaveValue("500000");
  await expect(page.getByLabel("Content weight")).toHaveValue("1");
  await expect(page.getByLabel("Learning rate")).toHaveValue("1");
  await expect(page.getByLabel("L-BFGS history")).toHaveValue("100");
  await expect(page.getByLabel("L-BFGS tolerance change")).toHaveValue("1e-9");
});

test("phase 5 full style transfer endpoint returns losses", async ({
  page,
}) => {
  test.setTimeout(300000);
  await gotoStableApp(page);
  const result = await page.evaluate(async () => {
    const loadJson = async <T>(url: string): Promise<T | null> => {
      const response = await fetch(url);
      if (!response.ok) return null;
      const text = await response.text();
      try {
        return JSON.parse(text) as T;
      } catch {
        return null;
      }
    };
    const weights = await loadJson<
      Record<string, number[] | [number, number, number, number]>
    >("/vgg19-phase3-full-pass/vgg19_conv0_to_conv28_weights.json");
    const fixture = await loadJson<Phase3FullPassFixture>(
      "/vgg19-phase3-full-pass/vgg19_phase3_full_pass_fixture.json",
    );
    if (weights === null || fixture === null)
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
    await ask({ type: "init-webgpu", id: "phase5-full-init" });
    const out = await ask({
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
    worker.terminate();
    if (out.type !== "run-style-transfer-result")
      return { ok: false as const, reason: "wrong-response" as const };
    if (!out.ok)
      return {
        ok: false as const,
        reason: "worker-failed" as const,
        message: out.message,
      };
    return { ok: true as const, losses: out.losses };
  });
  test.skip(
    !result.ok && result.reason === "missing-fixtures",
    "Missing phase3 full-pass fixtures. Run python-reference/export_vgg19_phase3_full_pass.py first.",
  );
  if (!result.ok && result.reason !== "missing-fixtures")
    throw new Error(
      result.reason === "worker-failed" ? result.message : result.reason,
    );
  if (!result.ok) return;
  expect(result.losses.length).toBe(5);
  expect(result.losses.every((loss) => Number.isFinite(loss))).toBeTruthy();
});

test("phase 5 full style transfer supports rectangular style shape", async ({
  page,
}) => {
  test.setTimeout(300000);
  await gotoStableApp(page);
  const result = await page.evaluate(async () => {
    const loadJson = async <T>(url: string): Promise<T | null> => {
      const response = await fetch(url);
      if (!response.ok) return null;
      const text = await response.text();
      try {
        return JSON.parse(text) as T;
      } catch {
        return null;
      }
    };
    const weights = await loadJson<
      Record<string, number[] | [number, number, number, number]>
    >("/vgg19-phase3-full-pass/vgg19_conv0_to_conv28_weights.json");
    if (weights === null)
      return { ok: false as const, reason: "missing-fixtures" as const };

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
    await ask({ type: "init-webgpu", id: "phase5-rect-style-init" });
    const out = await ask({
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
    worker.terminate();
    if (out.type !== "run-style-transfer-result")
      return { ok: false as const, reason: "wrong-response" as const };
    if (!out.ok)
      return {
        ok: false as const,
        reason: "worker-failed" as const,
        message: out.message,
      };
    return {
      ok: true as const,
      losses: out.losses,
      finalValuesLength: out.finalValues.length,
      expectedLength: inputImageValues.length,
    };
  });
  test.skip(
    !result.ok && result.reason === "missing-fixtures",
    "Missing phase3 full-pass weights.",
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

test("phase 5 run-style-transfer first-step gradient matches pytorch oracle", async ({
  page,
}) => {
  test.setTimeout(300000);
  await gotoStableApp(page);
  const result = await page.evaluate(async () => {
    const loadJson = async <T>(url: string): Promise<T | null> => {
      const response = await fetch(url);
      if (!response.ok) return null;
      const text = await response.text();
      try {
        return JSON.parse(text) as T;
      } catch {
        return null;
      }
    };
    const weights = await loadJson<
      Record<string, number[] | [number, number, number, number]>
    >("/vgg19-phase3-full-pass/vgg19_conv0_to_conv28_weights.json");
    const fixture = await loadJson<Phase3FullPassFixture>(
      "/vgg19-phase3-full-pass/vgg19_phase3_full_pass_fixture.json",
    );
    if (weights === null || fixture === null)
      return { ok: false as const, reason: "missing-fixtures" as const };
    if (fixture.expectedGradients?.total === undefined)
      return {
        ok: false as const,
        reason: "missing-gradient-fixture" as const,
      };
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
    await ask({ type: "init-webgpu", id: "phase5-grad-init" });
    const learningRate = 1e-5;
    const out = await ask({
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
    worker.terminate();
    if (out.type !== "run-style-transfer-result")
      return { ok: false as const, reason: "wrong-response" as const };
    if (!out.ok)
      return {
        ok: false as const,
        reason: "worker-failed" as const,
        message: out.message,
      };
    const gradOracle = fixture.expectedGradients.total as number[];
    const gradObserved = out.finalValues.map(
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
    return { ok: true as const, maxDiff, meanDiff };
  });
  test.skip(
    !result.ok && result.reason === "missing-fixtures",
    "Missing phase3 full-pass fixtures. Run python-reference/export_vgg19_phase3_full_pass.py first.",
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
  expect(result.maxDiff).toBeLessThan(2e-1);
  expect(result.meanDiff).toBeLessThan(2e-2);
});

test("phase 5 weighted first-step gradient composes content and style gradients", async ({
  page,
}) => {
  test.setTimeout(300000);
  await gotoStableApp(page);
  const result = await page.evaluate(async () => {
    const loadJson = async <T>(url: string): Promise<T | null> => {
      const response = await fetch(url);
      if (!response.ok) return null;
      const text = await response.text();
      try {
        return JSON.parse(text) as T;
      } catch {
        return null;
      }
    };
    const weights = await loadJson<
      Record<string, number[] | [number, number, number, number]>
    >("/vgg19-phase3-full-pass/vgg19_conv0_to_conv28_weights.json");
    const fixture = await loadJson<Phase3FullPassFixture>(
      "/vgg19-phase3-full-pass/vgg19_phase3_full_pass_fixture.json",
    );
    if (weights === null || fixture === null)
      return { ok: false as const, reason: "missing-fixtures" as const };
    if (
      fixture.expectedGradients?.content === undefined ||
      fixture.expectedGradients.styleByLayer === undefined
    )
      return {
        ok: false as const,
        reason: "missing-gradient-fixture" as const,
      };
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
    await ask({ type: "init-webgpu", id: "phase5-weighted-grad-init" });
    const learningRate = 1e-5;
    const contentWeight = 2.5;
    const styleWeight = 7;
    const out = await ask({
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
    worker.terminate();
    if (out.type !== "run-style-transfer-result")
      return { ok: false as const, reason: "wrong-response" as const };
    if (!out.ok)
      return {
        ok: false as const,
        reason: "worker-failed" as const,
        message: out.message,
      };
    const contentGrad = fixture.expectedGradients.content;
    const styleByLayer = fixture.expectedGradients.styleByLayer;
    const gradObserved = out.finalValues.map(
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
    return { ok: true as const, maxDiff, meanDiff };
  });
  test.skip(
    !result.ok && result.reason === "missing-fixtures",
    "Missing phase3 full-pass fixtures. Run python-reference/export_vgg19_phase3_full_pass.py first.",
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
  expect(result.maxDiff).toBeLessThan(6e-1);
  expect(result.meanDiff).toBeLessThan(2e-2);
});

test("phase 5 style transfer supports adam and lbfgs", async ({
  page,
}) => {
  test.setTimeout(300000);
  await gotoStableApp(page);
  const result = await page.evaluate(async () => {
    const loadJson = async <T>(url: string): Promise<T | null> => {
      const response = await fetch(url);
      if (!response.ok) return null;
      const text = await response.text();
      try {
        return JSON.parse(text) as T;
      } catch {
        return null;
      }
    };
    const weights = await loadJson<
      Record<string, number[] | [number, number, number, number]>
    >("/vgg19-phase3-full-pass/vgg19_conv0_to_conv28_weights.json");
    const fixture = await loadJson<Phase3FullPassFixture>(
      "/vgg19-phase3-full-pass/vgg19_phase3_full_pass_fixture.json",
    );
    if (weights === null || fixture === null)
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
    await ask({ type: "init-webgpu", id: "phase5-optimizers-init" });
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
    for (const testCase of cases) {
      const run = await ask({
        ...common,
        id: `phase5-optimizer-${testCase.id}`,
        optimizer: testCase.optimizer,
        steps: testCase.steps,
      });
      if (run.type !== "run-style-transfer-result") {
        worker.terminate();
        return { ok: false as const, reason: "wrong-response" as const };
      }
      if (!run.ok) {
        worker.terminate();
        return {
          ok: false as const,
          reason: "worker-failed" as const,
          message: run.message,
        };
      }
      results.push({
        id: testCase.id,
        finalLoss: run.losses[run.losses.length - 1],
        finalValuesLength: run.finalValues.length,
      });
    }
    worker.terminate();
    return {
      ok: true as const,
      results,
      expectedLength: fixture.inputImageValues.length,
    };
  });
  test.skip(
    !result.ok && result.reason === "missing-fixtures",
    "Missing phase3 full-pass fixtures.",
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

test("phase 5 fused conv+relu op matches separate ops", async ({ page }) => {
  test.setTimeout(120000);
  await gotoStableApp(page);
  const result = await page.evaluate(async () => {
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
    await ask({ type: "init-webgpu", id: "phase5-fused-init" });
    const input = Array.from(
      { length: 3 * 8 * 8 },
      (_, i) => Math.sin(i * 0.13) * 0.5,
    );
    const weight = Array.from(
      { length: 4 * 3 * 3 * 3 },
      (_, i) => Math.cos(i * 0.07) * 0.1,
    );
    const bias = [0.01, -0.02, 0.03, -0.04];
    const conv = await ask({
      type: "tensor-op",
      id: "phase5-fused-conv",
      op: "conv2d-forward",
      input: { shape: [1, 3, 8, 8], values: input },
      weight: { shape: [4, 3, 3, 3], values: weight },
      bias,
    });
    if (conv.type !== "tensor-op-result" || !conv.ok || !("values" in conv))
      return { ok: false as const, reason: "conv-failed" as const };
    const relu = await ask({
      type: "tensor-op",
      id: "phase5-fused-relu",
      op: "relu-forward",
      input: { shape: [1, 4, 8, 8], values: conv.values },
    });
    const fused = await ask({
      type: "tensor-op",
      id: "phase5-fused-fused",
      op: "conv2d-relu-forward",
      input: { shape: [1, 3, 8, 8], values: input },
      weight: { shape: [4, 3, 3, 3], values: weight },
      bias,
    });
    worker.terminate();
    if (relu.type !== "tensor-op-result" || !relu.ok || !("values" in relu))
      return { ok: false as const, reason: "relu-failed" as const };
    if (fused.type !== "tensor-op-result" || !fused.ok || !("values" in fused))
      return { ok: false as const, reason: "fused-failed" as const };
    let maxDiff = 0;
    for (let i = 0; i < relu.values.length; i += 1)
      maxDiff = Math.max(maxDiff, Math.abs(relu.values[i] - fused.values[i]));
    return { ok: true as const, maxDiff };
  });
  if (!result.ok) throw new Error(result.reason);
  expect(result.maxDiff).toBeLessThan(1e-4);
});
