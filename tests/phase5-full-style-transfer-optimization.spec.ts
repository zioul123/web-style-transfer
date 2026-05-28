import { expect, test } from "@playwright/test";
import type { WorkerRequest, WorkerResponse } from "../src/types";
import { gotoStableApp } from "./helpers/appPage";



type ManifestTensorEntry = {
  shape: number[];
  dtype: "float32";
  shard: string;
  offset: number;
  length: number;
};

type Vgg19Manifest = {
  layers: Record<string, { weight: ManifestTensorEntry; bias: ManifestTensorEntry }>;
  shards: Array<{ name: string; byteLength: number }>;
};

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

test("phase 6 model pack selector reloads manifest-backed weights and supports fp32", async ({
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

  await expect(page.getByLabel("Model pack")).toHaveValue("int8log-per-channel");
  await expect
    .poll(async () => {
      const urls = await page.evaluate(() => {
        const state = window as Window & { __packFetches?: string[] };
        return state.__packFetches ?? [];
      });
      return urls.some((url) => url.includes("/vgg19-models/int8log-per-channel/manifest.json"));
    })
    .toBeTruthy();

  await page.getByLabel("Model pack").selectOption("fp32");
  await expect(page.getByLabel("Model pack")).toHaveValue("fp32");
  await expect
    .poll(async () => {
      const urls = await page.evaluate(() => {
        const state = window as Window & { __packFetches?: string[] };
        return state.__packFetches ?? [];
      });
      return urls.some((url) => url.includes("/vgg19-models/fp32/manifest.json"));
    })
    .toBeTruthy();
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
    const manifest = await loadJson<Vgg19Manifest>("/vgg19-models/fp32/manifest.json");
    if (manifest === null) return { ok: false as const, reason: "missing-fixtures" as const };
    const shardPayload = await Promise.all(
      manifest.shards.map(async (shard) => {
        const response = await fetch(`/vgg19-models/fp32/${shard.name}`);
        if (!response.ok) return null;
        return [shard.name, await response.arrayBuffer()] as const;
      }),
    );
    if (shardPayload.some((entry) => entry === null))
      return { ok: false as const, reason: "missing-fixtures" as const };
    const shardMap: Record<string, ArrayBuffer> = Object.fromEntries(
      shardPayload as Array<readonly [string, ArrayBuffer]>,
    );
    const weights: Record<string, number[] | [number, number, number, number]> = {};
    for (let layerIndex = 0; layerIndex <= 29; layerIndex += 1) {
      const layer = manifest.layers[`conv${layerIndex}`];
      if (layer === undefined) continue;
      const weightBytes = shardMap[layer.weight.shard].slice(
        layer.weight.offset,
        layer.weight.offset + layer.weight.length,
      );
      const biasBytes = shardMap[layer.bias.shard].slice(
        layer.bias.offset,
        layer.bias.offset + layer.bias.length,
      );
      weights[`conv${layerIndex}.weightShape`] = layer.weight.shape as [number, number, number, number];
      weights[`conv${layerIndex}.weightValues`] = Array.from(new Float32Array(weightBytes));
      weights[`conv${layerIndex}.biasValues`] = Array.from(new Float32Array(biasBytes));
    }
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
    const manifest = await loadJson<Vgg19Manifest>("/vgg19-models/fp32/manifest.json");
    if (manifest === null) return { ok: false as const, reason: "missing-fixtures" as const };
    const shardPayload = await Promise.all(
      manifest.shards.map(async (shard) => {
        const response = await fetch(`/vgg19-models/fp32/${shard.name}`);
        if (!response.ok) return null;
        return [shard.name, await response.arrayBuffer()] as const;
      }),
    );
    if (shardPayload.some((entry) => entry === null))
      return { ok: false as const, reason: "missing-fixtures" as const };
    const shardMap: Record<string, ArrayBuffer> = Object.fromEntries(
      shardPayload as Array<readonly [string, ArrayBuffer]>,
    );
    const weights: Record<string, number[] | [number, number, number, number]> = {};
    for (let layerIndex = 0; layerIndex <= 29; layerIndex += 1) {
      const layer = manifest.layers[`conv${layerIndex}`];
      if (layer === undefined) continue;
      const weightBytes = shardMap[layer.weight.shard].slice(
        layer.weight.offset,
        layer.weight.offset + layer.weight.length,
      );
      const biasBytes = shardMap[layer.bias.shard].slice(
        layer.bias.offset,
        layer.bias.offset + layer.bias.length,
      );
      weights[`conv${layerIndex}.weightShape`] = layer.weight.shape as [number, number, number, number];
      weights[`conv${layerIndex}.weightValues`] = Array.from(new Float32Array(weightBytes));
      weights[`conv${layerIndex}.biasValues`] = Array.from(new Float32Array(biasBytes));
    }
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
    const manifest = await loadJson<Vgg19Manifest>("/vgg19-models/fp32/manifest.json");
    if (manifest === null) return { ok: false as const, reason: "missing-fixtures" as const };
    const shardPayload = await Promise.all(
      manifest.shards.map(async (shard) => {
        const response = await fetch(`/vgg19-models/fp32/${shard.name}`);
        if (!response.ok) return null;
        return [shard.name, await response.arrayBuffer()] as const;
      }),
    );
    if (shardPayload.some((entry) => entry === null))
      return { ok: false as const, reason: "missing-fixtures" as const };
    const shardMap: Record<string, ArrayBuffer> = Object.fromEntries(
      shardPayload as Array<readonly [string, ArrayBuffer]>,
    );
    const weights: Record<string, number[] | [number, number, number, number]> = {};
    for (let layerIndex = 0; layerIndex <= 29; layerIndex += 1) {
      const layer = manifest.layers[`conv${layerIndex}`];
      if (layer === undefined) continue;
      const weightBytes = shardMap[layer.weight.shard].slice(
        layer.weight.offset,
        layer.weight.offset + layer.weight.length,
      );
      const biasBytes = shardMap[layer.bias.shard].slice(
        layer.bias.offset,
        layer.bias.offset + layer.bias.length,
      );
      weights[`conv${layerIndex}.weightShape`] = layer.weight.shape as [number, number, number, number];
      weights[`conv${layerIndex}.weightValues`] = Array.from(new Float32Array(weightBytes));
      weights[`conv${layerIndex}.biasValues`] = Array.from(new Float32Array(biasBytes));
    }
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
    const manifest = await loadJson<Vgg19Manifest>("/vgg19-models/fp32/manifest.json");
    if (manifest === null) return { ok: false as const, reason: "missing-fixtures" as const };
    const shardPayload = await Promise.all(
      manifest.shards.map(async (shard) => {
        const response = await fetch(`/vgg19-models/fp32/${shard.name}`);
        if (!response.ok) return null;
        return [shard.name, await response.arrayBuffer()] as const;
      }),
    );
    if (shardPayload.some((entry) => entry === null))
      return { ok: false as const, reason: "missing-fixtures" as const };
    const shardMap: Record<string, ArrayBuffer> = Object.fromEntries(
      shardPayload as Array<readonly [string, ArrayBuffer]>,
    );
    const weights: Record<string, number[] | [number, number, number, number]> = {};
    for (let layerIndex = 0; layerIndex <= 29; layerIndex += 1) {
      const layer = manifest.layers[`conv${layerIndex}`];
      if (layer === undefined) continue;
      const weightBytes = shardMap[layer.weight.shard].slice(
        layer.weight.offset,
        layer.weight.offset + layer.weight.length,
      );
      const biasBytes = shardMap[layer.bias.shard].slice(
        layer.bias.offset,
        layer.bias.offset + layer.bias.length,
      );
      weights[`conv${layerIndex}.weightShape`] = layer.weight.shape as [number, number, number, number];
      weights[`conv${layerIndex}.weightValues`] = Array.from(new Float32Array(weightBytes));
      weights[`conv${layerIndex}.biasValues`] = Array.from(new Float32Array(biasBytes));
    }
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
    const manifest = await loadJson<Vgg19Manifest>("/vgg19-models/fp32/manifest.json");
    if (manifest === null) return { ok: false as const, reason: "missing-fixtures" as const };
    const shardPayload = await Promise.all(
      manifest.shards.map(async (shard) => {
        const response = await fetch(`/vgg19-models/fp32/${shard.name}`);
        if (!response.ok) return null;
        return [shard.name, await response.arrayBuffer()] as const;
      }),
    );
    if (shardPayload.some((entry) => entry === null))
      return { ok: false as const, reason: "missing-fixtures" as const };
    const shardMap: Record<string, ArrayBuffer> = Object.fromEntries(
      shardPayload as Array<readonly [string, ArrayBuffer]>,
    );
    const weights: Record<string, number[] | [number, number, number, number]> = {};
    for (let layerIndex = 0; layerIndex <= 29; layerIndex += 1) {
      const layer = manifest.layers[`conv${layerIndex}`];
      if (layer === undefined) continue;
      const weightBytes = shardMap[layer.weight.shard].slice(
        layer.weight.offset,
        layer.weight.offset + layer.weight.length,
      );
      const biasBytes = shardMap[layer.bias.shard].slice(
        layer.bias.offset,
        layer.bias.offset + layer.bias.length,
      );
      weights[`conv${layerIndex}.weightShape`] = layer.weight.shape as [number, number, number, number];
      weights[`conv${layerIndex}.weightValues`] = Array.from(new Float32Array(weightBytes));
      weights[`conv${layerIndex}.biasValues`] = Array.from(new Float32Array(biasBytes));
    }
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

test("phase 5 gpu vector reductions match cpu references", async ({ page }) => {
  test.setTimeout(120000);
  await gotoStableApp(page);
  const result = await page.evaluate(async () => {
    if (!("gpu" in navigator))
      return { ok: false as const, reason: "webgpu-unavailable" as const };
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter === null)
      return { ok: false as const, reason: "adapter-unavailable" as const };
    const device = await adapter.requestDevice();
    const { createGpuVectorOps } = await import(
      "/src/ml/worker/pipelines/optimization/input-optimizer/gpuVectorOps.ts"
    );
    const { uploadToOwnedBuffer, releaseOwnedBuffer } = await import(
      "/src/ml/worker/runtime/bufferKernels.ts"
    );

    const count = 3 * 128 + 17;
    const a = new Float32Array(count);
    const b = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
      a[i] = Math.sin(i * 0.17) * 0.75;
      b[i] = Math.cos(i * 0.11) * 0.5;
    }

    let expectedDot = 0;
    let expectedAbs = 0;
    for (let i = 0; i < count; i += 1) {
      expectedDot += a[i] * b[i];
      expectedAbs += Math.abs(a[i]);
    }

    const aBuffer = uploadToOwnedBuffer(device, a);
    const bBuffer = uploadToOwnedBuffer(device, b);
    try {
      const ops = createGpuVectorOps(device, count);
      const dot = await ops.dot(aBuffer, bBuffer);
      const abs = await ops.absSum(aBuffer);
      return {
        ok: true as const,
        dot,
        abs,
        expectedDot,
        expectedAbs,
        dotDiff: Math.abs(dot - expectedDot),
        absDiff: Math.abs(abs - expectedAbs),
      };
    } finally {
      releaseOwnedBuffer(aBuffer);
      releaseOwnedBuffer(bBuffer);
      device.destroy();
    }
  });
  if (!result.ok) throw new Error(result.reason);
  expect(
    result.dotDiff,
    `dot=${result.dot}, expectedDot=${result.expectedDot}`,
  ).toBeLessThan(1e-2);
  expect(
    result.absDiff,
    `abs=${result.abs}, expectedAbs=${result.expectedAbs}`,
  ).toBeLessThan(1e-2);
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

test("phase 6 style-transfer supports gram/style kernel variants", async ({
  page,
}) => {
  test.setTimeout(300000);
  await gotoStableApp(page);
  const result = await page.evaluate(async () => {
    const loadJson = async <T>(url: string): Promise<T | null> => {
      const response = await fetch(url);
      if (!response.ok) return null;
      try {
        return (await response.json()) as T;
      } catch {
        return null;
      }
    };
    const manifest = await loadJson<Vgg19Manifest>("/vgg19-models/fp32/manifest.json");
    const fixture = await loadJson<Phase3FullPassFixture>(
      "/vgg19-phase3-full-pass/vgg19_phase3_full_pass_fixture.json",
    );
    if (manifest === null || fixture === null)
      return { ok: false as const, reason: "missing-fixtures" as const };
    const shardPayload = await Promise.all(
      manifest.shards.map(async (shard) => {
        const response = await fetch(`/vgg19-models/fp32/${shard.name}`);
        if (!response.ok) return null;
        return [shard.name, await response.arrayBuffer()] as const;
      }),
    );
    if (shardPayload.some((entry) => entry === null))
      return { ok: false as const, reason: "missing-fixtures" as const };
    const shardMap: Record<string, ArrayBuffer> = Object.fromEntries(
      shardPayload as Array<readonly [string, ArrayBuffer]>,
    );
    const weights: Record<string, number[] | [number, number, number, number]> = {};
    for (let layerIndex = 0; layerIndex <= 29; layerIndex += 1) {
      const layer = manifest.layers[`conv${layerIndex}`];
      if (layer === undefined) continue;
      const weightBytes = shardMap[layer.weight.shard].slice(layer.weight.offset, layer.weight.offset + layer.weight.length);
      const biasBytes = shardMap[layer.bias.shard].slice(layer.bias.offset, layer.bias.offset + layer.bias.length);
      weights[`conv${layerIndex}.weightShape`] = layer.weight.shape as [number, number, number, number];
      weights[`conv${layerIndex}.weightValues`] = Array.from(new Float32Array(weightBytes));
      weights[`conv${layerIndex}.biasValues`] = Array.from(new Float32Array(biasBytes));
    }
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
    await ask({ type: "init-webgpu", id: "phase6-kernel-variant-init" });
    const runVariant = async (
      id: string,
      gramKernel: "parallel-dot" | "symmetric-parallel-dot",
      styleBackward: "two-pass" | "fused-from-gram-diff",
    ): Promise<{ ok: true; loss: number } | { ok: false; reason: string }> => {
      const out = await ask({
        type: "run-style-transfer",
        id,
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
        steps: 1,
        optimizer: "sgd",
        kernelFlags: {
          gramKernel,
          styleBackward,
          usePersistentWeightBuffers: true,
        },
      });
      if (out.type !== "run-style-transfer-result" || !out.ok || out.losses.length === 0) {
        return { ok: false, reason: out.type === "run-style-transfer-result" ? out.message : "wrong-response" };
      }
      return { ok: true, loss: out.losses[0] };
    };

    const variantA = await runVariant(
      "phase6-kernel-variant-a",
      "parallel-dot",
      "two-pass",
    );
    const variantB = await runVariant(
      "phase6-kernel-variant-b",
      "symmetric-parallel-dot",
      "fused-from-gram-diff",
    );
    worker.terminate();
    if (!variantA.ok) return variantA;
    if (!variantB.ok) return variantB;
    return {
      ok: true as const,
      lossA: variantA.loss,
      lossB: variantB.loss,
      delta: Math.abs(variantA.loss - variantB.loss),
    };
  });
  test.skip(
    !result.ok && result.reason === "missing-fixtures",
    "Missing phase3 full-pass fixtures. Run python-reference/export_vgg19_phase3_full_pass.py first.",
  );
  if (!result.ok && result.reason !== "missing-fixtures")
    throw new Error(result.reason);
  if (!result.ok) return;
  expect(Number.isFinite(result.lossA)).toBeTruthy();
  expect(Number.isFinite(result.lossB)).toBeTruthy();
  expect(result.delta).toBeLessThan(0.5);
});

test("phase 6 kernel flags preserve baseline semantics and support fp16-storage", async ({
  page,
}) => {
  test.setTimeout(300000);
  await gotoStableApp(page);
  const result = await page.evaluate(async () => {
    const loadJson = async <T>(url: string): Promise<T | null> => {
      const response = await fetch(url);
      if (!response.ok) return null;
      try {
        return (await response.json()) as T;
      } catch {
        return null;
      }
    };
    const manifest = await loadJson<Vgg19Manifest>("/vgg19-models/fp32/manifest.json");
    const fixture = await loadJson<Phase3FullPassFixture>(
      "/vgg19-phase3-full-pass/vgg19_phase3_full_pass_fixture.json",
    );
    if (manifest === null || fixture === null)
      return { ok: false as const, reason: "missing-fixtures" as const };
    const shardPayload = await Promise.all(
      manifest.shards.map(async (shard) => {
        const response = await fetch(`/vgg19-models/fp32/${shard.name}`);
        if (!response.ok) return null;
        return [shard.name, await response.arrayBuffer()] as const;
      }),
    );
    if (shardPayload.some((entry) => entry === null))
      return { ok: false as const, reason: "missing-fixtures" as const };
    const shardMap: Record<string, ArrayBuffer> = Object.fromEntries(
      shardPayload as Array<readonly [string, ArrayBuffer]>,
    );
    const weights: Record<string, number[] | [number, number, number, number]> = {};
    for (let layerIndex = 0; layerIndex <= 29; layerIndex += 1) {
      const layer = manifest.layers[`conv${layerIndex}`];
      if (layer === undefined) continue;
      const weightBytes = shardMap[layer.weight.shard].slice(layer.weight.offset, layer.weight.offset + layer.weight.length);
      const biasBytes = shardMap[layer.bias.shard].slice(layer.bias.offset, layer.bias.offset + layer.bias.length);
      weights[`conv${layerIndex}.weightShape`] = layer.weight.shape as [number, number, number, number];
      weights[`conv${layerIndex}.weightValues`] = Array.from(new Float32Array(weightBytes));
      weights[`conv${layerIndex}.biasValues`] = Array.from(new Float32Array(biasBytes));
    }
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
    await ask({ type: "init-webgpu", id: "phase6-kernel-storage-init" });
    type RunStyleRequest = Extract<WorkerRequest, { type: "run-style-transfer" }>;
    const baseRequest = (id: string): Omit<RunStyleRequest, "kernelFlags"> => ({
      type: "run-style-transfer",
      id,
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
      steps: 1,
      optimizer: "sgd",
    });
    const runVariant = async (
      id: string,
      kernelFlags?: RunStyleRequest["kernelFlags"],
    ): Promise<
      | { ok: true; loss: number; finalValues: number[] }
      | { ok: false; reason: string }
    > => {
      const out = await ask({ ...baseRequest(id), kernelFlags });
      if (out.type !== "run-style-transfer-result") return { ok: false, reason: "wrong-response" };
      if (!out.ok) return { ok: false, reason: out.message };
      return {
        ok: true,
        loss: out.losses[0] ?? Number.NaN,
        finalValues: out.finalValues,
      };
    };
    const runFailure = async (
      id: string,
      kernelFlags: RunStyleRequest["kernelFlags"],
    ): Promise<string> => {
      const out = await ask({ ...baseRequest(id), kernelFlags });
      if (out.type !== "run-style-transfer-result") return "wrong-response";
      return out.ok ? "unexpected-success" : out.message;
    };
    const meanAbsDelta = (a: readonly number[], b: readonly number[]): number => {
      if (a.length !== b.length) return Number.POSITIVE_INFINITY;
      let total = 0;
      for (let i = 0; i < a.length; i += 1) total += Math.abs(a[i] - b[i]);
      return total / a.length;
    };

    const baseline = await runVariant("phase6-kernel-baseline-default");
    const explicitPooled = await runVariant("phase6-kernel-explicit-pooled", {
      useStepBufferPool: true,
    });
    const fp16 = await runVariant("phase6-kernel-fp16-storage", {
      usePersistentWeightBuffers: true,
      weightStorage: "fp16-storage",
    });
    const tiledMessage = await runFailure("phase6-kernel-unsupported-tiled", {
      convForwardKernel: "tiled-spatial",
    });
    const spatialBackwardMessage = await runFailure("phase6-kernel-unsupported-spatial-backward", {
      convBackwardInputKernel: "spatial-vec4",
    });
    const fp16WithoutPersistentMessage = await runFailure("phase6-kernel-fp16-without-persistent", {
      weightStorage: "fp16-storage",
    });
    const int8Message = await runFailure("phase6-kernel-int8-storage", {
      usePersistentWeightBuffers: true,
      weightStorage: "int8-dequant-experimental",
    });
    worker.terminate();
    if (!baseline.ok) return baseline;
    if (!explicitPooled.ok) return explicitPooled;
    if (!fp16.ok) return fp16;
    return {
      ok: true as const,
      baselineLoss: baseline.loss,
      explicitPooledLossDelta: Math.abs(explicitPooled.loss - baseline.loss),
      explicitPooledOutputMeanDelta: meanAbsDelta(explicitPooled.finalValues, baseline.finalValues),
      fp16Loss: fp16.loss,
      fp16LossDelta: Math.abs(fp16.loss - baseline.loss),
      fp16OutputMeanDelta: meanAbsDelta(fp16.finalValues, baseline.finalValues),
      tiledMessage,
      spatialBackwardMessage,
      fp16WithoutPersistentMessage,
      int8Message,
    };
  });
  test.skip(
    !result.ok && result.reason === "missing-fixtures",
    "Missing phase3 full-pass fixtures. Run python-reference/export_vgg19_phase3_full_pass.py first.",
  );
  if (!result.ok && result.reason !== "missing-fixtures")
    throw new Error(result.reason);
  if (!result.ok) return;
  expect(Number.isFinite(result.baselineLoss)).toBeTruthy();
  expect(Number.isFinite(result.fp16Loss)).toBeTruthy();
  expect(result.explicitPooledLossDelta).toBeLessThan(1e-6);
  expect(result.explicitPooledOutputMeanDelta).toBeLessThan(1e-6);
  expect(result.fp16LossDelta).toBeLessThan(0.05);
  expect(result.fp16OutputMeanDelta).toBeLessThan(0.02);
  expect(result.tiledMessage).toContain("convForwardKernel=tiled-spatial");
  expect(result.spatialBackwardMessage).toContain("convBackwardInputKernel=spatial-vec4");
  expect(result.fp16WithoutPersistentMessage).toContain("requires usePersistentWeightBuffers=true");
  expect(result.int8Message).toContain("weightStorage=int8-dequant-experimental");
});
