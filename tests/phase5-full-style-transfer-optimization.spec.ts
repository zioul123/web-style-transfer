import { expect, test } from "@playwright/test";
import type { WorkerRequest } from "../src/types";
import { gotoStableApp } from "./helpers/appPage";

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

  await page.getByRole("button", { name: "Show options" }).click();
  const modelPackSelect = page.getByLabel("Model pack");
  await expect(modelPackSelect).toBeVisible();
  const initialPack = await modelPackSelect.inputValue();
  await expect
    .poll(async () => {
      const urls = await page.evaluate(() => {
        const state = window as Window & { __packFetches?: string[] };
        return state.__packFetches ?? [];
      });
      return urls.some((url) =>
        url.includes(`/vgg19-models/${initialPack}/manifest.json`),
      );
    })
    .toBeTruthy();

  await modelPackSelect.selectOption("fp32");
  await expect(modelPackSelect).toHaveValue("fp32");
  await expect
    .poll(async () => {
      const urls = await page.evaluate(() => {
        const state = window as Window & { __packFetches?: string[] };
        return state.__packFetches ?? [];
      });
      return urls.some((url) =>
        url.includes("/vgg19-models/fp32/manifest.json"),
      );
    })
    .toBeTruthy();
});

test("phase 5 full style transfer endpoint returns losses", async ({
  page,
}) => {
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

test("phase 5 full style transfer supports rectangular style shape", async ({
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

test("phase 5 run-style-transfer first-step gradient matches pytorch oracle", async ({
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

test("phase 5 weighted first-step gradient composes content and style gradients", async ({
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

test("phase 5 style transfer supports adam and lbfgs", async ({ page }) => {
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
    const { createGpuVectorOps } =
      await import("/src/ml/worker/pipelines/optimization/input-optimizer/gpuVectorOps.ts");
    const { uploadToOwnedBuffer, releaseOwnedBuffer } =
      await import("/src/ml/worker/runtime/bufferKernels.ts");

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
    const { withBrowserStyleTransferWorkerClient } =
      await import("/tests/helpers/browserWorkerClient.ts");
    const input = Array.from(
      { length: 3 * 8 * 8 },
      (_, i) => Math.sin(i * 0.13) * 0.5,
    );
    const weight = Array.from(
      { length: 4 * 3 * 3 * 3 },
      (_, i) => Math.cos(i * 0.07) * 0.1,
    );
    const bias = [0.01, -0.02, 0.03, -0.04];
    return withBrowserStyleTransferWorkerClient(async (workerClient) => {
      await workerClient.initWebGpu("phase5-fused-init");
      const conv = await workerClient.ask({
        type: "tensor-op",
        id: "phase5-fused-conv",
        op: "conv2d-forward",
        input: { shape: [1, 3, 8, 8], values: input },
        weight: { shape: [4, 3, 3, 3], values: weight },
        bias,
      });
      if (conv.type !== "tensor-op-result" || !conv.ok || !("values" in conv))
        return { ok: false as const, reason: "conv-failed" as const };
      const relu = await workerClient.ask({
        type: "tensor-op",
        id: "phase5-fused-relu",
        op: "relu-forward",
        input: { shape: [1, 4, 8, 8], values: conv.values },
      });
      const fused = await workerClient.ask({
        type: "tensor-op",
        id: "phase5-fused-fused",
        op: "conv2d-relu-forward",
        input: { shape: [1, 3, 8, 8], values: input },
        weight: { shape: [4, 3, 3, 3], values: weight },
        bias,
      });
      if (relu.type !== "tensor-op-result" || !relu.ok || !("values" in relu))
        return { ok: false as const, reason: "relu-failed" as const };
      if (
        fused.type !== "tensor-op-result" ||
        !fused.ok ||
        !("values" in fused)
      )
        return { ok: false as const, reason: "fused-failed" as const };
      let maxDiff = 0;
      for (let i = 0; i < relu.values.length; i += 1)
        maxDiff = Math.max(maxDiff, Math.abs(relu.values[i] - fused.values[i]));
      return { ok: true as const, maxDiff };
    });
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
    const { loadFullPassArtifacts } =
      await import("/tests/helpers/fullPassArtifacts.ts");
    const artifactsResult = await loadFullPassArtifacts();
    if (!artifactsResult.ok) return artifactsResult;
    const { fixture, weights } = artifactsResult.artifacts;
    const { askRunStyleTransfer, withBrowserStyleTransferWorkerClient } =
      await import("/tests/helpers/browserWorkerClient.ts");
    const runVariant = async (
      workerClient: Parameters<typeof askRunStyleTransfer>[0],
      id: string,
      gramKernel: "parallel-dot" | "symmetric-parallel-dot",
      styleBackward: "two-pass" | "fused-from-gram-diff",
    ): Promise<{ ok: true; loss: number } | { ok: false; reason: string }> => {
      const out = await askRunStyleTransfer(workerClient, {
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
      if (!out.ok) {
        return {
          ok: false,
          reason: out.reason === "worker-failed" ? out.message : out.reason,
        };
      }
      if (out.response.losses.length === 0) {
        return { ok: false, reason: "missing-loss" };
      }
      return { ok: true, loss: out.response.losses[0] };
    };

    const [variantA, variantB] = await withBrowserStyleTransferWorkerClient(
      async (workerClient) => {
        await workerClient.initWebGpu("phase6-kernel-variant-init");
        return [
          await runVariant(
            workerClient,
            "phase6-kernel-variant-a",
            "parallel-dot",
            "two-pass",
          ),
          await runVariant(
            workerClient,
            "phase6-kernel-variant-b",
            "symmetric-parallel-dot",
            "fused-from-gram-diff",
          ),
        ] as const;
      },
    );
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
    result.ok ? "" : result.message,
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
    const { loadFullPassArtifacts } =
      await import("/tests/helpers/fullPassArtifacts.ts");
    const artifactsResult = await loadFullPassArtifacts();
    if (!artifactsResult.ok) return artifactsResult;
    const { fixture, weights } = artifactsResult.artifacts;
    const {
      askRunStyleTransfer,
      validateRunStyleTransferResult,
      withBrowserStyleTransferWorkerClient,
    } = await import("/tests/helpers/browserWorkerClient.ts");
    type RunStyleRequest = Extract<
      WorkerRequest,
      { type: "run-style-transfer" }
    >;
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
      workerClient: Parameters<typeof askRunStyleTransfer>[0],
      id: string,
      kernelFlags?: RunStyleRequest["kernelFlags"],
    ): Promise<
      | { ok: true; loss: number; finalValues: number[] }
      | { ok: false; reason: string }
    > => {
      const out = await askRunStyleTransfer(workerClient, {
        ...baseRequest(id),
        kernelFlags,
      });
      if (!out.ok) {
        return {
          ok: false,
          reason: out.reason === "worker-failed" ? out.message : out.reason,
        };
      }
      return {
        ok: true,
        loss: out.response.losses[0] ?? Number.NaN,
        finalValues: out.response.finalValues,
      };
    };
    const runFailure = async (
      workerClient: Parameters<typeof askRunStyleTransfer>[0],
      id: string,
      kernelFlags: RunStyleRequest["kernelFlags"],
    ): Promise<string> => {
      const out = validateRunStyleTransferResult(
        await workerClient.ask({ ...baseRequest(id), kernelFlags }),
      );
      if (out.ok) return "unexpected-success";
      return out.reason === "worker-failed" ? out.message : out.reason;
    };
    const meanAbsDelta = (
      a: readonly number[],
      b: readonly number[],
    ): number => {
      if (a.length !== b.length) return Number.POSITIVE_INFINITY;
      let total = 0;
      for (let i = 0; i < a.length; i += 1) total += Math.abs(a[i] - b[i]);
      return total / a.length;
    };

    const {
      baseline,
      explicitPooled,
      fp16,
      tiledMessage,
      spatialBackwardMessage,
      fp16WithoutPersistentMessage,
      int8Message,
    } = await withBrowserStyleTransferWorkerClient(async (workerClient) => {
      await workerClient.initWebGpu("phase6-kernel-storage-init");
      return {
        baseline: await runVariant(
          workerClient,
          "phase6-kernel-baseline-default",
        ),
        explicitPooled: await runVariant(
          workerClient,
          "phase6-kernel-explicit-pooled",
          {
            useStepBufferPool: true,
          },
        ),
        fp16: await runVariant(workerClient, "phase6-kernel-fp16-storage", {
          usePersistentWeightBuffers: true,
          weightStorage: "fp16-storage",
        }),
        tiledMessage: await runFailure(
          workerClient,
          "phase6-kernel-unsupported-tiled",
          {
            convForwardKernel: "tiled-spatial",
          },
        ),
        spatialBackwardMessage: await runFailure(
          workerClient,
          "phase6-kernel-unsupported-spatial-backward",
          {
            convBackwardInputKernel: "spatial-vec4",
          },
        ),
        fp16WithoutPersistentMessage: await runFailure(
          workerClient,
          "phase6-kernel-fp16-without-persistent",
          {
            weightStorage: "fp16-storage",
          },
        ),
        int8Message: await runFailure(
          workerClient,
          "phase6-kernel-int8-storage",
          {
            usePersistentWeightBuffers: true,
            weightStorage: "int8-dequant-experimental",
          },
        ),
      };
    });
    if (!baseline.ok) return baseline;
    if (!explicitPooled.ok) return explicitPooled;
    if (!fp16.ok) return fp16;
    return {
      ok: true as const,
      baselineLoss: baseline.loss,
      explicitPooledLossDelta: Math.abs(explicitPooled.loss - baseline.loss),
      explicitPooledOutputMeanDelta: meanAbsDelta(
        explicitPooled.finalValues,
        baseline.finalValues,
      ),
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
    result.ok ? "" : result.message,
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
  expect(result.spatialBackwardMessage).toContain(
    "convBackwardInputKernel=spatial-vec4",
  );
  expect(result.fp16WithoutPersistentMessage).toContain(
    "requires usePersistentWeightBuffers=true",
  );
  expect(result.int8Message).toContain(
    "weightStorage=int8-dequant-experimental",
  );
});
