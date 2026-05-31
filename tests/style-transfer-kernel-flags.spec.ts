import { expect, test } from "@playwright/test";
import type { WorkerRequest } from "../src/types";
import { gotoStableApp } from "./helpers/appPage";

test("style transfer supports gram/style kernel variants", async ({ page }) => {
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

test("kernel flags preserve baseline semantics and support fp16-storage", async ({
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
