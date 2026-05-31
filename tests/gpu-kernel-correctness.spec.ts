import { expect, test } from "@playwright/test";
import { gotoStableApp } from "./helpers/appPage";

test("gpu vector reductions match cpu references", async ({ page }) => {
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

test("fused conv+relu op matches separate ops", async ({ page }) => {
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
