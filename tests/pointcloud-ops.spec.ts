import { expect, test } from "@playwright/test";
import { POINT_CLOUD_CONV_DEFAULT_INDEX_MAP } from "../src/ml/pointcloud/metadata";
import {
  cpuPointCloudConvBackwardFeatures,
  cpuPointCloudConvForward,
  cpuPointFeatureNormalizeBackward,
  cpuPointFeatureNormalizeForward,
  cpuPointwiseExpBackward,
  cpuPointwiseExpForward,
  cpuSurfacePoolBackward,
  cpuSurfacePoolForward,
} from "../src/ml/ops/pointcloud";
import {
  createFeatureMatrix,
  createPointCloudKnn,
  createSurfacePoolMap,
  createTensor,
} from "../src/ml/tensor";
import type {
  WorkerFeatureMatrix,
  WorkerPointCloudKnn,
  WorkerSurfacePoolMap,
  WorkerTensor,
} from "../src/types";
import { gotoStableApp } from "./helpers/appPage";
import { expectTensorCloseTo } from "./helpers/tensorAssertions";
import {
  createStyleTransferWorkerClient,
  expectWebGpuInitOk,
  expectWorkerTensorVectorResponse,
} from "./helpers/workerClient";

const makeFeatureMatrix = (
  pointCount: number,
  channelCount: number,
  values: readonly number[],
): WorkerFeatureMatrix => ({
  pointCount,
  channelCount,
  values: [...values],
});

const makeKnn = (
  sampleCount: number,
  neighborCount: number,
  indices: readonly number[],
  weights: readonly number[],
): WorkerPointCloudKnn => ({
  sampleCount,
  kernelPointCount: 9,
  neighborCount,
  indices: [...indices],
  weights: [...weights],
});

const makeSurfacePoolMap = (
  inputPointCount: number,
  outputPointCount: number,
  mapping: readonly number[],
): WorkerSurfacePoolMap => ({
  inputPointCount,
  outputPointCount,
  mapping: [...mapping],
});

test("point-cloud exp worker ops match CPU references", async ({ page }) => {
  await gotoStableApp(page);
  const workerClient = await createStyleTransferWorkerClient(page);

  try {
    const input = makeFeatureMatrix(2, 3, [-1, 0, 1, 2, -2, 0.5]);
    const gradOut = makeFeatureMatrix(2, 3, [1, -2, 0.25, 3, -1, 0.5]);
    const inputCpu = createFeatureMatrix(
      input.pointCount,
      input.channelCount,
      input.values,
    );
    const gradOutCpu = createFeatureMatrix(
      gradOut.pointCount,
      gradOut.channelCount,
      gradOut.values,
    );

    expectWebGpuInitOk(await workerClient.initWebGpu());

    const forward = await workerClient.ask({
      type: "tensor-op",
      id: "pc-exp-forward",
      op: "exp-forward",
      input,
    });
    const backward = await workerClient.ask({
      type: "tensor-op",
      id: "pc-exp-backward",
      op: "exp-backward",
      input,
      gradOut,
    });

    expectTensorCloseTo(
      expectWorkerTensorVectorResponse(forward).values,
      Array.from(cpuPointwiseExpForward(inputCpu).values),
      5,
    );
    expectTensorCloseTo(
      expectWorkerTensorVectorResponse(backward).values,
      Array.from(cpuPointwiseExpBackward(inputCpu, gradOutCpu).values),
      5,
    );
  } finally {
    await workerClient.dispose();
  }
});

test("point-feature normalize worker ops match CPU references", async ({
  page,
}) => {
  await gotoStableApp(page);
  const workerClient = await createStyleTransferWorkerClient(page);

  try {
    const input = makeFeatureMatrix(
      5,
      3,
      [-0.5, 0, 1.5, 2, -2, 0.25, 1.25, 3, -1, -3, 0.75, 2.5, 4, -4, 0.5],
    );
    const gradOut = makeFeatureMatrix(
      5,
      3,
      [0.25, -1, 2, 1.5, 0, -0.75, -2, 3, 0.5, 1, -1.25, 2.25, 0.75, 1.25, -3],
    );
    const mean = [0.5, -1, 2];
    const std = [0.5, 2, 4];
    const inputCpu = createFeatureMatrix(
      input.pointCount,
      input.channelCount,
      input.values,
    );
    const gradOutCpu = createFeatureMatrix(
      gradOut.pointCount,
      gradOut.channelCount,
      gradOut.values,
    );

    expectWebGpuInitOk(await workerClient.initWebGpu());

    const forward = await workerClient.ask({
      type: "tensor-op",
      id: "pc-normalize-forward",
      op: "point-feature-normalize-forward",
      input,
      mean,
      std,
    });
    const backward = await workerClient.ask({
      type: "tensor-op",
      id: "pc-normalize-backward",
      op: "point-feature-normalize-backward",
      gradOut,
      std,
    });

    expectTensorCloseTo(
      expectWorkerTensorVectorResponse(forward).values,
      Array.from(cpuPointFeatureNormalizeForward(inputCpu, mean, std).values),
      5,
    );
    expectTensorCloseTo(
      expectWorkerTensorVectorResponse(backward).values,
      Array.from(cpuPointFeatureNormalizeBackward(gradOutCpu, std).values),
      5,
    );
  } finally {
    await workerClient.dispose();
  }
});

test("point-cloud conv CPU ref uses default VGG kernel reorder", () => {
  const input = createFeatureMatrix(9, 1, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const knn = createPointCloudKnn(
    1,
    9,
    1,
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
  );
  const weight = createTensor(
    [1, 1, 3, 3],
    [10, 20, 30, 40, 50, 60, 70, 80, 90],
  );

  const output = cpuPointCloudConvForward(input, knn, weight, [0]);
  const expected = POINT_CLOUD_CONV_DEFAULT_INDEX_MAP.reduce(
    (sum, mappedKernel, kernel) => sum + (kernel + 1) * (mappedKernel + 1) * 10,
    0,
  );

  expect(output.values[0]).toBe(expected);
});

test("point-cloud conv worker ops match CPU references", async ({ page }) => {
  await gotoStableApp(page);
  const workerClient = await createStyleTransferWorkerClient(page);

  try {
    const input = makeFeatureMatrix(5, 2, [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]);
    const sampleCount = 2;
    const neighborCount = 2;
    const indices: number[] = [];
    const weights: number[] = [];
    for (let sample = 0; sample < sampleCount; sample += 1) {
      for (let kernel = 0; kernel < 9; kernel += 1) {
        if (sample === 0) {
          indices.push(kernel % 5, (kernel + 1) % 5);
          weights.push(0.25, 0.75);
        } else {
          indices.push(1, kernel % 5);
          weights.push(0.6, 0.4);
        }
      }
    }
    const knn = makeKnn(sampleCount, neighborCount, indices, weights);
    const weightValues = Array.from(
      { length: 2 * 2 * 3 * 3 },
      (_unused: unknown, index: number): number => ((index % 13) - 6) / 5,
    );
    const weight: WorkerTensor = {
      shape: [2, 2, 3, 3],
      values: weightValues,
    };
    const bias = [0.25, -0.5];
    const gradOut = makeFeatureMatrix(sampleCount, 2, [1, -2, 0.5, 3]);

    const inputCpu = createFeatureMatrix(
      input.pointCount,
      input.channelCount,
      input.values,
    );
    const knnCpu = createPointCloudKnn(
      knn.sampleCount,
      knn.kernelPointCount,
      knn.neighborCount,
      knn.indices,
      knn.weights,
    );
    const weightCpu = createTensor(weight.shape, weight.values);
    const gradOutCpu = createFeatureMatrix(
      gradOut.pointCount,
      gradOut.channelCount,
      gradOut.values,
    );

    expectWebGpuInitOk(await workerClient.initWebGpu());

    const forward = await workerClient.ask({
      type: "tensor-op",
      id: "pc-conv-forward",
      op: "pc-conv-forward",
      input,
      knn,
      weight,
      bias,
    });
    const backward = await workerClient.ask({
      type: "tensor-op",
      id: "pc-conv-backward-features",
      op: "pc-conv-backward-features",
      inputShape: {
        pointCount: input.pointCount,
        channelCount: input.channelCount,
      },
      gradOut,
      knn,
      weight,
    });

    expectTensorCloseTo(
      expectWorkerTensorVectorResponse(forward).values,
      Array.from(
        cpuPointCloudConvForward(inputCpu, knnCpu, weightCpu, bias).values,
      ),
      4,
    );
    expectTensorCloseTo(
      expectWorkerTensorVectorResponse(backward).values,
      Array.from(
        cpuPointCloudConvBackwardFeatures(
          { pointCount: input.pointCount, channelCount: input.channelCount },
          gradOutCpu,
          knnCpu,
          weightCpu,
        ).values,
      ),
      4,
    );
  } finally {
    await workerClient.dispose();
  }
});

test("optimized point-cloud conv worker kernels match CPU references", async ({
  page,
}) => {
  await gotoStableApp(page);
  const workerClient = await createStyleTransferWorkerClient(page);

  try {
    const input = makeFeatureMatrix(
      6,
      3,
      [1, -1, 0.5, 2, -2, 1.5, 3, -3, 2.5, 4, -4, 3.5, 5, -5, 4.5, 6, -6, 5.5],
    );
    const weightValues = Array.from(
      { length: 4 * 3 * 3 * 3 },
      (_unused: unknown, index: number): number => ((index % 17) - 8) / 7,
    );
    const weight: WorkerTensor = {
      shape: [4, 3, 3, 3],
      values: weightValues,
    };
    const bias = [0.25, -0.5, 0.75, -1];
    const kernelIndexMap = [8, 5, 2, 7, 4, 1, 6, 3, 0];
    const sampleCount = 5;
    const maxIntermediateBytes = 2 * 9 * input.channelCount * 4;

    expectWebGpuInitOk(await workerClient.initWebGpu());

    for (const neighborCount of [1, 3]) {
      const indices: number[] = [];
      const weights: number[] = [];
      for (let sample = 0; sample < sampleCount; sample += 1) {
        for (let kernel = 0; kernel < 9; kernel += 1) {
          const first = (sample + kernel) % input.pointCount;
          if (neighborCount === 1) {
            indices.push(first);
            weights.push(1);
          } else {
            indices.push(first, first, (sample * 2 + kernel + 1) % 6);
            weights.push(
              kernel % 2 === 0 ? 0.5 : 1,
              kernel % 2 === 0 ? 0.25 : 0,
              kernel % 2 === 0 ? 0.25 : 0,
            );
          }
        }
      }
      const knn = makeKnn(sampleCount, neighborCount, indices, weights);
      const gradOut = makeFeatureMatrix(
        sampleCount,
        4,
        Array.from(
          { length: sampleCount * 4 },
          (_unused: unknown, index: number): number => ((index % 11) - 5) / 3,
        ),
      );

      const inputCpu = createFeatureMatrix(
        input.pointCount,
        input.channelCount,
        input.values,
      );
      const knnCpu = createPointCloudKnn(
        knn.sampleCount,
        knn.kernelPointCount,
        knn.neighborCount,
        knn.indices,
        knn.weights,
      );
      const weightCpu = createTensor(weight.shape, weight.values);
      const gradOutCpu = createFeatureMatrix(
        gradOut.pointCount,
        gradOut.channelCount,
        gradOut.values,
      );

      const forward = await workerClient.ask({
        type: "tensor-op",
        id: `pc-conv-forward-optimized-${neighborCount}`,
        op: "pc-conv-forward",
        input,
        knn,
        weight,
        bias,
        kernelIndexMap,
        pcConvForwardKernel: "interpolate-project",
        pcConvMaxIntermediateBytes: maxIntermediateBytes,
      });
      const backward = await workerClient.ask({
        type: "tensor-op",
        id: `pc-conv-backward-optimized-${neighborCount}`,
        op: "pc-conv-backward-features",
        inputShape: {
          pointCount: input.pointCount,
          channelCount: input.channelCount,
        },
        gradOut,
        knn,
        weight,
        kernelIndexMap,
        pcConvBackwardFeaturesKernel: "reverse-adjacency",
        pcConvMaxIntermediateBytes: maxIntermediateBytes,
      });

      expectTensorCloseTo(
        expectWorkerTensorVectorResponse(forward).values,
        Array.from(
          cpuPointCloudConvForward(
            inputCpu,
            knnCpu,
            weightCpu,
            bias,
            kernelIndexMap,
          ).values,
        ),
        4,
      );
      expectTensorCloseTo(
        expectWorkerTensorVectorResponse(backward).values,
        Array.from(
          cpuPointCloudConvBackwardFeatures(
            { pointCount: input.pointCount, channelCount: input.channelCount },
            gradOutCpu,
            knnCpu,
            weightCpu,
            kernelIndexMap,
          ).values,
        ),
        4,
      );
    }
  } finally {
    await workerClient.dispose();
  }
});

test("optimized point-cloud conv backward splits reverse gather across dispatch limit", async ({
  page,
}) => {
  test.setTimeout(120000);
  await gotoStableApp(page);
  const maxComputeWorkgroupsPerDimension = await page.evaluate(async () => {
    const adapter = await navigator.gpu?.requestAdapter();
    return adapter?.limits.maxComputeWorkgroupsPerDimension ?? 65_535;
  });
  const maxDispatchElements = maxComputeWorkgroupsPerDimension * 64;
  test.skip(
    maxDispatchElements > 8_000_000,
    "adapter dispatch limit is too large for a bounded readback regression",
  );

  const workerClient = await createStyleTransferWorkerClient(page);

  try {
    const pointCount = maxDispatchElements + 1;
    const inputShape = { pointCount, channelCount: 1 };
    const knn = makeKnn(
      1,
      1,
      Array.from({ length: 9 }, () => 0),
      [1, 1, 1, 1, 1, 1, 1, 1, 1],
    );
    const weight: WorkerTensor = {
      shape: [1, 1, 3, 3],
      values: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    };
    const gradOut = makeFeatureMatrix(1, 1, [1]);

    expectWebGpuInitOk(await workerClient.initWebGpu());

    const backward = await workerClient.ask({
      type: "tensor-op",
      id: "pc-conv-backward-optimized-dispatch-split",
      op: "pc-conv-backward-features",
      inputShape,
      gradOut,
      knn,
      weight,
      pcConvBackwardFeaturesKernel: "reverse-adjacency",
      pcConvMaxIntermediateBytes: 1024,
    });
    const values = expectWorkerTensorVectorResponse(backward).values;

    expect(values).toHaveLength(pointCount);
    expect(values[0]).toBeCloseTo(45, 5);
    expect(values[values.length - 1]).toBe(0);
  } finally {
    await workerClient.dispose();
  }
});

test("surface pool worker ops match tie and orphan CPU references", async ({
  page,
}) => {
  await gotoStableApp(page);
  const workerClient = await createStyleTransferWorkerClient(page);

  try {
    const input = makeFeatureMatrix(5, 2, [1, 2, 3, 5, 3, 4, 0, 7, 2, 7]);
    const pool = makeSurfacePoolMap(5, 3, [-1, 0, 0, 1, 1]);
    const gradOut = makeFeatureMatrix(3, 2, [1, 2, 3, 4, 5, 6]);

    const inputCpu = createFeatureMatrix(
      input.pointCount,
      input.channelCount,
      input.values,
    );
    const gradOutCpu = createFeatureMatrix(
      gradOut.pointCount,
      gradOut.channelCount,
      gradOut.values,
    );
    const poolCpu = createSurfacePoolMap(
      pool.inputPointCount,
      pool.outputPointCount,
      pool.mapping,
    );
    const expectedForward = [3, 5, 2, 7, 2, 5.75];
    const expectedBackward = [0, 0, 1.75, 3.5, 1.75, 1.5, 1.25, 3.5, 4.25, 3.5];
    const cpuForward = cpuSurfacePoolForward(inputCpu, poolCpu);
    const cpuBackward = cpuSurfacePoolBackward(inputCpu, gradOutCpu, poolCpu);

    expectTensorCloseTo(Array.from(cpuForward.values), expectedForward, 5);
    expectTensorCloseTo(Array.from(cpuBackward.values), expectedBackward, 5);

    expectWebGpuInitOk(await workerClient.initWebGpu());

    const forward = await workerClient.ask({
      type: "tensor-op",
      id: "surface-pool-forward",
      op: "surface-pool-forward",
      input,
      pool,
    });
    const backward = await workerClient.ask({
      type: "tensor-op",
      id: "surface-pool-backward",
      op: "surface-pool-backward",
      input,
      gradOut,
      pool,
    });

    expectTensorCloseTo(
      expectWorkerTensorVectorResponse(forward).values,
      Array.from(cpuForward.values),
      5,
    );
    expectTensorCloseTo(
      expectWorkerTensorVectorResponse(backward).values,
      Array.from(cpuBackward.values),
      5,
    );
  } finally {
    await workerClient.dispose();
  }
});
