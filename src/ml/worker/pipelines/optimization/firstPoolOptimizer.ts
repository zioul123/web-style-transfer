/// <reference lib="webworker" />

import type { WorkerRequest } from "../../../../types";
import { createTensor } from "../../../index";
import { runReluForward, runReluBackward } from "../../ops/relu/relu.run";
import {
  runNormalizeBackward,
  runNormalizeForward,
} from "../../ops/normalization/normalize.run";
import {
  runMaxPool2dBackward,
  runMaxPool2dForward,
} from "../../ops/pooling/maxpool.run";
import {
  runConv2dBackwardInput,
  runConv2dForward,
} from "../../ops/convolution/conv2d.run";
import { runGramMatrix } from "../../ops/gram/gram.run";
import { runMse } from "../../ops/loss/mse.run";
import { runContentLossBackward } from "../../ops/loss/contentLoss.run";
import { runStyleLossBackward } from "../../ops/loss/styleLoss.run";
import {
  acquireReusableBuffer,
  releaseReusableBuffer,
} from "../../runtime/bufferPool";
import {
  BUFFER_USAGE_MAP_READ_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_SRC,
  BUFFER_USAGE_UNIFORM_COPY_DST,
  MAP_MODE_READ,
} from "../../runtime/gpuFlags";
import { runBinaryOp, runScalarBinaryOp } from "../../runtime/shaderRunner";
import { getGpuDevice } from "../../runtime/deviceState";
import { runUnary } from "../../runtime/computeContext";

type TensorShape4D = readonly [number, number, number, number];

type FirstPoolPersistentContext = {
  conv1Weight: ReturnType<typeof createTensor>;
  conv2Weight: ReturnType<typeof createTensor>;
  conv3Weight: ReturnType<typeof createTensor>;
  contentRelu2: Float32Array;
  styleRelu1: Float32Array;
  styleRelu3: Float32Array;
  styleGram1: Float32Array;
  styleGram3: Float32Array;
};

type FirstPoolStepTemporaries = {
  norm: Float32Array;
  conv1: Float32Array;
  relu1: Float32Array;
  conv2: Float32Array;
  relu2: Float32Array;
  pool: Float32Array;
  conv3: Float32Array;
  relu3: Float32Array;
  gInput: Float32Array;
};

type FirstPoolComputeContext = {
  persistent: FirstPoolPersistentContext;
  stepTemps: Partial<FirstPoolStepTemporaries>;
  acquireTempBuffer: (
    shape: TensorShape4D,
    usage: number,
    role: string,
  ) => GPUBuffer;
  releaseTempBuffer: (shape: TensorShape4D, usage: number, role: string) => void;
};

const IS_DEV = import.meta.env.DEV;
const elementCount = (shape: TensorShape4D): number =>
  shape[0] * shape[1] * shape[2] * shape[3];

const assertShape = (
  label: string,
  values: Float32Array,
  shape: TensorShape4D,
): void => {
  if (!IS_DEV) return;
  if (values.length !== elementCount(shape)) {
    throw new Error(
      `[firstPoolOptimizer] ${label} length mismatch. expected=${elementCount(shape)} actual=${values.length}`,
    );
  }
};

export const runFirstPoolOptimizer = async (
  payload: Extract<WorkerRequest, { type: "run-first-pool-optimizer" }>,
): Promise<{ losses: number[]; finalValues: number[] }> => {
  const device = getGpuDevice();
  if (device === null) throw new Error("WebGPU device is not initialized");
  const localBufferByKey: Map<string, GPUBuffer> = new Map();
  const localTempKey = (shape: TensorShape4D, usage: number, role: string): string =>
    `${shape.join("x")}:${usage}:${role}`;

  const acquireTempBuffer = (
    shape: TensorShape4D,
    usage: number,
    role: string,
  ): GPUBuffer => {
    const key = localTempKey(shape, usage, role);
    const existing = localBufferByKey.get(key);
    if (existing !== undefined) return existing;
    const size = elementCount(shape) * Float32Array.BYTES_PER_ELEMENT;
    const buffer = acquireReusableBuffer(device, size, usage);
    localBufferByKey.set(key, buffer);
    if (IS_DEV && size <= 0) {
      throw new Error(`[firstPoolOptimizer] invalid temp buffer size for ${role}`);
    }
    return buffer;
  };

  const releaseTempBuffer = (
    shape: TensorShape4D,
    usage: number,
    role: string,
  ): void => {
    const key = localTempKey(shape, usage, role);
    const buffer = localBufferByKey.get(key);
    if (buffer === undefined) return;
    localBufferByKey.delete(key);
    const size = elementCount(shape) * Float32Array.BYTES_PER_ELEMENT;
    releaseReusableBuffer(size, usage, buffer);
  };

  const convForward = async (
    inputValues: Float32Array,
    inputShape: TensorShape4D,
    weightValues: Float32Array,
    weightShape: TensorShape4D,
    bias: number[],
  ): Promise<Float32Array> =>
    await runConv2dForward(
      device,
      runUnary,
      inputValues,
      inputShape,
      weightValues,
      weightShape,
      bias,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    );

  const conv1Weight = createTensor(
    payload.conv1Weight.shape,
    payload.conv1Weight.values,
  );
  const conv2Weight = createTensor(
    payload.conv2Weight.shape,
    payload.conv2Weight.values,
  );
  const conv3Weight = createTensor(
    payload.conv3Weight.shape,
    payload.conv3Weight.values,
  );

  const contentNorm = await runNormalizeForward(
    device,
    runUnary,
    new Float32Array(payload.contentImageValues),
    payload.inputShape,
    payload.mean,
    payload.std,
    BUFFER_USAGE_STORAGE_COPY_DST,
    BUFFER_USAGE_UNIFORM_COPY_DST,
  );
  const styleNorm = await runNormalizeForward(
    device,
    runUnary,
    new Float32Array(payload.styleImageValues),
    payload.inputShape,
    payload.mean,
    payload.std,
    BUFFER_USAGE_STORAGE_COPY_DST,
    BUFFER_USAGE_UNIFORM_COPY_DST,
  );

  const styleConv1 = await convForward(
    styleNorm,
    payload.inputShape,
    conv1Weight.values,
    conv1Weight.shape,
    payload.conv1Bias,
  );
  const styleRelu1 = await runReluForward(runUnary, styleConv1);
  const styleConv2 = await convForward(
    styleRelu1,
    [1, 64, 16, 16],
    conv2Weight.values,
    conv2Weight.shape,
    payload.conv2Bias,
  );
  const styleRelu2 = await runReluForward(runUnary, styleConv2);
  const stylePool = await runMaxPool2dForward(
    device,
    runUnary,
    styleRelu2,
    [1, 64, 16, 16],
    BUFFER_USAGE_UNIFORM_COPY_DST,
  );
  const styleConv3 = await convForward(
    stylePool,
    [1, 64, 8, 8],
    conv3Weight.values,
    conv3Weight.shape,
    payload.conv3Bias,
  );
  const styleRelu3 = await runReluForward(runUnary, styleConv3);

  const contentConv1 = await convForward(
    contentNorm,
    payload.inputShape,
    conv1Weight.values,
    conv1Weight.shape,
    payload.conv1Bias,
  );
  const contentRelu1 = await runReluForward(runUnary, contentConv1);
  assertShape("contentRelu1", contentRelu1, [1, 64, 16, 16]);
  const contentConv2 = await convForward(
    contentRelu1,
    [1, 64, 16, 16],
    conv2Weight.values,
    conv2Weight.shape,
    payload.conv2Bias,
  );
  const contentRelu2 = await runReluForward(runUnary, contentConv2);
  const styleGram1 = await runGramMatrix(
    device,
    runUnary,
    styleRelu1,
    [1, 64, 16, 16],
    BUFFER_USAGE_UNIFORM_COPY_DST,
  );
  const styleGram3 = await runGramMatrix(
    device,
    runUnary,
    styleRelu3,
    [1, 128, 8, 8],
    BUFFER_USAGE_UNIFORM_COPY_DST,
  );

  const context: FirstPoolComputeContext = {
    persistent: {
      conv1Weight,
      conv2Weight,
      conv3Weight,
      contentRelu2,
      styleRelu1,
      styleRelu3,
      styleGram1,
      styleGram3,
    },
    stepTemps: {},
    acquireTempBuffer,
    releaseTempBuffer,
  };

  let inputValues = new Float32Array(payload.initialInputValues);
  const losses: number[] = [];
  try {
    for (let step = 0; step < payload.steps; step += 1) {
    const norm = await runNormalizeForward(
      device,
      runUnary,
      inputValues,
      payload.inputShape,
      payload.mean,
      payload.std,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    );
    const conv1 = await convForward(
      norm,
      payload.inputShape,
      context.persistent.conv1Weight.values,
      context.persistent.conv1Weight.shape,
      payload.conv1Bias,
    );
    const relu1 = await runReluForward(runUnary, conv1);
    const conv2 = await convForward(
      relu1,
      [1, 64, 16, 16],
      context.persistent.conv2Weight.values,
      context.persistent.conv2Weight.shape,
      payload.conv2Bias,
    );
    const relu2 = await runReluForward(runUnary, conv2);
    const pool = await runMaxPool2dForward(
      device,
      runUnary,
      relu2,
      [1, 64, 16, 16],
      BUFFER_USAGE_UNIFORM_COPY_DST,
    );
    const conv3 = await convForward(
      pool,
      [1, 64, 8, 8],
      context.persistent.conv3Weight.values,
      context.persistent.conv3Weight.shape,
      payload.conv3Bias,
    );
    const relu3 = await runReluForward(runUnary, conv3);
    context.stepTemps = { norm, conv1, relu1, conv2, relu2, pool, conv3, relu3 };
    assertShape("relu1", relu1, [1, 64, 16, 16]);
    assertShape("relu2", relu2, [1, 64, 16, 16]);
    assertShape("relu3", relu3, [1, 128, 8, 8]);


    const styleLoss1 =
      (await runMse(
        device,
        acquireReusableBuffer,
        releaseReusableBuffer,
        await runGramMatrix(
          device,
          runUnary,
          relu1,
          [1, 64, 16, 16],
          BUFFER_USAGE_UNIFORM_COPY_DST,
        ),
        context.persistent.styleGram1,
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_STORAGE_COPY_SRC,
        BUFFER_USAGE_MAP_READ_COPY_DST,
        MAP_MODE_READ,
      )) * payload.styleWeightConv1;
    const styleLoss3 =
      (await runMse(
        device,
        acquireReusableBuffer,
        releaseReusableBuffer,
        await runGramMatrix(
          device,
          runUnary,
          relu3,
          [1, 128, 8, 8],
          BUFFER_USAGE_UNIFORM_COPY_DST,
        ),
        context.persistent.styleGram3,
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_STORAGE_COPY_SRC,
        BUFFER_USAGE_MAP_READ_COPY_DST,
        MAP_MODE_READ,
      )) * payload.styleWeightConv3;
    const contentLoss =
      (await runMse(
        device,
        acquireReusableBuffer,
        releaseReusableBuffer,
        relu2,
        context.persistent.contentRelu2,
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_STORAGE_COPY_SRC,
        BUFFER_USAGE_MAP_READ_COPY_DST,
        MAP_MODE_READ,
      )) * payload.contentWeight;
    losses.push(styleLoss1 + styleLoss3 + contentLoss);

    const gStyle1Relu = await runScalarBinaryOp(
      device,
      "mul",
      await runStyleLossBackward(
        device,
        runUnary,
        relu1,
        [1, 64, 16, 16],
        context.persistent.styleRelu1,
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_UNIFORM_COPY_DST,
      ),
      payload.styleWeightConv1,
      false,
    );
    const gStyle3Relu = await runScalarBinaryOp(
      device,
      "mul",
      await runStyleLossBackward(
        device,
        runUnary,
        relu3,
        [1, 128, 8, 8],
        context.persistent.styleRelu3,
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_UNIFORM_COPY_DST,
      ),
      payload.styleWeightConv3,
      false,
    );
    const gContentRelu = await runScalarBinaryOp(
      device,
      "mul",
      await runContentLossBackward(
        device,
        runUnary,
        relu2,
        context.persistent.contentRelu2,
        BUFFER_USAGE_STORAGE_COPY_DST,
      ),
      payload.contentWeight,
      false,
    );
    const gStyle3 = await runReluBackward(
      device,
      runUnary,
      conv3,
      gStyle3Relu,
    );
    const gContent = await runReluBackward(
      device,
      runUnary,
      conv2,
      gContentRelu,
    );
    const gStyle1 = await runReluBackward(
      device,
      runUnary,
      conv1,
      gStyle1Relu,
    );

    const gPool = await runConv2dBackwardInput(
      device,
      runUnary,
      [1, 64, 8, 8],
      gStyle3,
      context.persistent.conv3Weight.values,
      context.persistent.conv3Weight.shape,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    );
    const gRelu2 = await runMaxPool2dBackward(
      device,
      runUnary,
      relu2,
      [1, 64, 16, 16],
      gPool,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    );
    const gConv2FromPool = await runReluBackward(
      device,
      runUnary,
      conv2,
      gRelu2,
    );
    const gConv2Total = await runBinaryOp(
      device,
      "add",
      gConv2FromPool,
      gContent,
      "tensorTensor",
    );
    const gRelu1 = await runConv2dBackwardInput(
      device,
      runUnary,
      [1, 64, 16, 16],
      gConv2Total,
      context.persistent.conv2Weight.values,
      context.persistent.conv2Weight.shape,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    );
    const gConv1FromConv2 = await runReluBackward(
      device,
      runUnary,
      conv1,
      gRelu1,
    );
    const gConv1Total = await runBinaryOp(
      device,
      "add",
      gConv1FromConv2,
      gStyle1,
      "tensorTensor",
    );
    const gNorm = await runConv2dBackwardInput(
      device,
      runUnary,
      payload.inputShape,
      gConv1Total,
      context.persistent.conv1Weight.values,
      context.persistent.conv1Weight.shape,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    );
    const gInput = await runNormalizeBackward(
      device,
      runUnary,
      gNorm,
      payload.inputShape,
      payload.std,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    );

    const next = new Float32Array(inputValues.length);
    for (let i = 0; i < inputValues.length; i += 1)
      next[i] = Math.max(
        0,
        Math.min(1, inputValues[i] - payload.learningRate * gInput[i]),
      );
    inputValues = next;
    context.stepTemps.gInput = gInput;
    assertShape("gInput", gInput, payload.inputShape);
    }
    return { losses, finalValues: Array.from(inputValues) };
  } finally {
    for (const [key, buffer] of localBufferByKey.entries()) {
      const [shapeKey, usageToken] = key.split(":");
      const shapeParts = shapeKey.split("x").map((part) => Number(part));
      if (shapeParts.length !== 4 || shapeParts.some((part) => Number.isNaN(part))) continue;
      const shape: TensorShape4D = [
        shapeParts[0],
        shapeParts[1],
        shapeParts[2],
        shapeParts[3],
      ];
      const usage = Number(usageToken);
      if (Number.isNaN(usage)) continue;
      releaseReusableBuffer(
        elementCount(shape) * Float32Array.BYTES_PER_ELEMENT,
        usage,
        buffer,
      );
    }
    localBufferByKey.clear();
  }
};
