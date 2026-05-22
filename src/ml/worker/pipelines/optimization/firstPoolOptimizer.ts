/// <reference lib="webworker" />

import type { WorkerRequest } from "../../../../types";
import { createTensor } from "../../../index";
import { runReluBackward } from "../../ops/relu/relu.run";
import {
  runNormalizeBackward,
} from "../../ops/normalization/normalize.run";
import {
  runMaxPool2dBackward,
} from "../../ops/pooling/maxpool.run";
import {
  runConv2dBackwardInput,
} from "../../ops/convolution/conv2d.run";
import { runGramMatrix } from "../../ops/gram/gram.run";
import { runMseBuffer } from "../../ops/loss/mse.run";
import { runContentLossBackward } from "../../ops/loss/contentLoss.run";
import { runStyleLossBackward } from "../../ops/loss/styleLoss.run";
import {
  acquireReusableBuffer,
  releaseReusableBuffer,
} from "../../runtime/bufferPool";
import {
  readGpuBufferToArray,
  releaseOwnedBuffer,
  uploadToOwnedBuffer,
  type GpuBufferRef,
} from "../../runtime/bufferKernels";
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
import { runNormalizeForwardBuffer } from "../../ops/normalization/normalize.run";
import { runConv2dForwardBuffer } from "../../ops/convolution/conv2d.run";
import { runReluForwardBuffer } from "../../ops/relu/relu.run";
import { runMaxPool2dForwardBuffer } from "../../ops/pooling/maxpool.run";
import { runGramMatrixBuffer } from "../../ops/gram/gram.run";

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
  styleGram1Buffer: GpuBufferRef;
  styleGram3Buffer: GpuBufferRef;
  contentRelu2Buffer: GpuBufferRef;
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

const readGpuTensor = async (
  device: GPUDevice,
  bufferRef: GpuBufferRef,
  shape: TensorShape4D,
): Promise<Float32Array> =>
  await readGpuBufferToArray(device, bufferRef.buffer, elementCount(shape));

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

  const contentInputBuffer = uploadToOwnedBuffer(
    device,
    new Float32Array(payload.contentImageValues),
  );
  const contentNormBuffer = await runNormalizeForwardBuffer(
    contentInputBuffer,
    payload.inputShape,
    payload.mean,
    payload.std,
  );
  const styleInputBuffer = uploadToOwnedBuffer(
    device,
    new Float32Array(payload.styleImageValues),
  );
  const styleNormBuffer = await runNormalizeForwardBuffer(
    styleInputBuffer,
    payload.inputShape,
    payload.mean,
    payload.std,
  );
  const styleConv1Buffer = await runConv2dForwardBuffer(
    styleNormBuffer,
    payload.inputShape,
    conv1Weight.values,
    conv1Weight.shape,
    payload.conv1Bias,
  );
  const styleRelu1Buffer = await runReluForwardBuffer(
    styleConv1Buffer,
    elementCount([1, 64, 16, 16]),
  );
  const styleConv2Buffer = await runConv2dForwardBuffer(
    styleRelu1Buffer,
    [1, 64, 16, 16],
    conv2Weight.values,
    conv2Weight.shape,
    payload.conv2Bias,
  );
  const styleRelu2Buffer = await runReluForwardBuffer(
    styleConv2Buffer,
    elementCount([1, 64, 16, 16]),
  );
  const stylePoolBuffer = await runMaxPool2dForwardBuffer(styleRelu2Buffer, [1, 64, 16, 16]);
  const styleConv3Buffer = await runConv2dForwardBuffer(
    stylePoolBuffer,
    [1, 64, 8, 8],
    conv3Weight.values,
    conv3Weight.shape,
    payload.conv3Bias,
  );
  const styleRelu3Buffer = await runReluForwardBuffer(
    styleConv3Buffer,
    elementCount([1, 128, 8, 8]),
  );
  const styleRelu1 = await readGpuTensor(device, styleRelu1Buffer, [1, 64, 16, 16]);
  const styleRelu3 = await readGpuTensor(device, styleRelu3Buffer, [1, 128, 8, 8]);
  assertShape("styleRelu1", styleRelu1, [1, 64, 16, 16]);
  assertShape("styleRelu3", styleRelu3, [1, 128, 8, 8]);

  const contentConv1Buffer = await runConv2dForwardBuffer(
    contentNormBuffer,
    payload.inputShape,
    conv1Weight.values,
    conv1Weight.shape,
    payload.conv1Bias,
  );
  const contentRelu1Buffer = await runReluForwardBuffer(
    contentConv1Buffer,
    elementCount([1, 64, 16, 16]),
  );
  const contentRelu1 = await readGpuTensor(device, contentRelu1Buffer, [1, 64, 16, 16]);
  assertShape("contentRelu1", contentRelu1, [1, 64, 16, 16]);
  const contentConv2Buffer = await runConv2dForwardBuffer(
    contentRelu1Buffer,
    [1, 64, 16, 16],
    conv2Weight.values,
    conv2Weight.shape,
    payload.conv2Bias,
  );
  const contentRelu2Buffer = await runReluForwardBuffer(
    contentConv2Buffer,
    elementCount([1, 64, 16, 16]),
  );
  const contentRelu2 = await readGpuTensor(device, contentRelu2Buffer, [1, 64, 16, 16]);
  assertShape("contentRelu2", contentRelu2, [1, 64, 16, 16]);
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
  const styleGram1Buffer = uploadToOwnedBuffer(device, styleGram1);
  const styleGram3Buffer = uploadToOwnedBuffer(device, styleGram3);
  const contentRelu2TargetBuffer = uploadToOwnedBuffer(device, contentRelu2);
  releaseOwnedBuffer(styleInputBuffer);
  releaseOwnedBuffer(styleNormBuffer);
  releaseOwnedBuffer(styleConv1Buffer);
  releaseOwnedBuffer(styleRelu1Buffer);
  releaseOwnedBuffer(styleConv2Buffer);
  releaseOwnedBuffer(styleRelu2Buffer);
  releaseOwnedBuffer(stylePoolBuffer);
  releaseOwnedBuffer(styleConv3Buffer);
  releaseOwnedBuffer(styleRelu3Buffer);
  releaseOwnedBuffer(contentInputBuffer);
  releaseOwnedBuffer(contentNormBuffer);
  releaseOwnedBuffer(contentConv1Buffer);
  releaseOwnedBuffer(contentRelu1Buffer);
  releaseOwnedBuffer(contentConv2Buffer);
  releaseOwnedBuffer(contentRelu2Buffer);

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
      styleGram1Buffer,
      styleGram3Buffer,
      contentRelu2Buffer: contentRelu2TargetBuffer,
    },
    stepTemps: {},
    acquireTempBuffer,
    releaseTempBuffer,
  };

  let inputValues = new Float32Array(payload.initialInputValues);
  const losses: number[] = [];
  try {
    for (let step = 0; step < payload.steps; step += 1) {
    const inputBuffer = uploadToOwnedBuffer(device, inputValues);
    const normBuffer = await runNormalizeForwardBuffer(
      inputBuffer,
      payload.inputShape,
      payload.mean,
      payload.std,
    );
    const conv1Buffer = await runConv2dForwardBuffer(
      normBuffer,
      payload.inputShape,
      context.persistent.conv1Weight.values,
      context.persistent.conv1Weight.shape,
      payload.conv1Bias,
    );
    const relu1Buffer = await runReluForwardBuffer(conv1Buffer, elementCount([1, 64, 16, 16]));
    const conv2Buffer = await runConv2dForwardBuffer(
      relu1Buffer,
      [1, 64, 16, 16],
      context.persistent.conv2Weight.values,
      context.persistent.conv2Weight.shape,
      payload.conv2Bias,
    );
    const relu2Buffer = await runReluForwardBuffer(conv2Buffer, elementCount([1, 64, 16, 16]));
    const poolBuffer = await runMaxPool2dForwardBuffer(relu2Buffer, [1, 64, 16, 16]);
    const conv3Buffer = await runConv2dForwardBuffer(
      poolBuffer,
      [1, 64, 8, 8],
      context.persistent.conv3Weight.values,
      context.persistent.conv3Weight.shape,
      payload.conv3Bias,
    );
    const relu3Buffer = await runReluForwardBuffer(conv3Buffer, elementCount([1, 128, 8, 8]));
    const relu1GramBuffer = await runGramMatrixBuffer(relu1Buffer, [1, 64, 16, 16]);
    const relu3GramBuffer = await runGramMatrixBuffer(relu3Buffer, [1, 128, 8, 8]);
    const styleLoss1 =
      (await runMseBuffer(
        device,
        acquireReusableBuffer,
        releaseReusableBuffer,
        relu1GramBuffer,
        context.persistent.styleGram1Buffer,
        64 * 64,
        BUFFER_USAGE_STORAGE_COPY_SRC,
        BUFFER_USAGE_MAP_READ_COPY_DST,
        MAP_MODE_READ,
      )) * payload.styleWeightConv1;
    const styleLoss3 =
      (await runMseBuffer(
        device,
        acquireReusableBuffer,
        releaseReusableBuffer,
        relu3GramBuffer,
        context.persistent.styleGram3Buffer,
        128 * 128,
        BUFFER_USAGE_STORAGE_COPY_SRC,
        BUFFER_USAGE_MAP_READ_COPY_DST,
        MAP_MODE_READ,
      )) * payload.styleWeightConv3;
    const contentLoss =
      (await runMseBuffer(
        device,
        acquireReusableBuffer,
        releaseReusableBuffer,
        relu2Buffer,
        context.persistent.contentRelu2Buffer,
        64 * 16 * 16,
        BUFFER_USAGE_STORAGE_COPY_SRC,
        BUFFER_USAGE_MAP_READ_COPY_DST,
        MAP_MODE_READ,
      )) * payload.contentWeight;
    losses.push(styleLoss1 + styleLoss3 + contentLoss);
    releaseOwnedBuffer(relu1GramBuffer);
    releaseOwnedBuffer(relu3GramBuffer);

    const norm = await readGpuTensor(device, normBuffer, payload.inputShape);
    const conv1 = await readGpuTensor(device, conv1Buffer, [1, 64, 16, 16]);
    const relu1 = await readGpuTensor(device, relu1Buffer, [1, 64, 16, 16]);
    const conv2 = await readGpuTensor(device, conv2Buffer, [1, 64, 16, 16]);
    const relu2 = await readGpuTensor(device, relu2Buffer, [1, 64, 16, 16]);
    const pool = await readGpuTensor(device, poolBuffer, [1, 64, 8, 8]);
    const conv3 = await readGpuTensor(device, conv3Buffer, [1, 128, 8, 8]);
    const relu3 = await readGpuTensor(device, relu3Buffer, [1, 128, 8, 8]);
    context.stepTemps = { norm, conv1, relu1, conv2, relu2, pool, conv3, relu3 };
    assertShape("relu1", relu1, [1, 64, 16, 16]);
    assertShape("relu2", relu2, [1, 64, 16, 16]);
    assertShape("relu3", relu3, [1, 128, 8, 8]);
    releaseOwnedBuffer(inputBuffer);
    releaseOwnedBuffer(normBuffer);
    releaseOwnedBuffer(conv1Buffer);
    releaseOwnedBuffer(relu1Buffer);
    releaseOwnedBuffer(conv2Buffer);
    releaseOwnedBuffer(relu2Buffer);
    releaseOwnedBuffer(poolBuffer);
    releaseOwnedBuffer(conv3Buffer);
    releaseOwnedBuffer(relu3Buffer);

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
    releaseOwnedBuffer(context.persistent.styleGram1Buffer);
    releaseOwnedBuffer(context.persistent.styleGram3Buffer);
    releaseOwnedBuffer(context.persistent.contentRelu2Buffer);
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
