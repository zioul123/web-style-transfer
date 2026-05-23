/// <reference lib="webworker" />

import type { WorkerRequest } from "../../../../types";
import { createTensor } from "../../../index";
import { runReluBackwardBuffer } from "../../ops/relu/relu.run";
import {
  runNormalizeBackwardBuffer,
} from "../../ops/normalization/normalize.run";
import {
  runMaxPool2dBackwardBuffer,
} from "../../ops/pooling/maxpool.run";
import {
  runConv2dBackwardInputBuffer,
} from "../../ops/convolution/conv2d.run";
import { runMseBuffer } from "../../ops/loss/mse.run";
import {
  runContentLossBackwardBuffer,
} from "../../ops/loss/contentLoss.run";
import {
  runStyleLossBackward,
  runStyleLossBackwardFromTargetGramBuffer,
} from "../../ops/loss/styleLoss.run";
import {
  acquireReusableBuffer,
  releaseReusableBuffer,
} from "../../runtime/bufferPool";
import {
  readGpuBufferToArray,
  releaseOwnedBuffer,
  runUnaryShaderToBuffer,
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
import { runBinaryOpToBuffer } from "../../runtime/shaderRunner";
import { getGpuDevice } from "../../runtime/deviceState";
import { runUnary } from "../../runtime/computeContext";
import { runNormalizeForwardBuffer } from "../../ops/normalization/normalize.run";
import { runConv2dReluForwardBuffer } from "../../ops/convolution/conv2d.run";
import { runMaxPool2dForwardBuffer } from "../../ops/pooling/maxpool.run";
import { runGramMatrixBuffer } from "../../ops/gram/gram.run";
import { runReluBackward } from "../../ops/relu/relu.run";
import { runMaxPool2dBackward } from "../../ops/pooling/maxpool.run";
import { runConv2dBackwardInput } from "../../ops/convolution/conv2d.run";
import { runNormalizeBackward } from "../../ops/normalization/normalize.run";

type TensorShape4D = readonly [number, number, number, number];

type FirstPoolPersistentContext = {
  conv1Weight: ReturnType<typeof createTensor>;
  conv2Weight: ReturnType<typeof createTensor>;
  conv3Weight: ReturnType<typeof createTensor>;
  contentRelu2: Float32Array;
  styleRelu1: Float32Array;
  styleRelu3: Float32Array;
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
const DEBUG_COMPARE_GPU_MIGRATION = IS_DEV && import.meta.env.VITE_DEBUG_FIRST_POOL_GPU_COMPARE === "1";
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

const runScalarMulBuffer = (
  device: GPUDevice,
  input: GpuBufferRef,
  count: number,
  scalar: number,
): GpuBufferRef => {
  const scalarBuffer = uploadToOwnedBuffer(device, new Float32Array([scalar]));
  const out = runBinaryOpToBuffer(
    device,
    "mul",
    input.buffer,
    scalarBuffer.buffer,
    count,
    "tensorScalar",
  );
  releaseOwnedBuffer(scalarBuffer);
  return { buffer: out, owned: true };
};

const makeFusedUpdateClampShader = (count: number, learningRate: number): string => `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let updated = input[i] - ${learningRate} * grad[i];
  out[i] = clamp(updated, 0.0, 1.0);
}`;

const runFusedUpdateClampBuffer = (
  device: GPUDevice,
  input: GpuBufferRef,
  grad: GpuBufferRef,
  count: number,
  learningRate: number,
): GpuBufferRef => ({
  buffer: runUnaryShaderToBuffer(device, makeFusedUpdateClampShader(count, learningRate), input.buffer, count, [
    { binding: 1, resource: { buffer: grad.buffer } },
  ]),
  owned: true,
});

const assertNear = (label: string, a: Float32Array, b: Float32Array): void => {
  if (!DEBUG_COMPARE_GPU_MIGRATION) return;
  if (a.length !== b.length) throw new Error(`${label} debug compare length mismatch.`);
  let maxDiff = 0;
  for (let i = 0; i < a.length; i += 1) maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
  if (maxDiff > 1e-5) throw new Error(`${label} debug compare failed maxDiff=${maxDiff}`);
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
  const styleRelu1Buffer = await runConv2dReluForwardBuffer(
    styleNormBuffer,
    payload.inputShape,
    conv1Weight.values,
    conv1Weight.shape,
    payload.conv1Bias,
  );
  const styleRelu2Buffer = await runConv2dReluForwardBuffer(
    styleRelu1Buffer,
    [1, 64, 16, 16],
    conv2Weight.values,
    conv2Weight.shape,
    payload.conv2Bias,
  );
  const stylePoolBuffer = await runMaxPool2dForwardBuffer(styleRelu2Buffer, [1, 64, 16, 16]);
  const styleRelu3Buffer = await runConv2dReluForwardBuffer(
    stylePoolBuffer,
    [1, 64, 8, 8],
    conv3Weight.values,
    conv3Weight.shape,
    payload.conv3Bias,
  );
  const styleRelu1 = await readGpuTensor(device, styleRelu1Buffer, [1, 64, 16, 16]);
  const styleRelu3 = await readGpuTensor(device, styleRelu3Buffer, [1, 128, 8, 8]);
  assertShape("styleRelu1", styleRelu1, [1, 64, 16, 16]);
  assertShape("styleRelu3", styleRelu3, [1, 128, 8, 8]);

  const contentRelu1Buffer = await runConv2dReluForwardBuffer(
    contentNormBuffer,
    payload.inputShape,
    conv1Weight.values,
    conv1Weight.shape,
    payload.conv1Bias,
  );
  const contentRelu1 = await readGpuTensor(device, contentRelu1Buffer, [1, 64, 16, 16]);
  assertShape("contentRelu1", contentRelu1, [1, 64, 16, 16]);
  const contentRelu2Buffer = await runConv2dReluForwardBuffer(
    contentRelu1Buffer,
    [1, 64, 16, 16],
    conv2Weight.values,
    conv2Weight.shape,
    payload.conv2Bias,
  );
  const contentRelu2 = await readGpuTensor(device, contentRelu2Buffer, [1, 64, 16, 16]);
  assertShape("contentRelu2", contentRelu2, [1, 64, 16, 16]);
  const styleGram1Buffer = await runGramMatrixBuffer(styleRelu1Buffer, [1, 64, 16, 16]);
  const styleGram3Buffer = await runGramMatrixBuffer(styleRelu3Buffer, [1, 128, 8, 8]);
  const contentRelu2TargetBuffer = uploadToOwnedBuffer(device, contentRelu2);
  releaseOwnedBuffer(styleInputBuffer);
  releaseOwnedBuffer(styleNormBuffer);
  releaseOwnedBuffer(styleRelu1Buffer);
  releaseOwnedBuffer(styleRelu2Buffer);
  releaseOwnedBuffer(stylePoolBuffer);
  releaseOwnedBuffer(styleRelu3Buffer);
  releaseOwnedBuffer(contentInputBuffer);
  releaseOwnedBuffer(contentNormBuffer);
  releaseOwnedBuffer(contentRelu1Buffer);
  releaseOwnedBuffer(contentRelu2Buffer);

  const context: FirstPoolComputeContext = {
    persistent: {
      conv1Weight,
      conv2Weight,
      conv3Weight,
      contentRelu2,
      styleRelu1,
      styleRelu3,
      styleGram1Buffer,
      styleGram3Buffer,
      contentRelu2Buffer: contentRelu2TargetBuffer,
    },
    stepTemps: {},
    acquireTempBuffer,
    releaseTempBuffer,
  };

  let inputBuffer: GpuBufferRef = uploadToOwnedBuffer(
    device,
    new Float32Array(payload.initialInputValues),
  );
  const losses: number[] = [];
  try {
    for (let step = 0; step < payload.steps; step += 1) {
    const stepOwnedBuffers: GpuBufferRef[] = [];
    try {
    const normBuffer = await runNormalizeForwardBuffer(
      inputBuffer,
      payload.inputShape,
      payload.mean,
      payload.std,
    );
    stepOwnedBuffers.push(normBuffer);
    const relu1Buffer = await runConv2dReluForwardBuffer(
      normBuffer,
      payload.inputShape,
      context.persistent.conv1Weight.values,
      context.persistent.conv1Weight.shape,
      payload.conv1Bias,
    );
    stepOwnedBuffers.push(relu1Buffer);
    const relu2Buffer = await runConv2dReluForwardBuffer(
      relu1Buffer,
      [1, 64, 16, 16],
      context.persistent.conv2Weight.values,
      context.persistent.conv2Weight.shape,
      payload.conv2Bias,
    );
    stepOwnedBuffers.push(relu2Buffer);
    const poolBuffer = await runMaxPool2dForwardBuffer(relu2Buffer, [1, 64, 16, 16]);
    stepOwnedBuffers.push(poolBuffer);
    const relu3Buffer = await runConv2dReluForwardBuffer(
      poolBuffer,
      [1, 64, 8, 8],
      context.persistent.conv3Weight.values,
      context.persistent.conv3Weight.shape,
      payload.conv3Bias,
    );
    stepOwnedBuffers.push(relu3Buffer);
    const relu1GramBuffer = await runGramMatrixBuffer(relu1Buffer, [1, 64, 16, 16]);
    const relu3GramBuffer = await runGramMatrixBuffer(relu3Buffer, [1, 128, 8, 8]);
    stepOwnedBuffers.push(relu1GramBuffer, relu3GramBuffer);
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
    const norm = await readGpuTensor(device, normBuffer, payload.inputShape);
    const conv1 = await readGpuTensor(device, relu1Buffer, [1, 64, 16, 16]);
    const relu1 = await readGpuTensor(device, relu1Buffer, [1, 64, 16, 16]);
    const conv2 = await readGpuTensor(device, relu2Buffer, [1, 64, 16, 16]);
    const relu2 = await readGpuTensor(device, relu2Buffer, [1, 64, 16, 16]);
    const pool = await readGpuTensor(device, poolBuffer, [1, 64, 8, 8]);
    const conv3 = await readGpuTensor(device, relu3Buffer, [1, 128, 8, 8]);
    const relu3 = await readGpuTensor(device, relu3Buffer, [1, 128, 8, 8]);
    context.stepTemps = { norm, conv1, relu1, conv2, relu2, pool, conv3, relu3 };
    assertShape("relu1", relu1, [1, 64, 16, 16]);
    assertShape("relu2", relu2, [1, 64, 16, 16]);
    assertShape("relu3", relu3, [1, 128, 8, 8]);
    const gStyle1ReluBuffer = runScalarMulBuffer(
      device,
      await runStyleLossBackwardFromTargetGramBuffer(
        relu1Buffer,
        [1, 64, 16, 16],
        context.persistent.styleGram1Buffer,
      ),
      elementCount([1, 64, 16, 16]),
      payload.styleWeightConv1,
    );
    stepOwnedBuffers.push(gStyle1ReluBuffer);
    const gStyle3ReluBuffer = runScalarMulBuffer(
      device,
      await runStyleLossBackwardFromTargetGramBuffer(
        relu3Buffer,
        [1, 128, 8, 8],
        context.persistent.styleGram3Buffer,
      ),
      elementCount([1, 128, 8, 8]),
      payload.styleWeightConv3,
    );
    stepOwnedBuffers.push(gStyle3ReluBuffer);
    const gContentReluBuffer = runScalarMulBuffer(
      device,
      await runContentLossBackwardBuffer(
        relu2Buffer,
        context.persistent.contentRelu2Buffer,
        elementCount([1, 64, 16, 16]),
      ),
      elementCount([1, 64, 16, 16]),
      payload.contentWeight,
    );
    stepOwnedBuffers.push(gContentReluBuffer);
    const gStyle3Buffer = await runReluBackwardBuffer(
      relu3Buffer,
      gStyle3ReluBuffer,
      elementCount([1, 128, 8, 8]),
    );
    stepOwnedBuffers.push(gStyle3Buffer);
    const gContentBuffer = await runReluBackwardBuffer(
      relu2Buffer,
      gContentReluBuffer,
      elementCount([1, 64, 16, 16]),
    );
    stepOwnedBuffers.push(gContentBuffer);
    const gStyle1Buffer = await runReluBackwardBuffer(
      relu1Buffer,
      gStyle1ReluBuffer,
      elementCount([1, 64, 16, 16]),
    );
    stepOwnedBuffers.push(gStyle1Buffer);

    const gPoolBuffer = await runConv2dBackwardInputBuffer(
      gStyle3Buffer,
      [1, 64, 8, 8],
      context.persistent.conv3Weight.values,
      context.persistent.conv3Weight.shape,
    );
    stepOwnedBuffers.push(gPoolBuffer);
    const gRelu2Buffer = await runMaxPool2dBackwardBuffer(
      relu2Buffer,
      [1, 64, 16, 16],
      gPoolBuffer,
    );
    stepOwnedBuffers.push(gRelu2Buffer);
    const gConv2FromPoolBuffer = await runReluBackwardBuffer(
      relu2Buffer,
      gRelu2Buffer,
      elementCount([1, 64, 16, 16]),
    );
    stepOwnedBuffers.push(gConv2FromPoolBuffer);
    const gConv2TotalBuffer: GpuBufferRef = {
      buffer: runBinaryOpToBuffer(
        device,
        "add",
        gConv2FromPoolBuffer.buffer,
        gContentBuffer.buffer,
        elementCount([1, 64, 16, 16]),
        "tensorTensor",
      ),
      owned: true,
    };
    const gRelu1Buffer = await runConv2dBackwardInputBuffer(
      gConv2TotalBuffer,
      [1, 64, 16, 16],
      context.persistent.conv2Weight.values,
      context.persistent.conv2Weight.shape,
    );
    stepOwnedBuffers.push(gConv2TotalBuffer, gRelu1Buffer);
    const gConv1FromConv2Buffer = await runReluBackwardBuffer(
      relu1Buffer,
      gRelu1Buffer,
      elementCount([1, 64, 16, 16]),
    );
    stepOwnedBuffers.push(gConv1FromConv2Buffer);
    const gConv1TotalBuffer: GpuBufferRef = {
      buffer: runBinaryOpToBuffer(
        device,
        "add",
        gConv1FromConv2Buffer.buffer,
        gStyle1Buffer.buffer,
        elementCount([1, 64, 16, 16]),
        "tensorTensor",
      ),
      owned: true,
    };
    const gNormBuffer = await runConv2dBackwardInputBuffer(
      gConv1TotalBuffer,
      payload.inputShape,
      context.persistent.conv1Weight.values,
      context.persistent.conv1Weight.shape,
    );
    stepOwnedBuffers.push(gConv1TotalBuffer, gNormBuffer);
    const gInputBuffer = await runNormalizeBackwardBuffer(
      gNormBuffer,
      payload.inputShape,
      payload.std,
    );
    stepOwnedBuffers.push(gInputBuffer);
    const clampedInputBuffer = runFusedUpdateClampBuffer(
      device,
      inputBuffer,
      gInputBuffer,
      elementCount(payload.inputShape),
      payload.learningRate,
    );
    const gInput = await readGpuTensor(device, gInputBuffer, payload.inputShape);

    if (DEBUG_COMPARE_GPU_MIGRATION && step === 0) {
      const legacyStyle = await runStyleLossBackward(device, runUnary, relu1, [1, 64, 16, 16], context.persistent.styleRelu1, BUFFER_USAGE_STORAGE_COPY_DST, BUFFER_USAGE_UNIFORM_COPY_DST);
      const migratedStyle = await readGpuTensor(device, gStyle1ReluBuffer, [1, 64, 16, 16]);
      const weightedLegacyStyle = new Float32Array(legacyStyle.length);
      for (let i = 0; i < legacyStyle.length; i += 1) weightedLegacyStyle[i] = legacyStyle[i] * payload.styleWeightConv1;
      assertNear("gStyle1Relu", migratedStyle, weightedLegacyStyle);

      const legacyStyle3 = await runStyleLossBackward(
        device,
        runUnary,
        relu3,
        [1, 128, 8, 8],
        context.persistent.styleRelu3,
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_UNIFORM_COPY_DST,
      );
      const legacyStyle3Weighted = new Float32Array(legacyStyle3.length);
      for (let i = 0; i < legacyStyle3.length; i += 1) legacyStyle3Weighted[i] = legacyStyle3[i] * payload.styleWeightConv3;
      const legacyGStyle3 = await runReluBackward(device, runUnary, conv3, legacyStyle3Weighted);
      const legacyGPool = await runConv2dBackwardInput(
        device,
        runUnary,
        [1, 64, 8, 8],
        legacyGStyle3,
        context.persistent.conv3Weight.values,
        context.persistent.conv3Weight.shape,
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_UNIFORM_COPY_DST,
      );
      const legacyGRelu2 = await runMaxPool2dBackward(
        device,
        runUnary,
        relu2,
        [1, 64, 16, 16],
        legacyGPool,
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_UNIFORM_COPY_DST,
      );
      const legacyGConv2FromPool = await runReluBackward(
        device,
        runUnary,
        conv2,
        legacyGRelu2,
      );
      const legacyContent = await runContentLossBackwardBuffer(
        relu2Buffer,
        context.persistent.contentRelu2Buffer,
        elementCount([1, 64, 16, 16]),
      );
      const legacyContentWeighted = runScalarMulBuffer(
        device,
        legacyContent,
        elementCount([1, 64, 16, 16]),
        payload.contentWeight,
      );
      stepOwnedBuffers.push(legacyContent, legacyContentWeighted);
      const legacyContentWeightedArray = await readGpuTensor(device, legacyContentWeighted, [1, 64, 16, 16]);
      const legacyContentArray = await runReluBackward(
        device,
        runUnary,
        conv2,
        legacyContentWeightedArray,
      );
      const legacyGConv2Total = new Float32Array(legacyGConv2FromPool.length);
      for (let i = 0; i < legacyGConv2Total.length; i += 1) legacyGConv2Total[i] = legacyGConv2FromPool[i] + legacyContentArray[i];
      const migratedGConv2Total = await readGpuTensor(device, gConv2TotalBuffer, [1, 64, 16, 16]);
      assertNear("gConv2Total", migratedGConv2Total, legacyGConv2Total);

      const legacyRelu1 = await runConv2dBackwardInput(
        device,
        runUnary,
        [1, 64, 16, 16],
        legacyGConv2Total,
        context.persistent.conv2Weight.values,
        context.persistent.conv2Weight.shape,
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_UNIFORM_COPY_DST,
      );
      const legacyConv1FromConv2 = await runReluBackward(device, runUnary, conv1, legacyRelu1);
      const legacyStyle1 = await runReluBackward(device, runUnary, conv1, weightedLegacyStyle);
      const legacyConv1Total = new Float32Array(legacyConv1FromConv2.length);
      for (let i = 0; i < legacyConv1Total.length; i += 1) legacyConv1Total[i] = legacyConv1FromConv2[i] + legacyStyle1[i];
      const legacyGNorm = await runConv2dBackwardInput(
        device,
        runUnary,
        payload.inputShape,
        legacyConv1Total,
        context.persistent.conv1Weight.values,
        context.persistent.conv1Weight.shape,
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_UNIFORM_COPY_DST,
      );
      const legacyGInput = await runNormalizeBackward(
        device,
        runUnary,
        legacyGNorm,
        payload.inputShape,
        payload.std,
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_UNIFORM_COPY_DST,
      );
      assertNear("gInput", gInput, legacyGInput);
    }

    const previousInputBuffer = inputBuffer;
    inputBuffer = clampedInputBuffer;
    releaseOwnedBuffer(previousInputBuffer);
    context.stepTemps.gInput = gInput;
    assertShape("gInput", gInput, payload.inputShape);
    } finally {
      for (const ownedBuffer of stepOwnedBuffers) releaseOwnedBuffer(ownedBuffer);
    }
    }
    const finalValues = await readGpuTensor(device, inputBuffer, payload.inputShape);
    return { losses, finalValues: Array.from(finalValues) };
  } finally {
    releaseOwnedBuffer(inputBuffer);
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
