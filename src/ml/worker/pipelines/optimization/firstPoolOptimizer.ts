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
  BUFFER_USAGE_STORAGE_COPY_SRC,
  MAP_MODE_READ,
} from "../../runtime/gpuFlags";
import { makeClampShader, runBinaryOpToBuffer } from "../../runtime/shaderRunner";
import { getGpuDevice } from "../../runtime/deviceState";
import { runNormalizeForwardBuffer } from "../../ops/normalization/normalize.run";
import { runConv2dForwardBuffer, runConv2dReluForwardBuffer } from "../../ops/convolution/conv2d.run";
import { runReluForwardBuffer } from "../../ops/relu/relu.run";
import { runMaxPool2dForwardBuffer } from "../../ops/pooling/maxpool.run";
import { runGramMatrixBuffer } from "../../ops/gram/gram.run";

type TensorShape4D = readonly [number, number, number, number];

type FirstPoolPersistentContext = {
  conv1Weight: ReturnType<typeof createTensor>;
  conv2Weight: ReturnType<typeof createTensor>;
  conv3Weight: ReturnType<typeof createTensor>;
  contentRelu2: Float32Array;
  styleGram1Buffer: GpuBufferRef;
  styleGram3Buffer: GpuBufferRef;
  contentRelu2Buffer: GpuBufferRef;
};

type FirstPoolComputeContext = {
  persistent: FirstPoolPersistentContext;
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

const runUnfusedUpdateClampBuffer = (
  device: GPUDevice,
  input: GpuBufferRef,
  grad: GpuBufferRef,
  count: number,
  learningRate: number,
): { clamped: GpuBufferRef; intermediates: GpuBufferRef[] } => {
  const scaledGrad = runScalarMulBuffer(device, grad, count, learningRate);
  const updated: GpuBufferRef = {
    buffer: runBinaryOpToBuffer(device, "sub", input.buffer, scaledGrad.buffer, count, "tensorTensor"),
    owned: true,
  };
  const clamped: GpuBufferRef = {
    buffer: runUnaryShaderToBuffer(device, makeClampShader(count, 0, 1), updated.buffer, count),
    owned: true,
  };
  return { clamped, intermediates: [scaledGrad, updated] };
};


export const runFirstPoolOptimizer = async (
  payload: Extract<WorkerRequest, { type: "run-first-pool-optimizer" }>,
): Promise<{ losses: number[]; finalValues: number[]; stats?: {
  elapsedMs: number; avgStepMs: number; forwardMs: number; lossMs: number; backwardMs: number; updateMs: number; readbackMs: number; steps: number;
} }> => {
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
      styleGram1Buffer,
      styleGram3Buffer,
      contentRelu2Buffer: contentRelu2TargetBuffer,
    },
    acquireTempBuffer,
    releaseTempBuffer,
  };

  let inputBuffer: GpuBufferRef = uploadToOwnedBuffer(
    device,
    new Float32Array(payload.initialInputValues),
  );
  const losses: number[] = [];
  const useFusedConvRelu = payload.useFusedConvRelu ?? true;
  const useFusedUpdateClamp = payload.useFusedUpdateClamp ?? true;
  const collectBenchmarkStats = payload.collectBenchmarkStats ?? false;
  let forwardMs = 0;
  let lossMs = 0;
  let backwardMs = 0;
  let updateMs = 0;
  let readbackMs = 0;
  const runStart = performance.now();
  try {
    for (let step = 0; step < payload.steps; step += 1) {
    const stepOwnedBuffers: GpuBufferRef[] = [];
    try {
    const forwardStart = performance.now();
    const normBuffer = await runNormalizeForwardBuffer(
      inputBuffer,
      payload.inputShape,
      payload.mean,
      payload.std,
    );
    stepOwnedBuffers.push(normBuffer);
    const relu1Buffer = useFusedConvRelu ? await runConv2dReluForwardBuffer(
      normBuffer,
      payload.inputShape,
      context.persistent.conv1Weight.values,
      context.persistent.conv1Weight.shape,
      payload.conv1Bias,
    ) : await runReluForwardBuffer(await runConv2dForwardBuffer(
      normBuffer, payload.inputShape, context.persistent.conv1Weight.values, context.persistent.conv1Weight.shape, payload.conv1Bias,
    ), elementCount([1, 64, 16, 16]));
    stepOwnedBuffers.push(relu1Buffer);
    const relu2Buffer = useFusedConvRelu ? await runConv2dReluForwardBuffer(
      relu1Buffer,
      [1, 64, 16, 16],
      context.persistent.conv2Weight.values,
      context.persistent.conv2Weight.shape,
      payload.conv2Bias,
    ) : await runReluForwardBuffer(await runConv2dForwardBuffer(
      relu1Buffer, [1, 64, 16, 16], context.persistent.conv2Weight.values, context.persistent.conv2Weight.shape, payload.conv2Bias,
    ), elementCount([1, 64, 16, 16]));
    stepOwnedBuffers.push(relu2Buffer);
    const poolBuffer = await runMaxPool2dForwardBuffer(relu2Buffer, [1, 64, 16, 16]);
    stepOwnedBuffers.push(poolBuffer);
    const relu3Buffer = useFusedConvRelu ? await runConv2dReluForwardBuffer(
      poolBuffer,
      [1, 64, 8, 8],
      context.persistent.conv3Weight.values,
      context.persistent.conv3Weight.shape,
      payload.conv3Bias,
    ) : await runReluForwardBuffer(await runConv2dForwardBuffer(
      poolBuffer, [1, 64, 8, 8], context.persistent.conv3Weight.values, context.persistent.conv3Weight.shape, payload.conv3Bias,
    ), elementCount([1, 128, 8, 8]));
    stepOwnedBuffers.push(relu3Buffer);
    forwardMs += performance.now() - forwardStart;
    const relu1GramBuffer = await runGramMatrixBuffer(relu1Buffer, [1, 64, 16, 16]);
    const relu3GramBuffer = await runGramMatrixBuffer(relu3Buffer, [1, 128, 8, 8]);
    stepOwnedBuffers.push(relu1GramBuffer, relu3GramBuffer);
    const lossStart = performance.now();
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
    lossMs += performance.now() - lossStart;
    const readbackStart = performance.now();
    const relu1 = await readGpuTensor(device, relu1Buffer, [1, 64, 16, 16]);
    const relu2 = await readGpuTensor(device, relu2Buffer, [1, 64, 16, 16]);
    const relu3 = await readGpuTensor(device, relu3Buffer, [1, 128, 8, 8]);
    readbackMs += performance.now() - readbackStart;
    assertShape("relu1", relu1, [1, 64, 16, 16]);
    assertShape("relu2", relu2, [1, 64, 16, 16]);
    assertShape("relu3", relu3, [1, 128, 8, 8]);
    const backwardStart = performance.now();
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
    backwardMs += performance.now() - backwardStart;
    const updateStart = performance.now();
    let clampedInputBuffer: GpuBufferRef;
    if (useFusedUpdateClamp) {
      clampedInputBuffer = runFusedUpdateClampBuffer(
      device,
      inputBuffer,
      gInputBuffer,
      elementCount(payload.inputShape),
      payload.learningRate,
      );
    } else {
      const unfused = runUnfusedUpdateClampBuffer(
        device,
        inputBuffer,
        gInputBuffer,
        elementCount(payload.inputShape),
        payload.learningRate,
      );
      clampedInputBuffer = unfused.clamped;
      stepOwnedBuffers.push(...unfused.intermediates);
    }
    updateMs += performance.now() - updateStart;
    const gInputReadStart = performance.now();
    const gInput = await readGpuTensor(device, gInputBuffer, payload.inputShape);
    readbackMs += performance.now() - gInputReadStart;

    const previousInputBuffer = inputBuffer;
    inputBuffer = clampedInputBuffer;
    releaseOwnedBuffer(previousInputBuffer);
    assertShape("gInput", gInput, payload.inputShape);
    } finally {
      for (const ownedBuffer of stepOwnedBuffers) releaseOwnedBuffer(ownedBuffer);
    }
    }
    const finalReadStart = performance.now();
    const finalValues = await readGpuTensor(device, inputBuffer, payload.inputShape);
    readbackMs += performance.now() - finalReadStart;
    const elapsedMs = performance.now() - runStart;
    return {
      losses,
      finalValues: Array.from(finalValues),
      stats: collectBenchmarkStats ? {
        elapsedMs,
        avgStepMs: payload.steps > 0 ? elapsedMs / payload.steps : 0,
        forwardMs,
        lossMs,
        backwardMs,
        updateMs,
        readbackMs,
        steps: payload.steps,
      } : undefined,
    };
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
