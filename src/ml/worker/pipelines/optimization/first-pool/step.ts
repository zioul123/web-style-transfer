import type { WorkerRequest } from "../../../../../types";
import { runMseBuffer } from "../../../ops/loss/mse.run";
import { readGpuBufferToArray, type GpuBufferRef, type OwnedGpuBuffer } from "../../../runtime/bufferKernels";
import { BUFFER_USAGE_MAP_READ_COPY_DST, BUFFER_USAGE_STORAGE_COPY_SRC, MAP_MODE_READ } from "../../../runtime/gpuFlags";
import type { FirstPoolPersistentContext, FirstPoolRuntimeContext, FirstPoolStepResult, TensorShape4D } from "./types";
import { runFusedUpdateClampBuffer, runUnfusedUpdateClampBuffer } from "./updateKernels";
import { createFirstPoolTrackedOps } from "./trackedOps";

const elementCount = (shape: TensorShape4D): number => shape[0] * shape[1] * shape[2] * shape[3];

const assertShape = (valuesLength: number, shape: TensorShape4D, name: string): void => {
  const expectedLength = elementCount(shape);
  if (valuesLength !== expectedLength) {
    throw new Error(`${name} has ${valuesLength} values but expected ${expectedLength}`);
  }
};

export const runFirstPoolStep = async (
  device: GPUDevice,
  payload: Extract<WorkerRequest, { type: "run-first-pool-optimizer" }>,
  persistent: FirstPoolPersistentContext,
  inputBuffer: GpuBufferRef,
  runtimeContext: FirstPoolRuntimeContext,
): Promise<FirstPoolStepResult> => {
  const useFusedConvRelu = payload.useFusedConvRelu ?? true;
  const useFusedUpdateClamp = payload.useFusedUpdateClamp ?? true;
  const debugValidateStepShapes = payload.debugValidateStepShapes ?? false;
  const debugReadbackGrad = payload.debugReadbackGrad ?? false;
  const debugUseLegacyCpuLossReadback = payload.debugUseLegacyCpuLossReadback ?? true;
  const acquireMseBuffer = (_gpuDevice: GPUDevice, size: number, usage: number): GPUBuffer => {
    const floatCount = Math.ceil(size / Float32Array.BYTES_PER_ELEMENT);
    return runtimeContext.acquireTemp([1, 1, 1, floatCount], usage, "mse-reduction");
  };
  const tracked = createFirstPoolTrackedOps(device, runtimeContext);
  try {
    const forwardStart = performance.now();
    const normBuffer = await tracked.normalizeForward(inputBuffer, payload.inputShape, payload.mean, payload.std);
    const relu1Buffer = useFusedConvRelu ? await tracked.conv2dReluForward(normBuffer, payload.inputShape, persistent.conv1Weight.values, persistent.conv1Weight.shape, payload.conv1Bias) : await tracked.reluForward(await tracked.conv2dForward(normBuffer, payload.inputShape, persistent.conv1Weight.values, persistent.conv1Weight.shape, payload.conv1Bias), elementCount([1, 64, 16, 16]));
    const relu2Buffer = useFusedConvRelu ? await tracked.conv2dReluForward(relu1Buffer, [1, 64, 16, 16], persistent.conv2Weight.values, persistent.conv2Weight.shape, payload.conv2Bias) : await tracked.reluForward(await tracked.conv2dForward(relu1Buffer, [1, 64, 16, 16], persistent.conv2Weight.values, persistent.conv2Weight.shape, payload.conv2Bias), elementCount([1, 64, 16, 16]));
    const poolBuffer = await tracked.maxPoolForward(relu2Buffer, [1, 64, 16, 16]);
    const relu3Buffer = useFusedConvRelu ? await tracked.conv2dReluForward(poolBuffer, [1, 64, 8, 8], persistent.conv3Weight.values, persistent.conv3Weight.shape, payload.conv3Bias) : await tracked.reluForward(await tracked.conv2dForward(poolBuffer, [1, 64, 8, 8], persistent.conv3Weight.values, persistent.conv3Weight.shape, payload.conv3Bias), elementCount([1, 128, 8, 8]));
    const forwardMs = performance.now() - forwardStart;

    const relu1GramBuffer = await tracked.gram(relu1Buffer, [1, 64, 16, 16]);
    const relu3GramBuffer = await tracked.gram(relu3Buffer, [1, 128, 8, 8]);

    const lossStart = performance.now();
    let totalLoss = 0;
    if (debugUseLegacyCpuLossReadback) {
      const styleLoss1 = (await runMseBuffer(device, acquireMseBuffer, relu1GramBuffer, persistent.styleGram1Buffer, 64 * 64, BUFFER_USAGE_STORAGE_COPY_SRC, BUFFER_USAGE_MAP_READ_COPY_DST, MAP_MODE_READ)) * payload.styleWeightConv1;
      const styleLoss3 = (await runMseBuffer(device, acquireMseBuffer, relu3GramBuffer, persistent.styleGram3Buffer, 128 * 128, BUFFER_USAGE_STORAGE_COPY_SRC, BUFFER_USAGE_MAP_READ_COPY_DST, MAP_MODE_READ)) * payload.styleWeightConv3;
      const contentLoss = (await runMseBuffer(device, acquireMseBuffer, relu2Buffer, persistent.contentRelu2Buffer, 64 * 16 * 16, BUFFER_USAGE_STORAGE_COPY_SRC, BUFFER_USAGE_MAP_READ_COPY_DST, MAP_MODE_READ)) * payload.contentWeight;
      totalLoss = styleLoss1 + styleLoss3 + contentLoss;
    } else {
      const styleLoss1Buffer = await tracked.mseScalarBuffer(acquireMseBuffer, relu1GramBuffer, persistent.styleGram1Buffer, 64 * 64, BUFFER_USAGE_STORAGE_COPY_SRC);
      const styleLoss3Buffer = await tracked.mseScalarBuffer(acquireMseBuffer, relu3GramBuffer, persistent.styleGram3Buffer, 128 * 128, BUFFER_USAGE_STORAGE_COPY_SRC);
      const contentLossBuffer = await tracked.mseScalarBuffer(acquireMseBuffer, relu2Buffer, persistent.contentRelu2Buffer, 64 * 16 * 16, BUFFER_USAGE_STORAGE_COPY_SRC);
      const weightedStyle1LossBuffer = tracked.scalarMul(styleLoss1Buffer, 1, payload.styleWeightConv1 / (64 * 64));
      const weightedStyle3LossBuffer = tracked.scalarMul(styleLoss3Buffer, 1, payload.styleWeightConv3 / (128 * 128));
      const weightedContentLossBuffer = tracked.scalarMul(contentLossBuffer, 1, payload.contentWeight / (64 * 16 * 16));
      const styleLossTotalBuffer = tracked.add(weightedStyle1LossBuffer, weightedStyle3LossBuffer, 1);
      const totalLossBuffer = tracked.add(styleLossTotalBuffer, weightedContentLossBuffer, 1);
      totalLoss = (await readGpuBufferToArray(device, totalLossBuffer.buffer, 1))[0];
    }
    const lossMs = performance.now() - lossStart;

    const backwardStart = performance.now();
    const gStyle1ReluBuffer = tracked.scalarMul(await tracked.styleLossBackward(relu1Buffer, [1, 64, 16, 16], persistent.styleGram1Buffer), elementCount([1, 64, 16, 16]), payload.styleWeightConv1);
    const gStyle3ReluBuffer = tracked.scalarMul(await tracked.styleLossBackward(relu3Buffer, [1, 128, 8, 8], persistent.styleGram3Buffer), elementCount([1, 128, 8, 8]), payload.styleWeightConv3);
    const gContentReluBuffer = tracked.scalarMul(await tracked.contentLossBackward(relu2Buffer, persistent.contentRelu2Buffer, elementCount([1, 64, 16, 16])), elementCount([1, 64, 16, 16]), payload.contentWeight);
    const gStyle3Buffer = await tracked.reluBackward(relu3Buffer, gStyle3ReluBuffer, elementCount([1, 128, 8, 8]));
    const gContentBuffer = await tracked.reluBackward(relu2Buffer, gContentReluBuffer, elementCount([1, 64, 16, 16]));
    const gStyle1Buffer = await tracked.reluBackward(relu1Buffer, gStyle1ReluBuffer, elementCount([1, 64, 16, 16]));
    const gPoolBuffer = await tracked.conv2dBackwardInput(gStyle3Buffer, [1, 64, 8, 8], persistent.conv3Weight.values, persistent.conv3Weight.shape);
    const gRelu2Buffer = await tracked.maxPoolBackward(relu2Buffer, [1, 64, 16, 16], gPoolBuffer);
    const gConv2FromPoolBuffer = await tracked.reluBackward(relu2Buffer, gRelu2Buffer, elementCount([1, 64, 16, 16]));
    const gConv2TotalBuffer = tracked.add(gConv2FromPoolBuffer, gContentBuffer, elementCount([1, 64, 16, 16]));
    const gRelu1Buffer = await tracked.conv2dBackwardInput(gConv2TotalBuffer, [1, 64, 16, 16], persistent.conv2Weight.values, persistent.conv2Weight.shape);
    const gConv1FromConv2Buffer = await tracked.reluBackward(relu1Buffer, gRelu1Buffer, elementCount([1, 64, 16, 16]));
    const gConv1TotalBuffer = tracked.add(gConv1FromConv2Buffer, gStyle1Buffer, elementCount([1, 64, 16, 16]));
    const gNormBuffer = await tracked.conv2dBackwardInput(gConv1TotalBuffer, payload.inputShape, persistent.conv1Weight.values, persistent.conv1Weight.shape);
    const gInputBuffer = await tracked.normalizeBackward(gNormBuffer, payload.inputShape, payload.std);
    const backwardMs = performance.now() - backwardStart;

    const updateStart = performance.now();
    let clampedInputBuffer: OwnedGpuBuffer;
    if (useFusedUpdateClamp) {
      clampedInputBuffer = runFusedUpdateClampBuffer(device, inputBuffer, gInputBuffer, elementCount(payload.inputShape), payload.learningRate);
    } else {
      const unfused = runUnfusedUpdateClampBuffer(device, inputBuffer, gInputBuffer, elementCount(payload.inputShape), payload.learningRate);
      tracked.ownMany(...unfused.intermediates);
      clampedInputBuffer = unfused.clamped;
    }
    const updateMs = performance.now() - updateStart;

    let diagnosticsReadbackMs = 0;
    assertShape(elementCount([1, 64, 16, 16]), [1, 64, 16, 16], "relu1");
    assertShape(elementCount([1, 64, 16, 16]), [1, 64, 16, 16], "relu2");
    assertShape(elementCount([1, 128, 8, 8]), [1, 128, 8, 8], "relu3");
    assertShape(elementCount(payload.inputShape), payload.inputShape, "gInput");

    if (debugValidateStepShapes) {
      const debugShapeReadStart = performance.now();
      const relu1 = await readGpuBufferToArray(device, relu1Buffer.buffer, elementCount([1, 64, 16, 16]));
      const relu2 = await readGpuBufferToArray(device, relu2Buffer.buffer, elementCount([1, 64, 16, 16]));
      const relu3 = await readGpuBufferToArray(device, relu3Buffer.buffer, elementCount([1, 128, 8, 8]));
      diagnosticsReadbackMs += performance.now() - debugShapeReadStart;
      assertShape(relu1.length, [1, 64, 16, 16], "relu1");
      assertShape(relu2.length, [1, 64, 16, 16], "relu2");
      assertShape(relu3.length, [1, 128, 8, 8], "relu3");
    }

    if (debugReadbackGrad) {
      const debugGradReadStart = performance.now();
      const gInput = await readGpuBufferToArray(device, gInputBuffer.buffer, elementCount(payload.inputShape));
      diagnosticsReadbackMs += performance.now() - debugGradReadStart;
      assertShape(gInput.length, payload.inputShape, "gInput");
    }

    return { nextInputBuffer: clampedInputBuffer, totalLoss, forwardMs, lossMs, backwardMs, updateMs, diagnosticsReadbackMs };
  } finally {
    runtimeContext.releaseStepOwned();
  }
};
