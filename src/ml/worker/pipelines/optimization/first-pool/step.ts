import type { WorkerRequest } from "../../../../../types";
import { runConv2dBackwardInputBuffer, runConv2dForwardBuffer, runConv2dReluForwardBuffer } from "../../../ops/convolution/conv2d.run";
import { runGramMatrixBuffer } from "../../../ops/gram/gram.run";
import { runContentLossBackwardBuffer } from "../../../ops/loss/contentLoss.run";
import { runMseBuffer } from "../../../ops/loss/mse.run";
import { runStyleLossBackwardFromTargetGramBuffer } from "../../../ops/loss/styleLoss.run";
import { runNormalizeBackwardBuffer, runNormalizeForwardBuffer } from "../../../ops/normalization/normalize.run";
import { runMaxPool2dBackwardBuffer, runMaxPool2dForwardBuffer } from "../../../ops/pooling/maxpool.run";
import { runReluBackwardBuffer, runReluForwardBuffer } from "../../../ops/relu/relu.run";
import { readGpuBufferToArray, releaseOwnedBuffer, type GpuBufferRef, type OwnedGpuBuffer } from "../../../runtime/bufferKernels";
import { BUFFER_USAGE_MAP_READ_COPY_DST, BUFFER_USAGE_STORAGE_COPY_SRC, MAP_MODE_READ } from "../../../runtime/gpuFlags";
import { runBinaryOpToBuffer } from "../../../runtime/shaderRunner";
import type { FirstPoolPersistentContext, FirstPoolStepResult, FirstPoolTempBufferStore, TensorShape4D } from "./types";
import { runFusedUpdateClampBuffer, runScalarMulBuffer, runUnfusedUpdateClampBuffer } from "./updateKernels";

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
  tempBuffers: FirstPoolTempBufferStore,
): Promise<FirstPoolStepResult> => {
  const stepOwnedBuffers: GpuBufferRef[] = [];
  const useFusedConvRelu = payload.useFusedConvRelu ?? true;
  const useFusedUpdateClamp = payload.useFusedUpdateClamp ?? true;
  const debugValidateStepShapes = payload.debugValidateStepShapes ?? false;
  const debugReadbackGrad = payload.debugReadbackGrad ?? false;
  const acquireMseBuffer = (_gpuDevice: GPUDevice, size: number, usage: number): GPUBuffer => {
    const floatCount = Math.ceil(size / Float32Array.BYTES_PER_ELEMENT);
    return tempBuffers.acquireTempBuffer([1, 1, 1, floatCount], usage, "mse-reduction");
  };
  const releaseMseBuffer = (size: number, usage: number): void => {
    const floatCount = Math.ceil(size / Float32Array.BYTES_PER_ELEMENT);
    tempBuffers.releaseTempBuffer([1, 1, 1, floatCount], usage, "mse-reduction");
  };
  try {
    const forwardStart = performance.now();
    const normBuffer = await runNormalizeForwardBuffer(inputBuffer, payload.inputShape, payload.mean, payload.std);
    stepOwnedBuffers.push(normBuffer);
    const relu1Buffer = useFusedConvRelu ? await runConv2dReluForwardBuffer(normBuffer, payload.inputShape, persistent.conv1Weight.values, persistent.conv1Weight.shape, payload.conv1Bias) : await runReluForwardBuffer(await runConv2dForwardBuffer(normBuffer, payload.inputShape, persistent.conv1Weight.values, persistent.conv1Weight.shape, payload.conv1Bias), elementCount([1, 64, 16, 16]));
    const relu2Buffer = useFusedConvRelu ? await runConv2dReluForwardBuffer(relu1Buffer, [1, 64, 16, 16], persistent.conv2Weight.values, persistent.conv2Weight.shape, payload.conv2Bias) : await runReluForwardBuffer(await runConv2dForwardBuffer(relu1Buffer, [1, 64, 16, 16], persistent.conv2Weight.values, persistent.conv2Weight.shape, payload.conv2Bias), elementCount([1, 64, 16, 16]));
    const poolBuffer = await runMaxPool2dForwardBuffer(relu2Buffer, [1, 64, 16, 16]);
    const relu3Buffer = useFusedConvRelu ? await runConv2dReluForwardBuffer(poolBuffer, [1, 64, 8, 8], persistent.conv3Weight.values, persistent.conv3Weight.shape, payload.conv3Bias) : await runReluForwardBuffer(await runConv2dForwardBuffer(poolBuffer, [1, 64, 8, 8], persistent.conv3Weight.values, persistent.conv3Weight.shape, payload.conv3Bias), elementCount([1, 128, 8, 8]));
    stepOwnedBuffers.push(relu1Buffer, relu2Buffer, poolBuffer, relu3Buffer);
    const forwardMs = performance.now() - forwardStart;

    const relu1GramBuffer = await runGramMatrixBuffer(relu1Buffer, [1, 64, 16, 16]);
    const relu3GramBuffer = await runGramMatrixBuffer(relu3Buffer, [1, 128, 8, 8]);
    stepOwnedBuffers.push(relu1GramBuffer, relu3GramBuffer);

    const lossStart = performance.now();
    const styleLoss1 = (await runMseBuffer(device, acquireMseBuffer, releaseMseBuffer, relu1GramBuffer, persistent.styleGram1Buffer, 64 * 64, BUFFER_USAGE_STORAGE_COPY_SRC, BUFFER_USAGE_MAP_READ_COPY_DST, MAP_MODE_READ)) * payload.styleWeightConv1;
    const styleLoss3 = (await runMseBuffer(device, acquireMseBuffer, releaseMseBuffer, relu3GramBuffer, persistent.styleGram3Buffer, 128 * 128, BUFFER_USAGE_STORAGE_COPY_SRC, BUFFER_USAGE_MAP_READ_COPY_DST, MAP_MODE_READ)) * payload.styleWeightConv3;
    const contentLoss = (await runMseBuffer(device, acquireMseBuffer, releaseMseBuffer, relu2Buffer, persistent.contentRelu2Buffer, 64 * 16 * 16, BUFFER_USAGE_STORAGE_COPY_SRC, BUFFER_USAGE_MAP_READ_COPY_DST, MAP_MODE_READ)) * payload.contentWeight;
    const lossMs = performance.now() - lossStart;

    const backwardStart = performance.now();
    const gStyle1ReluBuffer = runScalarMulBuffer(device, await runStyleLossBackwardFromTargetGramBuffer(relu1Buffer, [1, 64, 16, 16], persistent.styleGram1Buffer), elementCount([1, 64, 16, 16]), payload.styleWeightConv1);
    const gStyle3ReluBuffer = runScalarMulBuffer(device, await runStyleLossBackwardFromTargetGramBuffer(relu3Buffer, [1, 128, 8, 8], persistent.styleGram3Buffer), elementCount([1, 128, 8, 8]), payload.styleWeightConv3);
    const gContentReluBuffer = runScalarMulBuffer(device, await runContentLossBackwardBuffer(relu2Buffer, persistent.contentRelu2Buffer, elementCount([1, 64, 16, 16])), elementCount([1, 64, 16, 16]), payload.contentWeight);
    stepOwnedBuffers.push(gStyle1ReluBuffer, gStyle3ReluBuffer, gContentReluBuffer);
    const gStyle3Buffer = await runReluBackwardBuffer(relu3Buffer, gStyle3ReluBuffer, elementCount([1, 128, 8, 8]));
    const gContentBuffer = await runReluBackwardBuffer(relu2Buffer, gContentReluBuffer, elementCount([1, 64, 16, 16]));
    const gStyle1Buffer = await runReluBackwardBuffer(relu1Buffer, gStyle1ReluBuffer, elementCount([1, 64, 16, 16]));
    const gPoolBuffer = await runConv2dBackwardInputBuffer(gStyle3Buffer, [1, 64, 8, 8], persistent.conv3Weight.values, persistent.conv3Weight.shape);
    const gRelu2Buffer = await runMaxPool2dBackwardBuffer(relu2Buffer, [1, 64, 16, 16], gPoolBuffer);
    const gConv2FromPoolBuffer = await runReluBackwardBuffer(relu2Buffer, gRelu2Buffer, elementCount([1, 64, 16, 16]));
    const gConv2TotalBuffer: GpuBufferRef = { buffer: runBinaryOpToBuffer(device, "add", gConv2FromPoolBuffer.buffer, gContentBuffer.buffer, elementCount([1, 64, 16, 16]), "tensorTensor"), owned: true };
    const gRelu1Buffer = await runConv2dBackwardInputBuffer(gConv2TotalBuffer, [1, 64, 16, 16], persistent.conv2Weight.values, persistent.conv2Weight.shape);
    const gConv1FromConv2Buffer = await runReluBackwardBuffer(relu1Buffer, gRelu1Buffer, elementCount([1, 64, 16, 16]));
    const gConv1TotalBuffer: GpuBufferRef = { buffer: runBinaryOpToBuffer(device, "add", gConv1FromConv2Buffer.buffer, gStyle1Buffer.buffer, elementCount([1, 64, 16, 16]), "tensorTensor"), owned: true };
    const gNormBuffer = await runConv2dBackwardInputBuffer(gConv1TotalBuffer, payload.inputShape, persistent.conv1Weight.values, persistent.conv1Weight.shape);
    const gInputBuffer = await runNormalizeBackwardBuffer(gNormBuffer, payload.inputShape, payload.std);
    stepOwnedBuffers.push(gStyle3Buffer, gContentBuffer, gStyle1Buffer, gPoolBuffer, gRelu2Buffer, gConv2FromPoolBuffer, gConv2TotalBuffer, gRelu1Buffer, gConv1FromConv2Buffer, gConv1TotalBuffer, gNormBuffer, gInputBuffer);
    const backwardMs = performance.now() - backwardStart;

    const updateStart = performance.now();
    let clampedInputBuffer: OwnedGpuBuffer;
    if (useFusedUpdateClamp) {
      clampedInputBuffer = runFusedUpdateClampBuffer(device, inputBuffer, gInputBuffer, elementCount(payload.inputShape), payload.learningRate);
    } else {
      const unfused = runUnfusedUpdateClampBuffer(device, inputBuffer, gInputBuffer, elementCount(payload.inputShape), payload.learningRate);
      stepOwnedBuffers.push(...unfused.intermediates);
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

    return { nextInputBuffer: clampedInputBuffer, totalLoss: styleLoss1 + styleLoss3 + contentLoss, forwardMs, lossMs, backwardMs, updateMs, diagnosticsReadbackMs };
  } finally {
    for (const ownedBuffer of stepOwnedBuffers) releaseOwnedBuffer(ownedBuffer);
  }
};
