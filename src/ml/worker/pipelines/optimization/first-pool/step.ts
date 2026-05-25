import type { WorkerRequest } from "../../../../../types";
import { readGpuBufferToArray, type GpuBufferRef, type OwnedGpuBuffer } from "../../../runtime/bufferKernels";
import type { FirstPoolPersistentContext, FirstPoolRuntimeContext, FirstPoolStepResult, TensorShape4D } from "./types";
import { createOptimizationTrackedOps } from "../trackedOps";
import { convOutputShape, elementCount, pooledShape } from "../../../runtime/tensorShapes";

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
  const tracked = createOptimizationTrackedOps(device, runtimeContext);
  try {
    const relu1Shape = convOutputShape(payload.inputShape, persistent.conv1Weight.shape[0]);
    const relu2Shape = convOutputShape(relu1Shape, persistent.conv2Weight.shape[0]);
    const poolShape = pooledShape(relu2Shape);
    const relu3Shape = convOutputShape(poolShape, persistent.conv3Weight.shape[0]);
    const relu1GramCount = relu1Shape[1] * relu1Shape[1];
    const relu3GramCount = relu3Shape[1] * relu3Shape[1];

    const forwardStart = performance.now();
    const normBuffer = await tracked.normalizeForward(inputBuffer, payload.inputShape, payload.mean, payload.std);
    const relu1Buffer = useFusedConvRelu ? await tracked.conv2dReluForward(normBuffer, payload.inputShape, persistent.conv1Weight.values, persistent.conv1Weight.shape, payload.conv1Bias) : await tracked.reluForward(await tracked.conv2dForward(normBuffer, payload.inputShape, persistent.conv1Weight.values, persistent.conv1Weight.shape, payload.conv1Bias), elementCount(relu1Shape));
    const relu2Buffer = useFusedConvRelu ? await tracked.conv2dReluForward(relu1Buffer, relu1Shape, persistent.conv2Weight.values, persistent.conv2Weight.shape, payload.conv2Bias) : await tracked.reluForward(await tracked.conv2dForward(relu1Buffer, relu1Shape, persistent.conv2Weight.values, persistent.conv2Weight.shape, payload.conv2Bias), elementCount(relu2Shape));
    const poolBuffer = await tracked.maxPoolForward(relu2Buffer, relu2Shape);
    const relu3Buffer = useFusedConvRelu ? await tracked.conv2dReluForward(poolBuffer, poolShape, persistent.conv3Weight.values, persistent.conv3Weight.shape, payload.conv3Bias) : await tracked.reluForward(await tracked.conv2dForward(poolBuffer, poolShape, persistent.conv3Weight.values, persistent.conv3Weight.shape, payload.conv3Bias), elementCount(relu3Shape));
    const forwardMs = performance.now() - forwardStart;

    const relu1GramBuffer = await tracked.gram(relu1Buffer, relu1Shape);
    const relu3GramBuffer = await tracked.gram(relu3Buffer, relu3Shape);

    const lossStart = performance.now();
    const styleLoss1Buffer = await tracked.mseScalarBuffer(relu1GramBuffer, persistent.styleGram1Buffer, relu1GramCount);
    const styleLoss3Buffer = await tracked.mseScalarBuffer(relu3GramBuffer, persistent.styleGram3Buffer, relu3GramCount);
    const contentLossBuffer = await tracked.mseScalarBuffer(relu2Buffer, persistent.contentRelu2Buffer, elementCount(relu2Shape));
    const weightedStyle1LossBuffer = tracked.scalarMul(styleLoss1Buffer, 1, payload.styleWeightConv1 / relu1GramCount);
    const weightedStyle3LossBuffer = tracked.scalarMul(styleLoss3Buffer, 1, payload.styleWeightConv3 / relu3GramCount);
    const weightedContentLossBuffer = tracked.scalarMul(contentLossBuffer, 1, payload.contentWeight / elementCount(relu2Shape));
    const styleLossTotalBuffer = tracked.add(weightedStyle1LossBuffer, weightedStyle3LossBuffer, 1);
    const totalLossBuffer = tracked.add(styleLossTotalBuffer, weightedContentLossBuffer, 1);
    const totalLoss = (await readGpuBufferToArray(device, totalLossBuffer.buffer, 1))[0];
    const lossMs = performance.now() - lossStart;

    const backwardStart = performance.now();
    const gStyle1ReluBuffer = tracked.scalarMul(await tracked.styleLossBackward(relu1Buffer, relu1Shape, persistent.styleGram1Buffer), elementCount(relu1Shape), payload.styleWeightConv1);
    const gStyle3ReluBuffer = tracked.scalarMul(await tracked.styleLossBackward(relu3Buffer, relu3Shape, persistent.styleGram3Buffer), elementCount(relu3Shape), payload.styleWeightConv3);
    const gContentReluBuffer = tracked.scalarMul(await tracked.contentLossBackward(relu2Buffer, persistent.contentRelu2Buffer, elementCount(relu2Shape)), elementCount(relu2Shape), payload.contentWeight);
    const gStyle3Buffer = await tracked.reluBackward(relu3Buffer, gStyle3ReluBuffer, elementCount(relu3Shape));
    const gContentBuffer = await tracked.reluBackward(relu2Buffer, gContentReluBuffer, elementCount(relu2Shape));
    const gStyle1Buffer = await tracked.reluBackward(relu1Buffer, gStyle1ReluBuffer, elementCount(relu1Shape));
    const gPoolBuffer = await tracked.conv2dBackwardInput(gStyle3Buffer, poolShape, persistent.conv3Weight.values, persistent.conv3Weight.shape);
    const gRelu2Buffer = await tracked.maxPoolBackward(relu2Buffer, relu2Shape, gPoolBuffer);
    const gConv2FromPoolBuffer = await tracked.reluBackward(relu2Buffer, gRelu2Buffer, elementCount(relu2Shape));
    const gConv2TotalBuffer = tracked.add(gConv2FromPoolBuffer, gContentBuffer, elementCount(relu2Shape));
    const gRelu1Buffer = await tracked.conv2dBackwardInput(gConv2TotalBuffer, relu1Shape, persistent.conv2Weight.values, persistent.conv2Weight.shape);
    const gConv1FromConv2Buffer = await tracked.reluBackward(relu1Buffer, gRelu1Buffer, elementCount(relu1Shape));
    const gConv1TotalBuffer = tracked.add(gConv1FromConv2Buffer, gStyle1Buffer, elementCount(relu1Shape));
    const gNormBuffer = await tracked.conv2dBackwardInput(gConv1TotalBuffer, payload.inputShape, persistent.conv1Weight.values, persistent.conv1Weight.shape);
    const gInputBuffer = await tracked.normalizeBackward(gNormBuffer, payload.inputShape, payload.std);
    const backwardMs = performance.now() - backwardStart;

    const updateStart = performance.now();
    const clampedInputBuffer: OwnedGpuBuffer = tracked.updateClamp(
      inputBuffer,
      gInputBuffer,
      elementCount(payload.inputShape),
      payload.learningRate,
      useFusedUpdateClamp,
    );
    const updateMs = performance.now() - updateStart;

    let diagnosticsReadbackMs = 0;
    assertShape(elementCount(relu1Shape), relu1Shape, "relu1");
    assertShape(elementCount(relu2Shape), relu2Shape, "relu2");
    assertShape(elementCount(relu3Shape), relu3Shape, "relu3");
    assertShape(elementCount(payload.inputShape), payload.inputShape, "gInput");

    if (debugValidateStepShapes) {
      const debugShapeReadStart = performance.now();
      const relu1 = await readGpuBufferToArray(device, relu1Buffer.buffer, elementCount(relu1Shape));
      const relu2 = await readGpuBufferToArray(device, relu2Buffer.buffer, elementCount(relu2Shape));
      const relu3 = await readGpuBufferToArray(device, relu3Buffer.buffer, elementCount(relu3Shape));
      diagnosticsReadbackMs += performance.now() - debugShapeReadStart;
      assertShape(relu1.length, relu1Shape, "relu1");
      assertShape(relu2.length, relu2Shape, "relu2");
      assertShape(relu3.length, relu3Shape, "relu3");
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
