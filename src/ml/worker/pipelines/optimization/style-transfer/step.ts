import type { GpuBufferRef } from "../../../runtime/bufferKernels";
import { readGpuBufferToArray } from "../../../runtime/bufferKernels";
import type { InputOptimizer } from "../input-optimizer/types";
import { POOL_LAYERS, RELU_LAYERS } from "../layerSchedules";
import type { OptimizationTrackedOps } from "../trackedOps";
import { convOutputShape, elementCount, gramElementCount, pooledShape, type TensorShape4D } from "../../../runtime/tensorShapes";
import type {
  ConvLayerCacheEntry,
  StyleTransferForwardResult,
  StyleTransferPayload,
  StyleTransferPersistentContext,
  StyleTransferStepResult,
} from "./types";
import { VGG19_STYLE_TRANSFER_PLAN } from "./vgg19Plan";

type ConvLayerCache = Record<number, ConvLayerCacheEntry | undefined>;

const addReluGradient = (
  tracked: OptimizationTrackedOps,
  gradByReluLayer: Record<number, GpuBufferRef | undefined>,
  reluLayerIndex: number,
  incomingGrad: GpuBufferRef,
  count: number,
): void => {
  const existingGrad = gradByReluLayer[reluLayerIndex];
  gradByReluLayer[reluLayerIndex] =
    existingGrad === undefined
      ? incomingGrad
      : tracked.add(existingGrad, incomingGrad, count);
};

export const runStyleTransferForward = async (
  payload: StyleTransferPayload,
  convLayerCache: ConvLayerCache,
  tracked: OptimizationTrackedOps,
  startBuffer: GpuBufferRef,
): Promise<StyleTransferForwardResult> => {
  let shape: TensorShape4D = payload.inputShape;
  let values: GpuBufferRef = startBuffer;
  const reluOut: Record<number, GpuBufferRef | undefined> = {};
  const reluShape: Record<number, TensorShape4D | undefined> = {};
  const convInShape: Record<number, TensorShape4D | undefined> = {};
  const layerInput: Record<number, GpuBufferRef | undefined> = {};
  const layerInputShape: Record<number, TensorShape4D | undefined> = {};

  for (const planStep of VGG19_STYLE_TRANSFER_PLAN) {
    if (planStep.kind === "pool") {
      layerInput[planStep.poolLayerIndex] = values;
      layerInputShape[planStep.poolLayerIndex] = shape;
      values = await tracked.maxPoolForward(values, shape);
      shape = pooledShape(shape);
      continue;
    }

    layerInput[planStep.convLayerIndex] = values;
    layerInputShape[planStep.convLayerIndex] = shape;
    const convLayer = convLayerCache[planStep.convLayerIndex];
    if (convLayer === undefined) continue;

    convInShape[planStep.convLayerIndex] = shape;
    values = await tracked.conv2dReluForward(
      values,
      shape,
      convLayer.values,
      convLayer.shape,
      convLayer.bias,
    );

    shape = convOutputShape(shape, convLayer.shape[0]);
    reluOut[planStep.reluLayerIndex] = values;
    reluShape[planStep.reluLayerIndex] = shape;
  }

  return {
    reluOut,
    reluShape,
    convInShape,
    layerInput,
    layerInputShape,
    final: values,
  };
};

const computeStepLoss = async (
  device: GPUDevice,
  payload: StyleTransferPayload,
  tracked: OptimizationTrackedOps,
  run: StyleTransferForwardResult,
  persistent: StyleTransferPersistentContext,
): Promise<number> => {
  const scalarLossBuffers: GpuBufferRef[] = [];
  for (const reluLayerIndex of payload.styleLayerIndices) {
    const inputRelu = run.reluOut[reluLayerIndex];
    const reluShape = run.reluShape[reluLayerIndex];
    const targetGram = persistent.styleGramTargets[reluLayerIndex];
    if (inputRelu === undefined) {
      throw new Error(`Missing ReLU activation for style layer index ${reluLayerIndex}.`);
    }
    if (reluShape === undefined) {
      throw new Error(`Unsupported ReLU style layer index ${reluLayerIndex}.`);
    }
    if (targetGram === undefined) {
      throw new Error(
        `Missing precomputed style gram target for ReLU layer index ${reluLayerIndex}.`,
      );
    }

    const inputGram = await tracked.gram(inputRelu, reluShape);
    const styleSse = await tracked.mseScalarBuffer(
      inputGram,
      targetGram,
      gramElementCount(reluShape),
    );
    scalarLossBuffers.push(
      tracked.scalarMul(
        styleSse,
        1,
        payload.styleWeight / gramElementCount(reluShape),
      ),
    );
  }

  const contentRelu = run.reluOut[payload.contentLayerIndex];
  const contentReluShape = run.reluShape[payload.contentLayerIndex];
  if (contentRelu === undefined || contentReluShape === undefined) {
    throw new Error(
      `Missing ReLU activation for content layer index ${payload.contentLayerIndex}.`,
    );
  }
  const contentSse = await tracked.mseScalarBuffer(
    contentRelu,
    persistent.contentTargetRelu,
    elementCount(contentReluShape),
  );
  scalarLossBuffers.push(
    tracked.scalarMul(
      contentSse,
      1,
      payload.contentWeight / elementCount(contentReluShape),
    ),
  );

  let totalLossBuffer = scalarLossBuffers[0];
  if (totalLossBuffer === undefined) throw new Error("No loss terms were produced.");
  for (let i = 1; i < scalarLossBuffers.length; i += 1) {
    totalLossBuffer = tracked.add(totalLossBuffer, scalarLossBuffers[i], 1);
  }
  return (await readGpuBufferToArray(device, totalLossBuffer.buffer, 1))[0];
};

const seedReluGradients = async (
  payload: StyleTransferPayload,
  tracked: OptimizationTrackedOps,
  run: StyleTransferForwardResult,
  persistent: StyleTransferPersistentContext,
): Promise<Record<number, GpuBufferRef | undefined>> => {
  const gradByReluLayer: Record<number, GpuBufferRef | undefined> = {};
  for (const reluLayerIndex of payload.styleLayerIndices) {
    const inputRelu = run.reluOut[reluLayerIndex];
    const reluShape = run.reluShape[reluLayerIndex];
    const targetGram = persistent.styleGramTargets[reluLayerIndex];
    if (inputRelu === undefined) {
      throw new Error(
        `Missing ReLU activation for style gradient layer index ${reluLayerIndex}.`,
      );
    }
    if (reluShape === undefined) {
      throw new Error(`Unsupported ReLU style gradient layer index ${reluLayerIndex}.`);
    }
    if (targetGram === undefined) {
      throw new Error(
        `Missing precomputed style gram gradient target for ReLU layer index ${reluLayerIndex}.`,
      );
    }

    const styleGrad = await tracked.styleLossBackward(inputRelu, reluShape, targetGram);
    const weightedStyleGrad =
      payload.styleWeight === 1
        ? styleGrad
        : tracked.scalarMul(styleGrad, elementCount(reluShape), payload.styleWeight);
    addReluGradient(
      tracked,
      gradByReluLayer,
      reluLayerIndex,
      weightedStyleGrad,
      elementCount(reluShape),
    );
  }

  const contentRelu = run.reluOut[payload.contentLayerIndex];
  const contentReluShape = run.reluShape[payload.contentLayerIndex];
  if (contentRelu === undefined || contentReluShape === undefined) {
    throw new Error(
      `Missing ReLU activation for content layer index ${payload.contentLayerIndex}.`,
    );
  }
  const contentGrad = await tracked.contentLossBackward(
    contentRelu,
    persistent.contentTargetRelu,
    elementCount(contentReluShape),
  );
  const weightedContentGrad =
    payload.contentWeight === 1
      ? contentGrad
      : tracked.scalarMul(contentGrad, elementCount(contentReluShape), payload.contentWeight);
  addReluGradient(
    tracked,
    gradByReluLayer,
    payload.contentLayerIndex,
    weightedContentGrad,
    elementCount(contentReluShape),
  );
  return gradByReluLayer;
};

const backpropagateToInput = async (
  payload: StyleTransferPayload,
  convLayerCache: ConvLayerCache,
  tracked: OptimizationTrackedOps,
  run: StyleTransferForwardResult,
  gradByReluLayer: Record<number, GpuBufferRef | undefined>,
): Promise<GpuBufferRef> => {
  let grad: GpuBufferRef | null = null;
  const poolLayerSet = new Set(POOL_LAYERS);
  const reluLayerSet = new Set(RELU_LAYERS);
  for (let layerIndex = 29; layerIndex >= 0; layerIndex -= 1) {
    if (poolLayerSet.has(layerIndex)) {
      const poolInput = run.layerInput[layerIndex];
      const poolInputShape = run.layerInputShape[layerIndex];
      if (grad !== null && poolInput !== undefined && poolInputShape !== undefined) {
        grad = await tracked.maxPoolBackward(poolInput, poolInputShape, grad);
      }
    }

    if (reluLayerSet.has(layerIndex)) {
      const tapGrad = gradByReluLayer[layerIndex];
      const reluOut = run.reluOut[layerIndex];
      const reluShape = run.reluShape[layerIndex];
      if (tapGrad !== undefined) {
        if (reluShape === undefined) {
          throw new Error(`Missing ReLU shape for gradient layer index ${layerIndex}.`);
        }
        grad =
          grad === null ? tapGrad : tracked.add(grad, tapGrad, elementCount(reluShape));
      }
      if (grad !== null && reluOut !== undefined && reluShape !== undefined) {
        grad = await tracked.reluBackward(reluOut, grad, elementCount(reluShape));
      }
    }

    const convLayer = convLayerCache[layerIndex];
    const convInShape = run.convInShape[layerIndex];
    if (convLayer !== undefined && convInShape !== undefined && grad !== null) {
      grad = await tracked.conv2dBackwardInput(
        grad,
        convInShape,
        convLayer.values,
        convLayer.shape,
      );
    }
  }

  if (grad === null) throw new Error("No gradient path reached the input image.");
  return tracked.normalizeBackward(grad, payload.inputShape, payload.std);
};

export const runStyleTransferStep = async (
  device: GPUDevice,
  payload: StyleTransferPayload,
  convLayerCache: ConvLayerCache,
  persistent: StyleTransferPersistentContext,
  inputBuffer: GpuBufferRef,
  tracked: OptimizationTrackedOps,
  inputOptimizer: InputOptimizer<GpuBufferRef>,
  step: number,
): Promise<StyleTransferStepResult> => {
  const forwardStart = performance.now();
  const norm = await tracked.normalizeForward(
    inputBuffer,
    payload.inputShape,
    payload.mean,
    payload.std,
  );
  const run = await runStyleTransferForward(payload, convLayerCache, tracked, norm);
  const forwardMs = performance.now() - forwardStart;

  const lossStart = performance.now();
  const totalLoss = await computeStepLoss(device, payload, tracked, run, persistent);
  const lossMs = performance.now() - lossStart;

  const backwardStart = performance.now();
  const gradByReluLayer = await seedReluGradients(payload, tracked, run, persistent);
  const gInput = await backpropagateToInput(
    payload,
    convLayerCache,
    tracked,
    run,
    gradByReluLayer,
  );
  const backwardMs = performance.now() - backwardStart;

  const updateStart = performance.now();
  const nextInputBuffer = await inputOptimizer.step(inputBuffer, gInput, step);
  const updateMs = performance.now() - updateStart;

  return { nextInputBuffer, totalLoss, forwardMs, lossMs, backwardMs, updateMs };
};
