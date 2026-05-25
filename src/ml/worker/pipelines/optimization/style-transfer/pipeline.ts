/// <reference lib="webworker" />

import { getGpuDevice } from "../../../runtime/deviceState";
import {
  readGpuBufferToArray,
  releaseOwnedBuffer,
  uploadToOwnedBuffer,
  type GpuBufferRef,
} from "../../../runtime/bufferKernels";
import { createOptimizationRuntimeContext } from "../../../runtime/optimizationContext";
import {
  createOptimizationTrackedOps,
} from "../trackedOps";
import { createInputOptimizer } from "../input-optimizer/createInputOptimizer";
import { createGpuVectorOps } from "../input-optimizer/gpuVectorOps";
import type { InputOptimizerConfig } from "../input-optimizer/types";
import { elementCount, type TensorShape4D } from "../../../runtime/tensorShapes";
import { parseVgg19ConvLayerCache } from "../../../models/vgg19/weights";
import { createPersistentTargetContext } from "./persistentTargetContext";
import {
  runStyleTransferForward,
  runStyleTransferStep,
} from "./step";
import type {
  StyleTransferPayload,
  StyleTransferRunResult,
} from "./types";

const shapesMatch = (a: TensorShape4D, b: TensorShape4D): boolean =>
  a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];

const assertValuesMatchShape = (
  label: string,
  values: readonly number[],
  shape: TensorShape4D,
): void => {
  const expectedCount = elementCount(shape);
  if (values.length !== expectedCount) {
    throw new Error(
      `${label} has ${values.length} values, expected ${expectedCount} for shape [${shape.join(", ")}].`,
    );
  }
};

export const runStyleTransferPipeline = async (
  payload: StyleTransferPayload,
): Promise<StyleTransferRunResult> => {
  const device = getGpuDevice();
  if (device === null) throw new Error("WebGPU device is not initialized.");

  const contentShape: TensorShape4D = payload.contentShape ?? payload.inputShape;
  const styleShape: TensorShape4D = payload.styleShape ?? payload.inputShape;
  if (!shapesMatch(payload.inputShape, contentShape)) {
    throw new Error(
      "Style transfer expects the optimized input shape to match the content image shape.",
    );
  }
  assertValuesMatchShape("inputImageValues", payload.inputImageValues, payload.inputShape);
  assertValuesMatchShape("contentImageValues", payload.contentImageValues, contentShape);
  assertValuesMatchShape("styleImageValues", payload.styleImageValues, styleShape);

  const runtimeContext = createOptimizationRuntimeContext(device);
  const convLayerCache = parseVgg19ConvLayerCache(payload.weights);
  const losses: number[] = [];
  let forwardMs = 0;
  let backwardMs = 0;
  let lossMs = 0;
  let updateMs = 0;
  const clampMs = 0;
  const totalStart = performance.now();
  const inputCount = elementCount(payload.inputShape);
  const optimizerConfig: InputOptimizerConfig = {
    optimizer: payload.optimizer ?? "lbfgs",
    count: inputCount,
    learningRate: payload.learningRate,
    adamBeta1: payload.adamBeta1 ?? 0.9,
    adamBeta2: payload.adamBeta2 ?? 0.999,
    adamEpsilon: payload.adamEpsilon ?? 1e-8,
    lbfgsMemory: payload.lbfgsMemory ?? 100,
    lbfgsEpsilon: payload.lbfgsEpsilon ?? 1e-9,
  };
  const inputOptimizer = createInputOptimizer(
    optimizerConfig,
    createGpuVectorOps(device, inputCount),
  );

  let inputBuffer: GpuBufferRef | null = uploadToOwnedBuffer(
    device,
    new Float32Array(payload.inputImageValues),
  );
  const contentImageBuffer = uploadToOwnedBuffer(
    device,
    new Float32Array(payload.contentImageValues),
  );
  const styleImageBuffer = uploadToOwnedBuffer(
    device,
    new Float32Array(payload.styleImageValues),
  );
  const persistentBuffers: GpuBufferRef[] = [contentImageBuffer, styleImageBuffer];

  try {
    const targetContext = createPersistentTargetContext(
      runtimeContext,
      persistentBuffers,
    );
    const targetTracked = createOptimizationTrackedOps(device, targetContext);
    const contentNorm = await targetTracked.normalizeForward(
      contentImageBuffer,
      contentShape,
      payload.mean,
      payload.std,
    );
    const styleNorm = await targetTracked.normalizeForward(
      styleImageBuffer,
      styleShape,
      payload.mean,
      payload.std,
    );
    const contentTargets = await runStyleTransferForward(
      convLayerCache,
      targetTracked,
      contentNorm,
      contentShape,
    );
    const styleTargets = await runStyleTransferForward(
      convLayerCache,
      targetTracked,
      styleNorm,
      styleShape,
    );
    const contentTargetRelu = contentTargets.reluOut[payload.contentLayerIndex];
    if (contentTargetRelu === undefined) {
      throw new Error(
        `Missing precomputed content activation for ReLU layer index ${payload.contentLayerIndex}.`,
      );
    }
    persistentBuffers.push(contentTargetRelu);
    const styleGramTargets: Record<number, GpuBufferRef | undefined> = {};
    for (const reluLayerIndex of payload.styleLayerIndices) {
      const styleRelu = styleTargets.reluOut[reluLayerIndex];
      const styleShape = styleTargets.reluShape[reluLayerIndex];
      if (styleRelu === undefined || styleShape === undefined) {
        throw new Error(
          `Missing precomputed style activation for ReLU layer index ${reluLayerIndex}.`,
        );
      }
      const gram = await targetTracked.gram(styleRelu, styleShape);
      styleGramTargets[reluLayerIndex] = gram;
      persistentBuffers.push(gram);
    }
    targetContext.releaseStepOwned();

    for (let step = 0; step < payload.steps; step += 1) {
      if (inputBuffer === null) throw new Error("Input buffer was released unexpectedly.");
      const tracked = createOptimizationTrackedOps(device, runtimeContext);
      try {
        const stepResult = await runStyleTransferStep(
          device,
          payload,
          convLayerCache,
          { contentTargetRelu, styleGramTargets },
          inputBuffer,
          tracked,
          inputOptimizer,
          step,
        );
        losses.push(stepResult.totalLoss);
        forwardMs += stepResult.forwardMs;
        lossMs += stepResult.lossMs;
        backwardMs += stepResult.backwardMs;
        updateMs += stepResult.updateMs;
        const previousInput = inputBuffer;
        inputBuffer = stepResult.nextInputBuffer;
        releaseOwnedBuffer(previousInput);
      } finally {
        runtimeContext.releaseStepOwned();
      }
    }

    if (inputBuffer === null) throw new Error("Input buffer was released unexpectedly.");
    const finalValues = await readGpuBufferToArray(
      device,
      inputBuffer.buffer,
      elementCount(payload.inputShape),
    );
    const elapsedMs = performance.now() - totalStart;
    return {
      losses,
      finalValues: Array.from(finalValues),
      stats: {
        elapsedMs,
        avgStepMs: payload.steps === 0 ? 0 : elapsedMs / payload.steps,
        forwardMs,
        backwardMs,
        lossMs,
        updateMs,
        clampMs,
        steps: payload.steps,
      },
    };
  } finally {
    if (inputBuffer !== null) releaseOwnedBuffer(inputBuffer);
    for (const buffer of persistentBuffers) releaseOwnedBuffer(buffer);
    inputOptimizer.dispose();
    runtimeContext.disposeAll();
  }
};
