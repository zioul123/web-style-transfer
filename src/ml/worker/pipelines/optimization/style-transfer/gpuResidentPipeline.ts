/// <reference lib="webworker" />

import { getGpuDevice } from "../../../runtime/deviceState";
import {
  readGpuBufferToArray,
  releaseOwnedBuffer,
  uploadToOwnedBuffer,
  type GpuBufferRef,
} from "../../../runtime/bufferKernels";
import {
  createOptimizationRuntimeContext,
  type OptimizationRuntimeContext,
} from "../../../runtime/optimizationContext";
import {
  createOptimizationTrackedOps,
} from "../trackedOps";
import { createInputOptimizer } from "../input-optimizer/createInputOptimizer";
import { createGpuVectorOps } from "../input-optimizer/gpuVectorOps";
import type { InputOptimizerConfig } from "../input-optimizer/types";
import { elementCount } from "./shapes";
import {
  runStyleTransferForward,
  runStyleTransferStep,
} from "./step";
import type {
  ConvLayerCacheEntry,
  StyleTransferPayload,
  StyleTransferRunResult,
} from "./types";

const parseConvLayerCache = (
  weights: StyleTransferPayload["weights"],
): Record<number, ConvLayerCacheEntry | undefined> => {
  const convLayerCache: Record<number, ConvLayerCacheEntry | undefined> = {};
  for (let layerIndex = 0; layerIndex <= 29; layerIndex += 1) {
    const weightShapeRaw = weights[`conv${layerIndex}.weightShape`];
    const weightValues = weights[`conv${layerIndex}.weightValues`];
    const biasValues = weights[`conv${layerIndex}.biasValues`];
    if (
      Array.isArray(weightShapeRaw) &&
      Array.isArray(weightValues) &&
      Array.isArray(biasValues)
    ) {
      convLayerCache[layerIndex] = {
        shape: weightShapeRaw as [number, number, number, number],
        values: new Float32Array(weightValues),
        bias: new Float32Array(biasValues),
      };
    }
  }
  return convLayerCache;
};

const releaseUnpersistedBuffers = (
  ownedBuffers: readonly GpuBufferRef[],
  persistentBuffers: readonly GpuBufferRef[],
): void => {
  const persistentBufferSet = new Set<GPUBuffer>(
    persistentBuffers.map((ref) => ref.buffer),
  );
  for (const ref of ownedBuffers) {
    if (!persistentBufferSet.has(ref.buffer)) releaseOwnedBuffer(ref);
  }
};

export const runStyleTransferGpuResident = async (
  payload: StyleTransferPayload,
): Promise<StyleTransferRunResult> => {
  const device = getGpuDevice();
  if (device === null) throw new Error("WebGPU device is not initialized.");

  const runtimeContext = createOptimizationRuntimeContext(device);
  const convLayerCache = parseConvLayerCache(payload.weights);
  const losses: number[] = [];
  let forwardMs = 0;
  let backwardMs = 0;
  let lossMs = 0;
  let updateMs = 0;
  let clampMs = 0;
  const totalStart = performance.now();
  const inputCount = elementCount(payload.inputShape);
  const optimizerConfig: InputOptimizerConfig = {
    optimizer: payload.optimizer ?? "sgd",
    count: inputCount,
    learningRate: payload.learningRate,
    adamBeta1: payload.adamBeta1 ?? 0.9,
    adamBeta2: payload.adamBeta2 ?? 0.999,
    adamEpsilon: payload.adamEpsilon ?? 1e-8,
    lbfgsMemory: payload.lbfgsMemory ?? 10,
    lbfgsEpsilon: payload.lbfgsEpsilon ?? 1e-8,
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
    const targetOwnedBuffers: GpuBufferRef[] = [];
    const targetContext: OptimizationRuntimeContext = {
      acquireTemp: runtimeContext.acquireTemp,
      trackOwned: <T extends GpuBufferRef>(...refs: T[]): T => {
        targetOwnedBuffers.push(...refs);
        return refs[refs.length - 1];
      },
      releaseStepOwned: (): void => {
        releaseUnpersistedBuffers(targetOwnedBuffers, persistentBuffers);
        targetOwnedBuffers.length = 0;
      },
      disposeAll: (): void => undefined,
    };
    const targetTracked = createOptimizationTrackedOps(device, targetContext);
    const contentNorm = await targetTracked.normalizeForward(
      contentImageBuffer,
      payload.inputShape,
      payload.mean,
      payload.std,
    );
    const styleNorm = await targetTracked.normalizeForward(
      styleImageBuffer,
      payload.inputShape,
      payload.mean,
      payload.std,
    );
    const contentTargets = await runStyleTransferForward(
      payload,
      convLayerCache,
      targetTracked,
      contentNorm,
    );
    const styleTargets = await runStyleTransferForward(
      payload,
      convLayerCache,
      targetTracked,
      styleNorm,
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
        clampMs += stepResult.updateMs;
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
