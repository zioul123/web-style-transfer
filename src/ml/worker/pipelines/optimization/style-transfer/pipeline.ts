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
  type OptimizationPersistentKernelResources,
} from "../trackedOps";
import { createInputOptimizer } from "../input-optimizer/createInputOptimizer";
import { createGpuVectorOps } from "../input-optimizer/gpuVectorOps";
import type { InputOptimizer, InputOptimizerConfig } from "../input-optimizer/types";
import { elementCount, type TensorShape4D } from "../../../runtime/tensorShapes";
import { parseVgg19ConvLayerCache } from "../../../models/vgg19/weights";
import { createPersistentTargetContext } from "./persistentTargetContext";
import {
  runStyleTransferForward,
  runStyleTransferStep,
} from "./step";
import type { WorkerKernelOptimizationFlags, WorkerKernelStats } from "../../../../../types";
import { BUFFER_USAGE_STORAGE_COPY_DST } from "../../../runtime/gpuFlags";
import type {
  StyleTransferPayload,
  StyleTransferRunResult,
} from "./types";

type OptimizerSession = {
  key: string;
  stepOffset: number;
  inputOptimizer: InputOptimizer<GpuBufferRef>;
};

const optimizerSessionById = new Map<string, OptimizerSession>();

const buildOptimizerKey = (
  payload: StyleTransferPayload,
  inputCount: number,
): string =>
  JSON.stringify({
    optimizer: payload.optimizer ?? "lbfgs",
    count: inputCount,
    learningRate: payload.learningRate,
    adamBeta1: payload.adamBeta1 ?? 0.9,
    adamBeta2: payload.adamBeta2 ?? 0.999,
    adamEpsilon: payload.adamEpsilon ?? 1e-8,
    lbfgsMemory: payload.lbfgsMemory ?? 100,
    lbfgsEpsilon: payload.lbfgsEpsilon ?? 1e-9,
  });

const createSessionOptimizer = (
  device: GPUDevice,
  inputCount: number,
  optimizerConfig: InputOptimizerConfig,
): InputOptimizer<GpuBufferRef> =>
  createInputOptimizer(optimizerConfig, createGpuVectorOps(device, inputCount));

const normalizeLossReadbackInterval = (interval: number | undefined): number => {
  if (interval === undefined) return 1;
  if (!Number.isFinite(interval)) return 1;
  return Math.max(1, Math.floor(interval));
};

const getOrCreateOptimizer = (
  payload: StyleTransferPayload,
  device: GPUDevice,
  inputCount: number,
  optimizerConfig: InputOptimizerConfig,
): {
  inputOptimizer: InputOptimizer<GpuBufferRef>;
  sessionId: string | null;
  stepOffset: number;
} => {
  const sessionId = payload.sessionId ?? null;
  if (sessionId === null) {
    return {
      inputOptimizer: createSessionOptimizer(device, inputCount, optimizerConfig),
      sessionId: null,
      stepOffset: 0,
    };
  }

  const key = buildOptimizerKey(payload, inputCount);
  const existing = optimizerSessionById.get(sessionId);
  if (existing === undefined || existing.key !== key) {
    if (existing !== undefined) existing.inputOptimizer.dispose();
    const inputOptimizer = createSessionOptimizer(device, inputCount, optimizerConfig);
    optimizerSessionById.set(sessionId, {
      key,
      stepOffset: 0,
      inputOptimizer,
    });
    return { inputOptimizer, sessionId, stepOffset: 0 };
  }

  return {
    inputOptimizer: existing.inputOptimizer,
    sessionId,
    stepOffset: existing.stepOffset,
  };
};

export const clearStyleTransferOptimizerSession = (sessionId: string): void => {
  const session = optimizerSessionById.get(sessionId);
  if (session === undefined) return;
  session.inputOptimizer.dispose();
  optimizerSessionById.delete(sessionId);
};

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

const assertSupportedKernelFlags = (
  kernelFlags: WorkerKernelOptimizationFlags | undefined,
): void => {
  if (kernelFlags === undefined) return;
  if (kernelFlags.usePersistentWeightBuffers === false) {
    throw new Error("kernelFlags.usePersistentWeightBuffers=false is not supported.");
  }
  if (kernelFlags.useStepBufferPool === false) {
    throw new Error("kernelFlags.useStepBufferPool=false is not supported.");
  }
  const unsupported: Array<[string, unknown]> = [
    ["useVec4Pointwise", kernelFlags.useVec4Pointwise],
    ["usePoolBackwardScatter", kernelFlags.usePoolBackwardScatter],
    ["gramKernel", kernelFlags.gramKernel],
    ["styleBackward", kernelFlags.styleBackward],
    ["convForwardKernel", kernelFlags.convForwardKernel],
    ["convBackwardInputKernel", kernelFlags.convBackwardInputKernel],
    ["weightStorage", kernelFlags.weightStorage],
  ];
  for (const [name, value] of unsupported) {
    if (value !== undefined) {
      throw new Error(`kernelFlags.${name}=${String(value)} is not implemented yet.`);
    }
  }
};

const mergeKernelStats = (target: WorkerKernelStats, source: WorkerKernelStats): void => {
  for (const [family, stat] of Object.entries(source)) {
    if (stat === undefined) continue;
    const key = family as keyof WorkerKernelStats;
    const existing = target[key];
    if (existing === undefined) target[key] = { ...stat };
    else {
      existing.calls += stat.calls;
      existing.elapsedMs += stat.elapsedMs;
    }
  }
};

export const runStyleTransferPipeline = async (
  payload: StyleTransferPayload,
): Promise<StyleTransferRunResult> => {
  const device = getGpuDevice();
  if (device === null) throw new Error("WebGPU device is not initialized.");
  assertSupportedKernelFlags(payload.kernelFlags);

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
  const optimizerRunContext = getOrCreateOptimizer(
    payload,
    device,
    inputCount,
    optimizerConfig,
  );
  const { inputOptimizer, sessionId, stepOffset } = optimizerRunContext;
  const lossReadbackInterval = normalizeLossReadbackInterval(
    payload.lossReadbackInterval,
  );
  const synchronizePhaseTimings = payload.synchronizePhaseTimings ?? false;
  const collectKernelStats = payload.collectKernelStats ?? false;
  const aggregatedKernelStats: WorkerKernelStats = {};
  let weightUploadMs = 0;
  const persistentWeightBuffers: GPUBuffer[] = [];
  const persistentConvWeightBuffers = new Map<Float32Array, GPUBuffer>();
  const persistentConvBiasBuffers = new Map<Float32Array, GPUBuffer>();
  const persistentKernelResources: OptimizationPersistentKernelResources = {
    getConvForwardStaticBuffers: (weight: Float32Array, bias: Float32Array) => {
      let weightBuffer = persistentConvWeightBuffers.get(weight);
      if (weightBuffer === undefined) {
        const uploadStart = performance.now();
        weightBuffer = device.createBuffer({
          size: weight.byteLength,
          usage: BUFFER_USAGE_STORAGE_COPY_DST,
        });
        device.queue.writeBuffer(weightBuffer, 0, weight);
        persistentConvWeightBuffers.set(weight, weightBuffer);
        persistentWeightBuffers.push(weightBuffer);
        weightUploadMs += performance.now() - uploadStart;
      }
      let biasBuffer = persistentConvBiasBuffers.get(bias);
      if (biasBuffer === undefined) {
        const uploadStart = performance.now();
        biasBuffer = device.createBuffer({
          size: bias.byteLength,
          usage: BUFFER_USAGE_STORAGE_COPY_DST,
        });
        device.queue.writeBuffer(biasBuffer, 0, bias);
        persistentConvBiasBuffers.set(bias, biasBuffer);
        persistentWeightBuffers.push(biasBuffer);
        weightUploadMs += performance.now() - uploadStart;
      }
      return { weightBuffer, biasBuffer };
    },
    getConvBackwardStaticBuffers: (weight: Float32Array) => {
      let weightBuffer = persistentConvWeightBuffers.get(weight);
      if (weightBuffer === undefined) {
        const uploadStart = performance.now();
        weightBuffer = device.createBuffer({
          size: weight.byteLength,
          usage: BUFFER_USAGE_STORAGE_COPY_DST,
        });
        device.queue.writeBuffer(weightBuffer, 0, weight);
        persistentConvWeightBuffers.set(weight, weightBuffer);
        persistentWeightBuffers.push(weightBuffer);
        weightUploadMs += performance.now() - uploadStart;
      }
      return { weightBuffer };
    },
  };

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
    const targetTracked = createOptimizationTrackedOps(
      device,
      targetContext,
      payload.kernelFlags,
      collectKernelStats,
      payload.kernelFlags?.usePersistentWeightBuffers === true
        ? persistentKernelResources
        : undefined,
    );
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
    if (collectKernelStats) {
      mergeKernelStats(aggregatedKernelStats, targetTracked.getKernelStats());
    }

    for (let step = 0; step < payload.steps; step += 1) {
      if (inputBuffer === null) throw new Error("Input buffer was released unexpectedly.");
      const tracked = createOptimizationTrackedOps(
        device,
        runtimeContext,
        payload.kernelFlags,
        collectKernelStats,
        payload.kernelFlags?.usePersistentWeightBuffers === true
          ? persistentKernelResources
          : undefined,
      );
      try {
        const stepResult = await runStyleTransferStep(
          device,
          payload,
          convLayerCache,
          { contentTargetRelu, styleGramTargets },
          inputBuffer,
          tracked,
          inputOptimizer,
          stepOffset + step,
          step % lossReadbackInterval === 0 || step === payload.steps - 1,
          synchronizePhaseTimings,
        );
        if (stepResult.totalLoss !== null) losses.push(stepResult.totalLoss);
        forwardMs += stepResult.forwardMs;
        lossMs += stepResult.lossMs;
        backwardMs += stepResult.backwardMs;
        updateMs += stepResult.updateMs;
        if (collectKernelStats) {
          mergeKernelStats(aggregatedKernelStats, tracked.getKernelStats());
        }
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
    const kernelStats = collectKernelStats ? { ...aggregatedKernelStats } : undefined;
    if (kernelStats !== undefined) {
      kernelStats.weightUpload = { calls: 1, elapsedMs: weightUploadMs };
    }
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
        steps: payload.steps,
        kernelStats,
      },
    };
  } finally {
    if (inputBuffer !== null) releaseOwnedBuffer(inputBuffer);
    for (const buffer of persistentBuffers) releaseOwnedBuffer(buffer);
    if (sessionId === null) inputOptimizer.dispose();
    if (sessionId !== null) {
      const existing = optimizerSessionById.get(sessionId);
      if (existing !== undefined && existing.inputOptimizer === inputOptimizer) {
        existing.stepOffset += payload.steps;
      }
    }
    runtimeContext.disposeAll();
    for (const buffer of persistentWeightBuffers) buffer.destroy();
  }
};
