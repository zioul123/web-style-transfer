/// <reference lib="webworker" />

import type { WorkerRequest } from "../../../../types";
import { readGpuBufferToArray, releaseOwnedBuffer, uploadToOwnedBuffer } from "../../runtime/bufferKernels";
import { getGpuDevice } from "../../runtime/deviceState";
import { setupFirstPoolTargets } from "./first-pool/setupTargets";
import { runFirstPoolStep } from "./first-pool/step";
import { createFirstPoolTempBufferStore } from "./first-pool/tempBuffers";
import type { FirstPoolStatsAccumulator } from "./first-pool/types";

const elementCount = (shape: readonly [number, number, number, number]): number => shape[0] * shape[1] * shape[2] * shape[3];

export const runFirstPoolOptimizer = async (
  payload: Extract<WorkerRequest, { type: "run-first-pool-optimizer" }>,
): Promise<{ losses: number[]; finalValues: number[]; stats?: {
  elapsedMs: number; avgStepMs: number; forwardMs: number; lossMs: number; backwardMs: number; updateMs: number; readbackMs: number; mandatoryReadbackMs: number; diagnosticsReadbackMs: number; steps: number;
} }> => {
  const device = getGpuDevice();
  if (device === null) throw new Error("WebGPU device is not initialized");

  const tempBuffers = createFirstPoolTempBufferStore(device);
  const persistent = await setupFirstPoolTargets(device, payload);
  const stats: FirstPoolStatsAccumulator = { forwardMs: 0, lossMs: 0, backwardMs: 0, updateMs: 0, readbackMs: 0, mandatoryReadbackMs: 0, diagnosticsReadbackMs: 0 };
  const collectBenchmarkStats = payload.collectBenchmarkStats ?? false;
  const losses: number[] = [];
  let inputBuffer = uploadToOwnedBuffer(device, new Float32Array(payload.initialInputValues));
  const runStart = performance.now();
  try {
    for (let step = 0; step < payload.steps; step += 1) {
      const result = await runFirstPoolStep(device, payload, persistent, inputBuffer, tempBuffers);
      const previousInput = inputBuffer;
      inputBuffer = result.nextInputBuffer;
      releaseOwnedBuffer(previousInput);
      losses.push(result.totalLoss);
      stats.forwardMs += result.forwardMs;
      stats.lossMs += result.lossMs;
      stats.backwardMs += result.backwardMs;
      stats.updateMs += result.updateMs;
      stats.diagnosticsReadbackMs += result.diagnosticsReadbackMs;
      stats.readbackMs += result.diagnosticsReadbackMs;
    }
    const finalReadStart = performance.now();
    const finalValues = await readGpuBufferToArray(device, inputBuffer.buffer, elementCount(payload.inputShape));
    const finalReadMs = performance.now() - finalReadStart;
    stats.mandatoryReadbackMs += finalReadMs;
    stats.readbackMs += finalReadMs;
    const elapsedMs = performance.now() - runStart;
    return {
      losses,
      finalValues: Array.from(finalValues),
      stats: collectBenchmarkStats ? { elapsedMs, avgStepMs: payload.steps > 0 ? elapsedMs / payload.steps : 0, ...stats, steps: payload.steps } : undefined,
    };
  } finally {
    releaseOwnedBuffer(inputBuffer);
    persistent.dispose();
    tempBuffers.drain();
  }
};
