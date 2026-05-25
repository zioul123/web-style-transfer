import {
  releaseOwnedBuffer,
  type GpuBufferRef,
} from "../../../runtime/bufferKernels";
import type { OptimizationRuntimeContext } from "../../../runtime/optimizationContext";

const releaseUnpersistedBuffers = (
  ownedBuffers: readonly GpuBufferRef[],
  persistentBuffers: readonly GpuBufferRef[],
): void => {
  const persistentBufferSet: Set<GPUBuffer> = new Set(
    persistentBuffers.map((ref) => ref.buffer),
  );
  for (const ref of ownedBuffers) {
    if (!persistentBufferSet.has(ref.buffer)) releaseOwnedBuffer(ref);
  }
};

export const createPersistentTargetContext = (
  runtimeContext: OptimizationRuntimeContext,
  persistentBuffers: readonly GpuBufferRef[],
): OptimizationRuntimeContext => {
  const targetOwnedBuffers: GpuBufferRef[] = [];
  return {
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
};
