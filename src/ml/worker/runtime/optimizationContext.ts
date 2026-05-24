import { releaseOwnedBuffer, type GpuBufferRef } from "./bufferKernels";
import { acquireReusableBuffer, releaseReusableBuffer } from "./bufferPool";

export type OptimizationShape4D = readonly [number, number, number, number];

type TempRecord = {
  shape: OptimizationShape4D;
  usage: number;
  buffer: GPUBuffer;
};

export type OptimizationRuntimeContext = {
  acquireTemp: (shape: OptimizationShape4D, usage: number, role: string) => GPUBuffer;
  trackOwned: <T extends GpuBufferRef>(...refs: T[]) => T;
  releaseStepOwned: () => void;
  disposeAll: () => void;
};

const elementCount = (shape: OptimizationShape4D): number => shape[0] * shape[1] * shape[2] * shape[3];

export const createOptimizationRuntimeContext = (device: GPUDevice): OptimizationRuntimeContext => {
  const tempByKey: Map<string, TempRecord> = new Map();
  const stepOwned: GpuBufferRef[] = [];

  const tempKey = (shape: OptimizationShape4D, usage: number, role: string): string => `${shape.join("x")}:${usage}:${role}`;

  const acquireTemp = (shape: OptimizationShape4D, usage: number, role: string): GPUBuffer => {
    const key = tempKey(shape, usage, role);
    const existing = tempByKey.get(key);
    if (existing !== undefined) return existing.buffer;
    const size = elementCount(shape) * Float32Array.BYTES_PER_ELEMENT;
    const buffer = acquireReusableBuffer(device, size, usage);
    tempByKey.set(key, { shape, usage, buffer });
    return buffer;
  };

  const trackOwned = <T extends GpuBufferRef>(...refs: T[]): T => {
    for (const ref of refs) stepOwned.push(ref);
    return refs[refs.length - 1];
  };

  const releaseStepOwned = (): void => {
    while (stepOwned.length > 0) {
      const ref = stepOwned.pop();
      if (ref !== undefined) releaseOwnedBuffer(ref);
    }
  };

  const disposeAll = (): void => {
    releaseStepOwned();
    for (const tempRecord of tempByKey.values()) {
      releaseReusableBuffer(
        elementCount(tempRecord.shape) * Float32Array.BYTES_PER_ELEMENT,
        tempRecord.usage,
        tempRecord.buffer,
      );
    }
    tempByKey.clear();
  };

  return { acquireTemp, trackOwned, releaseStepOwned, disposeAll };
};
