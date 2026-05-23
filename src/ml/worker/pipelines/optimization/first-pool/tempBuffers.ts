import { acquireReusableBuffer, releaseReusableBuffer } from "../../../runtime/bufferPool";
import type { FirstPoolTempBufferStore, TensorShape4D } from "./types";

const elementCount = (shape: TensorShape4D): number => shape[0] * shape[1] * shape[2] * shape[3];

export const createFirstPoolTempBufferStore = (device: GPUDevice): FirstPoolTempBufferStore => {
  const localBufferByKey: Map<string, GPUBuffer> = new Map();
  const localTempKey = (shape: TensorShape4D, usage: number, role: string): string =>
    `${shape.join("x")}:${usage}:${role}`;

  const acquireTempBuffer = (shape: TensorShape4D, usage: number, role: string): GPUBuffer => {
    const key = localTempKey(shape, usage, role);
    const existing = localBufferByKey.get(key);
    if (existing !== undefined) return existing;
    const size = elementCount(shape) * Float32Array.BYTES_PER_ELEMENT;
    const buffer = acquireReusableBuffer(device, size, usage);
    localBufferByKey.set(key, buffer);
    return buffer;
  };

  const releaseTempBuffer = (shape: TensorShape4D, usage: number, role: string): void => {
    const key = localTempKey(shape, usage, role);
    const buffer = localBufferByKey.get(key);
    if (buffer === undefined) return;
    localBufferByKey.delete(key);
    releaseReusableBuffer(elementCount(shape) * Float32Array.BYTES_PER_ELEMENT, usage, buffer);
  };

  const drain = (): void => {
    for (const [key, buffer] of localBufferByKey.entries()) {
      const [shapeKey, usageToken] = key.split(":");
      const shapeParts = shapeKey.split("x").map((part) => Number(part));
      if (shapeParts.length !== 4 || shapeParts.some((part) => Number.isNaN(part))) continue;
      const usage = Number(usageToken);
      if (Number.isNaN(usage)) continue;
      const shape: TensorShape4D = [shapeParts[0], shapeParts[1], shapeParts[2], shapeParts[3]];
      releaseReusableBuffer(elementCount(shape) * Float32Array.BYTES_PER_ELEMENT, usage, buffer);
    }
    localBufferByKey.clear();
  };

  return { acquireTempBuffer, releaseTempBuffer, drain };
};
