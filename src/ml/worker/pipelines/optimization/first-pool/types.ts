import type { createTensor } from "../../../../index";
import type { GpuBufferRef, OwnedGpuBuffer } from "../../../runtime/bufferKernels";

export type TensorShape4D = readonly [number, number, number, number];

export type FirstPoolPersistentContext = {
  conv1Weight: ReturnType<typeof createTensor>;
  conv2Weight: ReturnType<typeof createTensor>;
  conv3Weight: ReturnType<typeof createTensor>;
  styleGram1Buffer: GpuBufferRef;
  styleGram3Buffer: GpuBufferRef;
  contentRelu2Buffer: GpuBufferRef;
  dispose: () => void;
};

export type FirstPoolStepResult = {
  nextInputBuffer: OwnedGpuBuffer;
  totalLoss: number;
  forwardMs: number;
  lossMs: number;
  backwardMs: number;
  updateMs: number;
  readbackMs: number;
};

export type FirstPoolStatsAccumulator = {
  forwardMs: number;
  lossMs: number;
  backwardMs: number;
  updateMs: number;
  readbackMs: number;
};

export type FirstPoolTempBufferStore = {
  acquireTempBuffer: (shape: TensorShape4D, usage: number, role: string) => GPUBuffer;
  releaseTempBuffer: (shape: TensorShape4D, usage: number, role: string) => void;
  drain: () => void;
};
