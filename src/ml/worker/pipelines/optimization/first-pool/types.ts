import type { createTensor } from "../../../../index";
import type {
  GpuBufferRef,
  OwnedGpuBuffer,
} from "../../../runtime/bufferKernels";
import type { OptimizationRuntimeContext } from "../../../runtime/optimizationContext";
import type { TensorShape4D } from "../../../runtime/tensorShapes";

export type { TensorShape4D };

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
  diagnosticsReadbackMs: number;
};

export type FirstPoolStatsAccumulator = {
  forwardMs: number;
  lossMs: number;
  backwardMs: number;
  updateMs: number;
  readbackMs: number;
  mandatoryReadbackMs: number;
  diagnosticsReadbackMs: number;
};

export type FirstPoolRuntimeContext = OptimizationRuntimeContext;
