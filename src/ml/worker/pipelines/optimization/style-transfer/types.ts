import type { WorkerRequest } from "../../../../../types";
import type { GpuBufferRef } from "../../../runtime/bufferKernels";

export type StyleTransferPayload = Extract<
  WorkerRequest,
  { type: "run-style-transfer" }
>;

export type TensorShape4D = readonly [number, number, number, number];

export type ConvLayerCacheEntry = {
  shape: [number, number, number, number];
  values: Float32Array;
  bias: Float32Array;
};

export type StyleTransferForwardResult = {
  reluOut: Record<number, GpuBufferRef | undefined>;
  reluShape: Record<number, TensorShape4D | undefined>;
  convInShape: Record<number, TensorShape4D | undefined>;
  layerInput: Record<number, GpuBufferRef | undefined>;
  layerInputShape: Record<number, TensorShape4D | undefined>;
  final: GpuBufferRef;
};

export type StyleTransferPersistentContext = {
  contentTargetRelu: GpuBufferRef;
  styleGramTargets: Record<number, GpuBufferRef | undefined>;
};

export type StyleTransferStepResult = {
  nextInputBuffer: GpuBufferRef;
  totalLoss: number;
  forwardMs: number;
  lossMs: number;
  backwardMs: number;
  updateMs: number;
};

export type StyleTransferRunResult = {
  losses: number[];
  finalValues: number[];
  stats: {
    elapsedMs: number;
    avgStepMs: number;
    forwardMs: number;
    backwardMs: number;
    lossMs: number;
    updateMs: number;
    clampMs: number;
    steps: number;
  };
};
