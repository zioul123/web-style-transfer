import type { TensorShape, WorkerFloatArray, WorkerTensor } from "./core";

export type WorkerKernelFamily =
  | "convForward"
  | "convBackwardInput"
  | "gramStyle"
  | "poolBackward"
  | "pointwiseUpdate"
  | "weightUpload";

export type WorkerKernelFamilyStat = {
  calls: number;
  elapsedMs: number;
};

export type WorkerKernelStats = Partial<
  Record<WorkerKernelFamily, WorkerKernelFamilyStat>
>;

export type WorkerKernelOptimizationFlags = {
  useCachedPipelines?: boolean;
  usePersistentWeightBuffers?: boolean;
  useStepBufferPool?: boolean;
  useVec4Pointwise?: boolean;
  usePoolBackwardScatter?: boolean;
  gramKernel?: "scalar" | "parallel-dot" | "symmetric-parallel-dot";
  styleBackward?: "two-pass" | "fused-from-gram-diff";
  convForwardKernel?: "scalar" | "spatial-vec4" | "tiled-spatial";
  convBackwardInputKernel?:
    | "scalar"
    | "spatial-vec4"
    | "transposed-weight-spatial-vec4";
  weightStorage?: "fp32" | "fp16-storage" | "int8-dequant-experimental";
};

export type WorkerRunStyleTransferRequest = {
  type: "run-style-transfer";
  id: string;
  sessionId?: string;
  optimizer: "sgd" | "adam" | "lbfgs";
  adamBeta1?: number;
  adamBeta2?: number;
  adamEpsilon?: number;
  lbfgsMemory?: number;
  lbfgsEpsilon?: number;
  inputShape: TensorShape;
  contentShape?: TensorShape;
  styleShape?: TensorShape;
  inputImageValues: number[];
  contentImageValues: number[];
  styleImageValues: number[];
  mean: [number, number, number];
  std: [number, number, number];
  styleLayerIndices: number[];
  contentLayerIndex: number;
  weights: Record<string, WorkerFloatArray | [number, number, number, number]>;
  contentWeight: number;
  styleWeight: number;
  learningRate: number;
  steps: number;
  lossReadbackInterval?: number;
  synchronizePhaseTimings?: boolean;
  kernelFlags?: WorkerKernelOptimizationFlags;
  collectKernelStats?: boolean;
};

export type WorkerClearStyleTransferSessionRequest = {
  type: "clear-style-transfer-session";
  id: string;
  sessionId: string;
};

export type WorkerRunFirstPoolOptimizerRequest = {
  type: "run-first-pool-optimizer";
  id: string;
  inputShape: TensorShape;
  mean: [number, number, number];
  std: [number, number, number];
  contentImageValues: number[];
  styleImageValues: number[];
  initialInputValues: number[];
  conv1Weight: WorkerTensor;
  conv1Bias: number[];
  conv2Weight: WorkerTensor;
  conv2Bias: number[];
  conv3Weight: WorkerTensor;
  conv3Bias: number[];
  contentWeight: number;
  styleWeightConv1: number;
  styleWeightConv3: number;
  learningRate: number;
  steps: number;
  useFusedConvRelu?: boolean;
  useFusedUpdateClamp?: boolean;
  debugValidateStepShapes?: boolean;
  debugReadbackGrad?: boolean;
  collectBenchmarkStats?: boolean;
};

export type WorkerFirstPoolBenchmarkStats = {
  elapsedMs: number;
  avgStepMs: number;
  forwardMs: number;
  lossMs: number;
  backwardMs: number;
  updateMs: number;
  readbackMs: number;
  mandatoryReadbackMs: number;
  diagnosticsReadbackMs: number;
  steps: number;
};

export type WorkerRunStats = {
  elapsedMs: number;
  avgStepMs: number;
  forwardMs: number;
  backwardMs: number;
  lossMs: number;
  updateMs: number;
  steps: number;
  kernelStats?: WorkerKernelStats;
};
