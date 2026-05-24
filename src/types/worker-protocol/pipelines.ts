import type { TensorShape, WorkerTensor } from "./core";

export type WorkerRunStyleTransferRequest = {
  type: "run-style-transfer";
  id: string;
  optimizer: "sgd" | "adam" | "lbfgs";
  gpuResident?: boolean;
  fusedOps?: boolean;
  superFusedOps?: boolean;
  adamBeta1?: number;
  adamBeta2?: number;
  adamEpsilon?: number;
  lbfgsMemory?: number;
  lbfgsEpsilon?: number;
  inputShape: TensorShape;
  inputImageValues: number[];
  contentImageValues: number[];
  styleImageValues: number[];
  mean: [number, number, number];
  std: [number, number, number];
  styleLayerIndices: number[];
  contentLayerIndex: number;
  weights: Record<string, number[] | [number, number, number, number]>;
  contentWeight: number;
  styleWeight: number;
  learningRate: number;
  steps: number;
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
  debugUseLegacyCpuLossReadback?: boolean;
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
  clampMs: number;
  steps: number;
};
