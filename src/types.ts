export type TensorShape = readonly [number, number, number, number];

export type TensorData = {
  shape: TensorShape;
  values: Float32Array;
};

export type WorkerTensor = {
  shape: TensorShape;
  values: number[];
};

export type WorkerTensorOperand =
  | { kind: "tensor"; tensor: WorkerTensor }
  | { kind: "scalar"; scalar: number };

type WorkerTensorBinaryOpRequest = {
  type: "tensor-op";
  id: string;
  op: "add" | "sub" | "mul" | "div" | "mse";
  a: WorkerTensorOperand;
  b: WorkerTensorOperand;
};

type WorkerTensorClampOpRequest = {
  type: "tensor-op";
  id: string;
  op: "clamp";
  a: WorkerTensorOperand;
  clampMin: number;
  clampMax: number;
};

type WorkerTensorConv2dForwardOpRequest = {
  type: "tensor-op";
  id: string;
  op: "conv2d-forward" | "conv2d-relu-forward";
  input: WorkerTensor;
  weight: WorkerTensor;
  bias: number[];
};

type WorkerTensorUnaryForwardOpRequest = {
  type: "tensor-op";
  id: string;
  op: "relu-forward" | "maxpool2d-forward";
  input: WorkerTensor;
};

type WorkerTensorReluBackwardOpRequest = {
  type: "tensor-op";
  id: string;
  op: "relu-backward";
  input: WorkerTensor;
  gradOut: WorkerTensor;
};

type WorkerTensorMaxPoolBackwardOpRequest = {
  type: "tensor-op";
  id: string;
  op: "maxpool2d-backward";
  input: WorkerTensor;
  gradOut: WorkerTensor;
};

type WorkerTensorNormalizeBackwardOpRequest = {
  type: "tensor-op";
  id: string;
  op: "normalize-backward";
  input: WorkerTensor;
  gradOut: WorkerTensor;
  std: number[];
};

type WorkerTensorConv2dBackwardInputOpRequest = {
  type: "tensor-op";
  id: string;
  op: "conv2d-backward-input";
  inputShape: TensorShape;
  gradOut: WorkerTensor;
  weight: WorkerTensor;
};

type WorkerTensorGramBackwardOpRequest = {
  type: "tensor-op";
  id: string;
  op: "gram-backward";
  input: WorkerTensor;
  gradOut: WorkerTensor;
};

type WorkerTensorContentLossBackwardOpRequest = {
  type: "tensor-op";
  id: string;
  op: "content-loss-backward";
  input: WorkerTensor;
  target: WorkerTensor;
  contentWeight?: number;
};

type WorkerTensorStyleLossBackwardOpRequest = {
  type: "tensor-op";
  id: string;
  op: "style-loss-backward";
  input: WorkerTensor;
  target: WorkerTensor;
  styleWeight?: number;
};

type WorkerTensorReshapeGramOpRequest = {
  type: "tensor-op";
  id: string;
  op: "reshape-chw-flatten" | "gram-matrix";
  input: WorkerTensor;
};

type WorkerTensorLossOpRequest = {
  type: "tensor-op";
  id: string;
  op: "content-loss" | "style-loss";
  input: WorkerTensor;
  target: WorkerTensor;
  contentWeight?: number;
  styleWeight?: number;
};

type WorkerTensorNormalizeForwardOpRequest = {
  type: "tensor-op";
  id: string;
  op: "normalize-forward";
  input: WorkerTensor;
  mean: number[];
  std: number[];
};

type WorkerRunStyleTransferRequest = {
  type: "run-style-transfer";
  id: string;
  optimizer: "sgd" | "adam" | "lbfgs";
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
type WorkerRunFirstPoolOptimizerRequest = {
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
};
export type WorkerRequest =
  | { type: "ping"; id: string }
  | { type: "init-webgpu"; id: string }
  | { type: "tensor-roundtrip"; id: string; tensor: WorkerTensor }
  | WorkerRunFirstPoolOptimizerRequest
  | WorkerRunStyleTransferRequest
  | WorkerTensorBinaryOpRequest
  | WorkerTensorClampOpRequest
  | WorkerTensorConv2dForwardOpRequest
  | WorkerTensorUnaryForwardOpRequest
  | WorkerTensorReluBackwardOpRequest
  | WorkerTensorMaxPoolBackwardOpRequest
  | WorkerTensorNormalizeBackwardOpRequest
  | WorkerTensorConv2dBackwardInputOpRequest
  | WorkerTensorGramBackwardOpRequest
  | WorkerTensorContentLossBackwardOpRequest
  | WorkerTensorStyleLossBackwardOpRequest
  | WorkerTensorNormalizeForwardOpRequest
  | WorkerTensorReshapeGramOpRequest
  | WorkerTensorLossOpRequest;

export type WorkerRoundtripResponse =
  | {
      type: "tensor-roundtrip-result";
      id: string;
      ok: true;
      tensor: WorkerTensor;
    }
  | { type: "tensor-roundtrip-result"; id: string; ok: false; message: string };

export type WorkerTensorScalarOpResponse = {
  type: "tensor-op-result";
  id: string;
  ok: true;
  scalar: number;
};
export type WorkerTensorVectorOpResponse = {
  type: "tensor-op-result";
  id: string;
  ok: true;
  values: number[];
};
export type WorkerTensorOpErrorResponse = {
  type: "tensor-op-result";
  id: string;
  ok: false;
  message: string;
};

export type WorkerTensorOpResponse =
  | WorkerTensorScalarOpResponse
  | WorkerTensorVectorOpResponse
  | WorkerTensorOpErrorResponse;

export const isWorkerTensorScalarOpResponse = (
  value: WorkerTensorOpResponse,
): value is WorkerTensorScalarOpResponse => value.ok && "scalar" in value;

export const isWorkerTensorVectorOpResponse = (
  value: WorkerTensorOpResponse,
): value is WorkerTensorVectorOpResponse => value.ok && "values" in value;

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

export type WorkerResponse =
  | { type: "pong"; id: string; timestamp: number }
  | { type: "webgpu-init-result"; id: string; ok: boolean; message: string }
  | WorkerRoundtripResponse
  | WorkerTensorOpResponse
  | { type: "error"; id: string; message: string }
  | {
      type: "run-first-pool-optimizer-result";
      id: string;
      ok: true;
      losses: number[];
      finalValues: number[];
    }
  | {
      type: "run-first-pool-optimizer-result";
      id: string;
      ok: false;
      message: string;
    }
  | {
      type: "run-style-transfer-result";
      id: string;
      ok: true;
      losses: number[];
      finalValues: number[];
      stats: WorkerRunStats;
    }
  | {
      type: "run-style-transfer-result";
      id: string;
      ok: false;
      message: string;
    };
