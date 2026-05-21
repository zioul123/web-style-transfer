import type { TensorShape, WorkerTensor, WorkerTensorOperand } from "./core";

export type WorkerTensorBinaryOpRequest = {
  type: "tensor-op";
  id: string;
  op: "add" | "sub" | "mul" | "div" | "mse";
  a: WorkerTensorOperand;
  b: WorkerTensorOperand;
};

export type WorkerTensorClampOpRequest = {
  type: "tensor-op";
  id: string;
  op: "clamp";
  a: WorkerTensorOperand;
  clampMin: number;
  clampMax: number;
};

export type WorkerTensorConv2dForwardOpRequest = {
  type: "tensor-op";
  id: string;
  op: "conv2d-forward" | "conv2d-relu-forward";
  input: WorkerTensor;
  weight: WorkerTensor;
  bias: number[];
};

export type WorkerTensorUnaryForwardOpRequest = {
  type: "tensor-op";
  id: string;
  op: "relu-forward" | "maxpool2d-forward";
  input: WorkerTensor;
};

export type WorkerTensorReluBackwardOpRequest = {
  type: "tensor-op";
  id: string;
  op: "relu-backward";
  input: WorkerTensor;
  gradOut: WorkerTensor;
};

export type WorkerTensorMaxPoolBackwardOpRequest = {
  type: "tensor-op";
  id: string;
  op: "maxpool2d-backward";
  input: WorkerTensor;
  gradOut: WorkerTensor;
};

export type WorkerTensorNormalizeBackwardOpRequest = {
  type: "tensor-op";
  id: string;
  op: "normalize-backward";
  input: WorkerTensor;
  gradOut: WorkerTensor;
  std: number[];
};

export type WorkerTensorConv2dBackwardInputOpRequest = {
  type: "tensor-op";
  id: string;
  op: "conv2d-backward-input";
  inputShape: TensorShape;
  gradOut: WorkerTensor;
  weight: WorkerTensor;
};

export type WorkerTensorGramBackwardOpRequest = {
  type: "tensor-op";
  id: string;
  op: "gram-backward";
  input: WorkerTensor;
  gradOut: WorkerTensor;
};

export type WorkerTensorContentLossBackwardOpRequest = {
  type: "tensor-op";
  id: string;
  op: "content-loss-backward";
  input: WorkerTensor;
  target: WorkerTensor;
  contentWeight?: number;
};

export type WorkerTensorStyleLossBackwardOpRequest = {
  type: "tensor-op";
  id: string;
  op: "style-loss-backward";
  input: WorkerTensor;
  target: WorkerTensor;
  styleWeight?: number;
};

export type WorkerTensorReshapeGramOpRequest = {
  type: "tensor-op";
  id: string;
  op: "reshape-chw-flatten" | "gram-matrix";
  input: WorkerTensor;
};

export type WorkerTensorLossOpRequest = {
  type: "tensor-op";
  id: string;
  op: "content-loss" | "style-loss";
  input: WorkerTensor;
  target: WorkerTensor;
  contentWeight?: number;
  styleWeight?: number;
};

export type WorkerTensorNormalizeForwardOpRequest = {
  type: "tensor-op";
  id: string;
  op: "normalize-forward";
  input: WorkerTensor;
  mean: number[];
  std: number[];
};

export type WorkerTensorOpRequest =
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
