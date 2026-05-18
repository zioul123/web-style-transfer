export type TensorShape = readonly [number, number, number, number]

export type TensorData = {
  shape: TensorShape
  values: Float32Array
}

export type WorkerTensor = {
  shape: TensorShape
  values: number[]
}

export type WorkerTensorOperand =
  | { kind: 'tensor'; tensor: WorkerTensor }
  | { kind: 'scalar'; scalar: number }

type WorkerTensorBinaryOpRequest = {
  type: 'tensor-op'
  id: string
  op: 'add' | 'sub' | 'mul' | 'div' | 'mse'
  a: WorkerTensorOperand
  b: WorkerTensorOperand
}

type WorkerTensorClampOpRequest = {
  type: 'tensor-op'
  id: string
  op: 'clamp'
  a: WorkerTensorOperand
  clampMin: number
  clampMax: number
}

type WorkerTensorConv2dForwardOpRequest = {
  type: 'tensor-op'
  id: string
  op: 'conv2d-forward'
  input: WorkerTensor
  weight: WorkerTensor
  bias: number[]
}

type WorkerTensorUnaryForwardOpRequest = {
  type: 'tensor-op'
  id: string
  op: 'relu-forward' | 'maxpool2d-forward'
  input: WorkerTensor
}

type WorkerTensorReshapeGramOpRequest = {
  type: 'tensor-op'
  id: string
  op: 'reshape-chw-flatten' | 'gram-matrix'
  input: WorkerTensor
}

type WorkerTensorLossOpRequest = {
  type: 'tensor-op'
  id: string
  op: 'content-loss' | 'style-loss'
  input: WorkerTensor
  target: WorkerTensor
}

type WorkerTensorNormalizeForwardOpRequest = {
  type: 'tensor-op'
  id: string
  op: 'normalize-forward'
  input: WorkerTensor
  mean: number[]
  std: number[]
}

export type WorkerRequest =
  | { type: 'ping'; id: string }
  | { type: 'init-webgpu'; id: string }
  | { type: 'tensor-roundtrip'; id: string; tensor: WorkerTensor }
  | WorkerTensorBinaryOpRequest
  | WorkerTensorClampOpRequest
  | WorkerTensorConv2dForwardOpRequest
  | WorkerTensorUnaryForwardOpRequest
  | WorkerTensorNormalizeForwardOpRequest
  | WorkerTensorReshapeGramOpRequest
  | WorkerTensorLossOpRequest

export type WorkerRoundtripResponse =
  | { type: 'tensor-roundtrip-result'; id: string; ok: true; tensor: WorkerTensor }
  | { type: 'tensor-roundtrip-result'; id: string; ok: false; message: string }

export type WorkerTensorScalarOpResponse = { type: 'tensor-op-result'; id: string; ok: true; scalar: number }
export type WorkerTensorVectorOpResponse = { type: 'tensor-op-result'; id: string; ok: true; values: number[] }
export type WorkerTensorOpErrorResponse = { type: 'tensor-op-result'; id: string; ok: false; message: string }

export type WorkerTensorOpResponse =
  | WorkerTensorScalarOpResponse
  | WorkerTensorVectorOpResponse
  | WorkerTensorOpErrorResponse

export const isWorkerTensorScalarOpResponse = (value: WorkerTensorOpResponse): value is WorkerTensorScalarOpResponse =>
  value.ok && 'scalar' in value

export const isWorkerTensorVectorOpResponse = (value: WorkerTensorOpResponse): value is WorkerTensorVectorOpResponse =>
  value.ok && 'values' in value

export type WorkerResponse =
  | { type: 'pong'; id: string; timestamp: number }
  | { type: 'webgpu-init-result'; id: string; ok: boolean; message: string }
  | WorkerRoundtripResponse
  | WorkerTensorOpResponse
  | { type: 'error'; id: string; message: string }
