export type TensorShape = readonly [number, number, number, number]

export type TensorData = {
  shape: TensorShape
  values: Float32Array
}

export type WorkerTensor = {
  shape: TensorShape
  values: number[]
}

type WorkerTensorBinaryOpRequest = {
  type: 'tensor-op'
  id: string
  op: 'add' | 'sub' | 'mul' | 'div' | 'mse'
  a: WorkerTensor
  b: WorkerTensor
}

type WorkerTensorClampOpRequest = {
  type: 'tensor-op'
  id: string
  op: 'clamp'
  a: WorkerTensor
  clampMin: number
  clampMax: number
}

export type WorkerRequest =
  | { type: 'ping'; id: string }
  | { type: 'init-webgpu'; id: string }
  | { type: 'tensor-roundtrip'; id: string; tensor: WorkerTensor }
  | WorkerTensorBinaryOpRequest
  | WorkerTensorClampOpRequest

export type WorkerRoundtripResponse =
  | { type: 'tensor-roundtrip-result'; id: string; ok: true; tensor: WorkerTensor }
  | { type: 'tensor-roundtrip-result'; id: string; ok: false; message: string }

export type WorkerTensorOpResponse =
  | { type: 'tensor-op-result'; id: string; ok: true; values: number[] }
  | { type: 'tensor-op-result'; id: string; ok: true; scalar: number }
  | { type: 'tensor-op-result'; id: string; ok: false; message: string }

export type WorkerResponse =
  | { type: 'pong'; id: string; timestamp: number }
  | { type: 'webgpu-init-result'; id: string; ok: boolean; message: string }
  | WorkerRoundtripResponse
  | WorkerTensorOpResponse
  | { type: 'error'; id: string; message: string }
