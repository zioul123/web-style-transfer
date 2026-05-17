export type TensorShape = readonly [number, number, number, number]

type WorkerTensor = {
  shape: TensorShape
  values: number[]
}

export type WorkerRequest =
  | { type: 'ping'; id: string }
  | { type: 'init-webgpu'; id: string }
  | { type: 'tensor-roundtrip'; id: string; tensor: WorkerTensor }
  | {
      type: 'tensor-op'
      id: string
      op: 'add' | 'sub' | 'mul' | 'div' | 'clamp' | 'mse'
      a: WorkerTensor
      b?: WorkerTensor
      clampMin?: number
      clampMax?: number
    }

export type WorkerResponse =
  | { type: 'pong'; id: string; timestamp: number }
  | { type: 'webgpu-init-result'; id: string; ok: boolean; message: string }
  | { type: 'tensor-roundtrip-result'; id: string; ok: boolean; tensor?: WorkerTensor; message?: string }
  | { type: 'tensor-op-result'; id: string; ok: boolean; values?: number[]; scalar?: number; message?: string }
  | { type: 'error'; id: string; message: string }
