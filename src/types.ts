export type WorkerRequest =
  | { type: 'ping'; id: string }
  | { type: 'init-webgpu'; id: string }

export type WorkerResponse =
  | { type: 'pong'; id: string; timestamp: number }
  | { type: 'webgpu-init-result'; id: string; ok: boolean; message: string }
  | { type: 'error'; id: string; message: string }
