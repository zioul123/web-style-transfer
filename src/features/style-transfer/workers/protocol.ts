export type WorkerRequest =
  | { id: number; type: 'ping' }
  | { id: number; type: 'init-webgpu' };

export type WorkerResponse =
  | { id: number; type: 'pong'; message: string }
  | { id: number; type: 'webgpu-status'; ok: boolean; reason?: string }
  | { id: number; type: 'error'; message: string };
