import type { WorkerRequest, WorkerResponse } from './protocol';

export function createStyleTransferWorker() {
  const worker = new Worker(new URL('./styleTransfer.worker.ts', import.meta.url), { type: 'module' });
  let nextId = 1;

  const request = <T extends WorkerResponse>(message: Omit<WorkerRequest, 'id'>): Promise<T> => {
    const id = nextId++;
    return new Promise<T>((resolve, reject) => {
      const onMessage = (event: MessageEvent<WorkerResponse>) => {
        if (event.data.id !== id) return;
        worker.removeEventListener('message', onMessage);
        if (event.data.type === 'error') {
          reject(new Error(event.data.message));
          return;
        }
        resolve(event.data as T);
      };
      worker.addEventListener('message', onMessage);
      worker.postMessage({ ...message, id });
    });
  };

  return {
    ping: async () => (await request<{ id: number; type: 'pong'; message: string }>({ type: 'ping' })).message,
    initWebGPU: () => request<{ id: number; type: 'webgpu-status'; ok: boolean; reason?: string }>({ type: 'init-webgpu' }),
    dispose: () => worker.terminate()
  };
}
