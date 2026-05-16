/// <reference lib="webworker" />
import { checkWebGPUSupport } from '../../../ml/webgpu/adapter';
import type { WorkerRequest, WorkerResponse } from './protocol';

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const req = event.data;
  try {
    if (req.type === 'ping') {
      const response: WorkerResponse = { id: req.id, type: 'pong', message: 'pong' };
      self.postMessage(response);
      return;
    }
    if (req.type === 'init-webgpu') {
      const status = await checkWebGPUSupport();
      const response: WorkerResponse = { id: req.id, type: 'webgpu-status', ok: status.supported, reason: status.reason };
      self.postMessage(response);
      return;
    }
  } catch (error) {
    const response: WorkerResponse = { id: req.id, type: 'error', message: String(error) };
    self.postMessage(response);
  }
};
