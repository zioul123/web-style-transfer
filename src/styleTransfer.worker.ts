/// <reference lib="webworker" />

import type { WorkerRequest, WorkerResponse } from './types'

const postResponse = (response: WorkerResponse): void => {
  self.postMessage(response)
}

const initWebGpu = async (id: string): Promise<void> => {
  if (!('gpu' in navigator)) {
    postResponse({
      type: 'webgpu-init-result',
      id,
      ok: false,
      message: 'navigator.gpu is unavailable in this worker context.',
    })
    return
  }

  const adapter = await navigator.gpu.requestAdapter()
  if (adapter === null) {
    postResponse({
      type: 'webgpu-init-result',
      id,
      ok: false,
      message: 'WebGPU adapter request returned null.',
    })
    return
  }

  const device = await adapter.requestDevice()
  postResponse({
    type: 'webgpu-init-result',
    id,
    ok: true,
    message: `WebGPU device initialized: ${device.label || 'unnamed-device'}`,
  })
}

self.onmessage = (event: MessageEvent<WorkerRequest>): void => {
  const payload = event.data

  switch (payload.type) {
    case 'ping': {
      postResponse({ type: 'pong', id: payload.id, timestamp: Date.now() })
      break
    }
    case 'init-webgpu': {
      void initWebGpu(payload.id).catch((error: unknown) => {
        const message = error instanceof Error ? error.message : 'Unknown worker error'
        postResponse({ type: 'error', id: payload.id, message })
      })
      break
    }
    default: {
      const exhaustivenessCheck: never = payload
      postResponse({
        type: 'error',
        id: 'unknown',
        message: `Received unsupported message type: ${String(exhaustivenessCheck)}`,
      })
    }
  }
}
