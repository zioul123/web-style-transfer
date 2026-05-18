import { useEffect, useMemo, useState } from 'react'
import type { WorkerRequest, WorkerResponse } from './types'

const createMessageId = (): string => `${Date.now()}-${Math.random().toString(16).slice(2)}`

function App() {
  const [workerStatus, setWorkerStatus] = useState<string>('Worker starting…')
  const [webGpuStatus, setWebGpuStatus] = useState<string>('Checking WebGPU support…')

  const hasWebGpuOnMainThread = useMemo<boolean>(() => 'gpu' in navigator, [])

  useEffect(() => {
    const worker = new Worker(new URL('./styleTransfer.worker.ts', import.meta.url), {
      type: 'module',
    })

    const handleMessage = (event: MessageEvent<WorkerResponse>): void => {
      const payload = event.data

      switch (payload.type) {
        case 'pong': {
          setWorkerStatus(`Worker responded to ping at ${new Date(payload.timestamp).toISOString()}.`)
          break
        }
        case 'webgpu-init-result': {
          setWebGpuStatus(payload.ok ? payload.message : `WebGPU fallback: ${payload.message}`)
          break
        }
        case 'error': {
          setWebGpuStatus(`WebGPU fallback: ${payload.message}`)
          break
        }
        case 'tensor-roundtrip-result':
        case 'tensor-op-result':
        case 'run-first-pool-optimizer-result':
        case 'run-style-transfer-result': {
          break
        }
        default: {
          const exhaustivenessCheck: never = payload
          setWebGpuStatus(`WebGPU fallback: Unhandled worker message: ${String(exhaustivenessCheck)}`)
        }
      }
    }

    worker.addEventListener('message', handleMessage)

    const pingMessage: WorkerRequest = { type: 'ping', id: createMessageId() }
    worker.postMessage(pingMessage)

    const initMessage: WorkerRequest = { type: 'init-webgpu', id: createMessageId() }
    worker.postMessage(initMessage)

    return () => {
      worker.removeEventListener('message', handleMessage)
      worker.terminate()
    }
  }, [])

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-3xl flex-col justify-center gap-6 px-6 py-16 text-slate-100">
      <h1 className="text-3xl font-semibold">WebGPU Style Transfer — Phase 0</h1>
      <p className="rounded-lg border border-slate-700 bg-slate-900 p-4">{workerStatus}</p>
      <p className="rounded-lg border border-slate-700 bg-slate-900 p-4">{webGpuStatus}</p>
      {!hasWebGpuOnMainThread ? (
        <p className="rounded-lg border border-amber-500/50 bg-amber-500/10 p-4 text-amber-200">
          This browser environment does not expose WebGPU on the main thread. Use Chrome/Chromium with WebGPU enabled.
        </p>
      ) : null}
    </main>
  )
}

export default App
