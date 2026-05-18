import { Canvas } from '@react-three/fiber'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { ReactElement } from 'react'
import * as THREE from 'three'
import type { WorkerRequest, WorkerResponse } from './types'

const createMessageId = (): string => `${Date.now()}-${Math.random().toString(16).slice(2)}`

type RgbImage = { width: number; height: number; values: Float32Array }

type GpuInfo = { adapterInfo: string; features: string; limits: string }

const imageToRgb = async (file: File, targetWidth: number, targetHeight: number): Promise<RgbImage> => {
  const bitmap = await createImageBitmap(file)
  const canvas = document.createElement('canvas')
  canvas.width = targetWidth
  canvas.height = targetHeight
  const context = canvas.getContext('2d')
  if (context === null) throw new Error('Unable to create 2D context.')
  context.drawImage(bitmap, 0, 0, targetWidth, targetHeight)
  const data = context.getImageData(0, 0, targetWidth, targetHeight).data
  const values = new Float32Array(targetWidth * targetHeight * 3)
  for (let i = 0; i < targetWidth * targetHeight; i += 1) {
    values[i] = data[i * 4] / 255
    values[targetWidth * targetHeight + i] = data[i * 4 + 1] / 255
    values[targetWidth * targetHeight * 2 + i] = data[i * 4 + 2] / 255
  }
  return { width: targetWidth, height: targetHeight, values }
}

const rgbToDataUrl = (image: RgbImage): string => {
  const canvas = document.createElement('canvas')
  canvas.width = image.width
  canvas.height = image.height
  const context = canvas.getContext('2d')
  if (context === null) return ''
  const imageData = context.createImageData(image.width, image.height)
  const hw = image.width * image.height
  for (let i = 0; i < hw; i += 1) {
    imageData.data[i * 4] = Math.max(0, Math.min(255, Math.round(image.values[i] * 255)))
    imageData.data[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(image.values[hw + i] * 255)))
    imageData.data[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(image.values[hw * 2 + i] * 255)))
    imageData.data[i * 4 + 3] = 255
  }
  context.putImageData(imageData, 0, 0)
  return canvas.toDataURL('image/png')
}

const ImagePlane = ({ textureUrl, x }: { textureUrl: string; x: number }): ReactElement | null => {
  const texture = useMemo(() => (textureUrl.length > 0 ? new THREE.TextureLoader().load(textureUrl) : null), [textureUrl])
  if (texture === null) return null
  return <mesh position={[x, 0, 0]}><planeGeometry args={[1.8, 1.8]} /><meshBasicMaterial map={texture} /></mesh>
}

function App() {
  const [workerStatus, setWorkerStatus] = useState<string>('Worker starting…')
  const [webGpuStatus, setWebGpuStatus] = useState<string>('Checking WebGPU support…')
  const [gpuInfo, setGpuInfo] = useState<GpuInfo | null>(null)
  const [contentFile, setContentFile] = useState<File | null>(null)
  const [styleFile, setStyleFile] = useState<File | null>(null)
  const [resolution, setResolution] = useState<number>(128)
  const [steps, setSteps] = useState<number>(25)
  const [currentStep, setCurrentStep] = useState<number>(0)
  const [loss, setLoss] = useState<number | null>(null)
  const [isRunning, setIsRunning] = useState<boolean>(false)
  const workerRef = useRef<Worker | null>(null)
  const [contentUrl, setContentUrl] = useState<string>('')
  const [styleUrl, setStyleUrl] = useState<string>('')
  const [outputUrl, setOutputUrl] = useState<string>('')

  const hasWebGpuOnMainThread = useMemo<boolean>(() => 'gpu' in navigator, [])

  useEffect(() => {
    const worker = new Worker(new URL('./styleTransfer.worker.ts', import.meta.url), { type: 'module' })
    workerRef.current = worker
    const handleMessage = (event: MessageEvent<WorkerResponse>): void => {
      const payload = event.data
      if (payload.type === 'pong') setWorkerStatus(`Worker responded to ping at ${new Date(payload.timestamp).toISOString()}.`)
      if (payload.type === 'webgpu-init-result') setWebGpuStatus(payload.ok ? payload.message : `WebGPU fallback: ${payload.message}`)
      if (payload.type === 'run-style-transfer-progress') {
        setCurrentStep(payload.step)
        setLoss(payload.loss)
        const hw = resolution * resolution
        setOutputUrl(rgbToDataUrl({ width: resolution, height: resolution, values: new Float32Array(payload.values.slice(0, hw * 3)) }))
      }
      if (payload.type === 'run-style-transfer-result') setIsRunning(false)
    }
    worker.addEventListener('message', handleMessage)
    worker.postMessage({ type: 'ping', id: createMessageId() } satisfies WorkerRequest)
    worker.postMessage({ type: 'init-webgpu', id: createMessageId() } satisfies WorkerRequest)
    void (async (): Promise<void> => {
      if (!('gpu' in navigator)) return
      const adapter = await navigator.gpu.requestAdapter()
      if (adapter === null) return
      const adapterWithInfo = adapter as GPUAdapter & { requestAdapterInfo?: () => Promise<{ vendor?: string; architecture?: string; description?: string }> }
      const info = adapterWithInfo.requestAdapterInfo === undefined ? null : await adapterWithInfo.requestAdapterInfo()
      setGpuInfo({ adapterInfo: info === null ? 'Adapter info unavailable' : `${info.vendor ?? 'unknown'} / ${info.architecture ?? 'unknown'} / ${info.description ?? 'unknown'}`, features: Array.from(adapter.features).join(', '), limits: `maxStorageBufferBindingSize=${adapter.limits.maxStorageBufferBindingSize}` })
    })()
    return () => worker.terminate()
  }, [resolution])

  const onRun = async (): Promise<void> => {
    if (workerRef.current === null || contentFile === null || styleFile === null) return
    setIsRunning(true)
    setCurrentStep(0)
    const content = await imageToRgb(contentFile, resolution, resolution)
    const style = await imageToRgb(styleFile, resolution, resolution)
    setContentUrl(rgbToDataUrl(content)); setStyleUrl(rgbToDataUrl(style))
    const fixtureResponse = await fetch('/vgg19-phase3-full-pass/vgg19_phase3_full_pass_fixture.json')
    const weightResponse = await fetch('/vgg19-phase3-full-pass/vgg19_conv0_to_conv28_weights.json')
    if (!fixtureResponse.ok || !weightResponse.ok) { setWebGpuStatus('Missing phase 3 fixture/weights in public assets.'); setIsRunning(false); return }
    const fixture = await fixtureResponse.json() as any
    const weights = await weightResponse.json() as Record<string, number[] | [number, number, number, number]>
    const req: WorkerRequest = { type: 'run-style-transfer', id: createMessageId(), inputShape: [1, 3, resolution, resolution], inputImageValues: Array.from(content.values), contentImageValues: Array.from(content.values), styleImageValues: Array.from(style.values), mean: fixture.mean, std: fixture.std, styleLayerIndices: fixture.styleLayerIndices, contentLayerIndex: fixture.contentLayerIndex, weights, contentWeight: 1, styleWeight: 1, learningRate: 0.01, steps }
    workerRef.current.postMessage(req)
  }

  return <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-4 px-6 py-8 text-slate-100"><h1 className="text-3xl font-semibold">WebGPU Style Transfer — Phase 6 UI</h1><p className="rounded-lg border border-slate-700 bg-slate-900 p-3">{workerStatus}</p><p className="rounded-lg border border-slate-700 bg-slate-900 p-3">{webGpuStatus}</p>{gpuInfo !== null ? <div className="rounded-lg border border-cyan-500/40 bg-cyan-500/10 p-3 text-sm">GPU: {gpuInfo.adapterInfo}<br />Features: {gpuInfo.features || 'n/a'}<br />Limits: {gpuInfo.limits}</div> : null}
  <div className="grid grid-cols-1 gap-3 md:grid-cols-3"><input type="file" accept="image/*" onChange={(event) => setContentFile(event.target.files?.[0] ?? null)} className="rounded bg-slate-800 p-2" /><input type="file" accept="image/*" onChange={(event) => setStyleFile(event.target.files?.[0] ?? null)} className="rounded bg-slate-800 p-2" /><select value={resolution} onChange={(event) => setResolution(Number(event.target.value))} className="rounded bg-slate-800 p-2"><option value={128}>128</option><option value={256}>256</option><option value={384}>384</option></select></div>
  <div className="flex items-center gap-3"><label>Steps</label><input type="range" min={5} max={100} value={steps} onChange={(event) => setSteps(Number(event.target.value))} /><span>{steps}</span><button onClick={() => { if (!isRunning) { void onRun() } }} className="rounded bg-emerald-500 px-4 py-2 text-black">{isRunning ? 'Running…' : 'Play'}</button><button onClick={() => setIsRunning(false)} className="rounded border px-4 py-2">Pause UI</button><span>Iteration: {currentStep}</span><span>Loss: {loss?.toFixed(5) ?? 'n/a'}</span></div>
  <section className="h-[420px] rounded-lg border border-slate-700 bg-black/40"><Canvas camera={{ position: [0, 0, 3.5] }}><ImagePlane textureUrl={contentUrl} x={-1.9} /><ImagePlane textureUrl={styleUrl} x={0} /><ImagePlane textureUrl={outputUrl} x={1.9} /></Canvas></section>
  {!hasWebGpuOnMainThread ? <p className="rounded-lg border border-amber-500/50 bg-amber-500/10 p-4 text-amber-200">This browser environment does not expose WebGPU on the main thread.</p> : null}
  <aside className="mt-4 rounded-lg border border-dashed border-slate-600 p-4 text-slate-300">Blog plug placeholder: <em>How we made this</em>.</aside>
  </main>
}

export default App
