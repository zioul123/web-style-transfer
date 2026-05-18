import { Canvas } from '@react-three/fiber'
import { type ReactElement, useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import type { WorkerRequest, WorkerResponse } from './types'

type LoadedImage = {
  fileName: string
  width: number
  height: number
  previewUrl: string
  values: number[]
}

type FixturePayload = {
  mean: [number, number, number]
  std: [number, number, number]
  styleLayerIndices: number[]
  contentLayerIndex: number
  weights: Record<string, number[] | [number, number, number, number]>
}

const createMessageId = (): string => `${Date.now()}-${Math.random().toString(16).slice(2)}`
const DEFAULT_SIZE: number = 128

const createImageValues = (image: HTMLImageElement, size: number): number[] => {
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  if (ctx === null) throw new Error('Could not create 2D context.')
  ctx.drawImage(image, 0, 0, size, size)
  const data = ctx.getImageData(0, 0, size, size).data
  const values: number[] = new Array(3 * size * size)
  const area = size * size
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const pixel = y * size + x
      const src = pixel * 4
      values[pixel] = data[src] / 255
      values[area + pixel] = data[src + 1] / 255
      values[2 * area + pixel] = data[src + 2] / 255
    }
  }
  return values
}

const ImagePlane = ({ src, x }: { src: string; x: number }): ReactElement => {
  const texture = useMemo<THREE.Texture>(() => new THREE.TextureLoader().load(src), [src])
  return <mesh position={[x, 0, 0]}><planeGeometry args={[1.8, 1.8]} /><meshBasicMaterial map={texture} /></mesh>
}

function App() {
  const workerRef = useRef<Worker | null>(null)
  const [workerStatus, setWorkerStatus] = useState<string>('Worker starting…')
  const [webGpuStatus, setWebGpuStatus] = useState<string>('Checking WebGPU support…')
  const [contentImage, setContentImage] = useState<LoadedImage | null>(null)
  const [styleImage, setStyleImage] = useState<LoadedImage | null>(null)
  const [outputImageUrl, setOutputImageUrl] = useState<string | null>(null)
  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const [stepCount, setStepCount] = useState<number>(0)
  const [maxSteps, setMaxSteps] = useState<number>(50)
  const [resolution, setResolution] = useState<number>(DEFAULT_SIZE)
  const [learningRate, setLearningRate] = useState<number>(0.01)
  const [contentWeight, setContentWeight] = useState<number>(1)
  const [styleWeight, setStyleWeight] = useState<number>(100000)
  const [currentInputValues, setCurrentInputValues] = useState<number[] | null>(null)
  const [fixturePayload, setFixturePayload] = useState<FixturePayload | null>(null)

  const hasWebGpuOnMainThread = useMemo<boolean>(() => 'gpu' in navigator, [])

  useEffect(() => {
    const worker = new Worker(new URL('./styleTransfer.worker.ts', import.meta.url), { type: 'module' })
    workerRef.current = worker

    const handleMessage = (event: MessageEvent<WorkerResponse>): void => {
      const payload = event.data
      switch (payload.type) {
        case 'pong':
          setWorkerStatus(`Worker responded at ${new Date(payload.timestamp).toISOString()}.`)
          break
        case 'webgpu-init-result':
          setWebGpuStatus(payload.ok ? payload.message : `WebGPU fallback: ${payload.message}`)
          break
        case 'run-style-transfer-result':
          if (payload.ok) {
            const values = payload.finalValues
            setCurrentInputValues(values)
            setStepCount((prev) => prev + payload.losses.length)
            const size = resolution
            const canvas = document.createElement('canvas')
            canvas.width = size
            canvas.height = size
            const ctx = canvas.getContext('2d')
            if (ctx !== null) {
              const imageData = ctx.createImageData(size, size)
              const area = size * size
              for (let i = 0; i < area; i += 1) {
                imageData.data[i * 4] = Math.round(values[i] * 255)
                imageData.data[i * 4 + 1] = Math.round(values[area + i] * 255)
                imageData.data[i * 4 + 2] = Math.round(values[2 * area + i] * 255)
                imageData.data[i * 4 + 3] = 255
              }
              ctx.putImageData(imageData, 0, 0)
              setOutputImageUrl(canvas.toDataURL('image/png'))
            }
          }
          break
        case 'error':
          setWebGpuStatus(`WebGPU fallback: ${payload.message}`)
          break
      }
    }

    worker.addEventListener('message', handleMessage)
    worker.postMessage({ type: 'ping', id: createMessageId() } satisfies WorkerRequest)
    worker.postMessage({ type: 'init-webgpu', id: createMessageId() } satisfies WorkerRequest)

    void (async (): Promise<void> => {
      const [weightsRes, fixtureRes] = await Promise.all([
        fetch('/vgg19-phase3-full-pass/vgg19_conv0_to_conv28_weights.json'),
        fetch('/vgg19-phase3-full-pass/vgg19_phase3_full_pass_fixture.json'),
      ])
      if (!weightsRes.ok || !fixtureRes.ok) return
      const weights = await weightsRes.json() as Record<string, number[] | [number, number, number, number]>
      const fixture = await fixtureRes.json() as Omit<FixturePayload, 'weights'>
      setFixturePayload({ ...fixture, weights })
    })()

    return () => {
      worker.removeEventListener('message', handleMessage)
      worker.terminate()
    }
  }, [resolution])

  useEffect(() => {
    if (!isPlaying || stepCount >= maxSteps || workerRef.current === null || contentImage === null || styleImage === null || fixturePayload === null || currentInputValues === null) return
    workerRef.current.postMessage({
      type: 'run-style-transfer',
      id: createMessageId(),
      inputShape: [1, 3, resolution, resolution],
      inputImageValues: currentInputValues,
      contentImageValues: contentImage.values,
      styleImageValues: styleImage.values,
      mean: fixturePayload.mean,
      std: fixturePayload.std,
      styleLayerIndices: fixturePayload.styleLayerIndices,
      contentLayerIndex: fixturePayload.contentLayerIndex,
      weights: fixturePayload.weights,
      contentWeight,
      styleWeight,
      learningRate,
      steps: 1,
    } satisfies WorkerRequest)
  }, [isPlaying, stepCount, maxSteps, resolution, contentImage, styleImage, fixturePayload, currentInputValues, contentWeight, styleWeight, learningRate])

  const loadImage = async (file: File, setter: (value: LoadedImage) => void): Promise<void> => {
    const previewUrl = URL.createObjectURL(file)
    const img = new Image()
    img.src = previewUrl
    await img.decode()
    setter({ fileName: file.name, width: img.width, height: img.height, previewUrl, values: createImageValues(img, resolution) })
  }

  const start = (): void => {
    if (contentImage === null || styleImage === null) return
    setCurrentInputValues(contentImage.values)
    setOutputImageUrl(contentImage.previewUrl)
    setStepCount(0)
    setIsPlaying(true)
  }

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-4 px-6 py-8 text-slate-100">
      <h1 className="text-3xl font-semibold">WebGPU Style Transfer — Phase 6 UI</h1>
      <p className="rounded border border-slate-700 bg-slate-900 p-3">{workerStatus}</p>
      <p className="rounded border border-slate-700 bg-slate-900 p-3">GPU status: {webGpuStatus}</p>
      <section className="grid grid-cols-1 gap-4 rounded border border-slate-700 bg-slate-900/60 p-4 md:grid-cols-2">
        <label>Content image<input type="file" accept="image/*" onChange={(e) => { const file = e.target.files?.[0]; if (file !== undefined) void loadImage(file, setContentImage) }} /></label>
        <label>Style image<input type="file" accept="image/*" onChange={(e) => { const file = e.target.files?.[0]; if (file !== undefined) void loadImage(file, setStyleImage) }} /></label>
        <label>Resolution <input type="number" value={resolution} min={64} max={512} step={32} onChange={(e) => setResolution(Number(e.target.value))} /></label>
        <label>Max iterations <input type="number" value={maxSteps} min={1} max={500} onChange={(e) => setMaxSteps(Number(e.target.value))} /></label>
        <label>Learning rate <input type="number" step={0.001} value={learningRate} onChange={(e) => setLearningRate(Number(e.target.value))} /></label>
        <label>Content weight <input type="number" step={0.1} value={contentWeight} onChange={(e) => setContentWeight(Number(e.target.value))} /></label>
        <label>Style weight <input type="number" step={1000} value={styleWeight} onChange={(e) => setStyleWeight(Number(e.target.value))} /></label>
      </section>
      <section className="h-[420px] rounded border border-slate-700 bg-black/50">
        <Canvas camera={{ position: [0, 0, 3] }}>
          {contentImage !== null ? <ImagePlane src={contentImage.previewUrl} x={-2} /> : null}
          {styleImage !== null ? <ImagePlane src={styleImage.previewUrl} x={0} /> : null}
          {outputImageUrl !== null ? <ImagePlane src={outputImageUrl} x={2} /> : null}
        </Canvas>
      </section>
      <section className="flex items-center gap-3">
        <button className="rounded bg-indigo-500 px-4 py-2" onClick={start}>Start</button>
        <button className="rounded bg-slate-700 px-4 py-2" onClick={() => setIsPlaying((prev) => !prev)}>{isPlaying ? 'Pause' : 'Play'}</button>
        <span>Iterations: {stepCount}/{maxSteps}</span>
      </section>
      <section className="rounded border border-dashed border-slate-500 p-4 text-slate-300">
        Space reserved: “How we made it” blog post plug.
      </section>
      {!hasWebGpuOnMainThread ? <p className="rounded border border-amber-500/50 bg-amber-500/10 p-4 text-amber-200">Main thread WebGPU unavailable.</p> : null}
    </main>
  )
}

export default App
