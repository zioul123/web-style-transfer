import { Canvas, useLoader } from '@react-three/fiber'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { ReactElement } from 'react'
import { TextureLoader } from 'three'
import type { WorkerRequest, WorkerResponse } from './types'

type ResolutionPreset = 128 | 256

type WeightShape = [number, number, number, number]

type FirstPoolWeightsArtifact = {
  conv1WeightShape: WeightShape
  conv1WeightValues: number[]
  conv1BiasValues: number[]
  conv2WeightShape: WeightShape
  conv2WeightValues: number[]
  conv2BiasValues: number[]
  conv3WeightShape: WeightShape
  conv3WeightValues: number[]
  conv3BiasValues: number[]
}

type WorkerWeightTensor = { shape: WeightShape; values: number[] }

type FirstPoolWeights = {
  conv1Weight: WorkerWeightTensor
  conv1Bias: number[]
  conv2Weight: WorkerWeightTensor
  conv2Bias: number[]
  conv3Weight: WorkerWeightTensor
  conv3Bias: number[]
}

const createMessageId = (): string => `${Date.now()}-${Math.random().toString(16).slice(2)}`

const readFileAsImage = (file: File): Promise<HTMLImageElement> =>
  new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file)
    const image = new Image()
    image.onload = (): void => {
      URL.revokeObjectURL(objectUrl)
      resolve(image)
    }
    image.onerror = (): void => {
      URL.revokeObjectURL(objectUrl)
      reject(new Error(`Failed to load ${file.name}`))
    }
    image.src = objectUrl
  })

const imageToTensorValues = (image: HTMLImageElement, resolution: ResolutionPreset): number[] => {
  const canvas = document.createElement('canvas')
  canvas.width = resolution
  canvas.height = resolution
  const context = canvas.getContext('2d')
  if (context === null) throw new Error('2D canvas unavailable.')

  context.drawImage(image, 0, 0, resolution, resolution)
  const rgba = context.getImageData(0, 0, resolution, resolution).data
  const spatial = resolution * resolution
  const chw = new Float32Array(3 * spatial)

  for (let i = 0; i < spatial; i += 1) {
    chw[i] = rgba[i * 4] / 255
    chw[spatial + i] = rgba[i * 4 + 1] / 255
    chw[spatial * 2 + i] = rgba[i * 4 + 2] / 255
  }

  return Array.from(chw)
}

const tensorValuesToDataUrl = (values: number[], resolution: ResolutionPreset): string => {
  const canvas = document.createElement('canvas')
  canvas.width = resolution
  canvas.height = resolution
  const context = canvas.getContext('2d')
  if (context === null) throw new Error('2D canvas unavailable.')

  const spatial = resolution * resolution
  const imageData = context.createImageData(resolution, resolution)
  for (let i = 0; i < spatial; i += 1) {
    imageData.data[i * 4] = Math.max(0, Math.min(255, Math.round(values[i] * 255)))
    imageData.data[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(values[spatial + i] * 255)))
    imageData.data[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(values[spatial * 2 + i] * 255)))
    imageData.data[i * 4 + 3] = 255
  }
  context.putImageData(imageData, 0, 0)
  return canvas.toDataURL('image/png')
}

const toFirstPoolWeights = (artifact: FirstPoolWeightsArtifact): FirstPoolWeights => ({
  conv1Weight: { shape: artifact.conv1WeightShape, values: artifact.conv1WeightValues },
  conv1Bias: artifact.conv1BiasValues,
  conv2Weight: { shape: artifact.conv2WeightShape, values: artifact.conv2WeightValues },
  conv2Bias: artifact.conv2BiasValues,
  conv3Weight: { shape: artifact.conv3WeightShape, values: artifact.conv3WeightValues },
  conv3Bias: artifact.conv3BiasValues,
})

const Plane = ({ url, x }: { url: string; x: number }): ReactElement => {
  const texture = useLoader(TextureLoader, url)
  return (
    <mesh position={[x, 0, 0]}>
      <planeGeometry args={[1.75, 1.75]} />
      <meshBasicMaterial map={texture} />
    </mesh>
  )
}

function App() {
  const [workerStatus, setWorkerStatus] = useState<string>('Worker starting…')
  const [gpuStatus, setGpuStatus] = useState<string>('Checking WebGPU support…')
  const [gpuInfo, setGpuInfo] = useState<string>('GPU details pending…')
  const [resolution, setResolution] = useState<ResolutionPreset>(128)
  const [stepsPerChunk, setStepsPerChunk] = useState<number>(4)
  const [contentWeight, setContentWeight] = useState<number>(1)
  const [styleWeight, setStyleWeight] = useState<number>(1000)
  const [learningRate, setLearningRate] = useState<number>(0.02)
  const [iterations, setIterations] = useState<number>(0)
  const [isRunning, setIsRunning] = useState<boolean>(false)
  const [lastLoss, setLastLoss] = useState<number | null>(null)
  const [contentImage, setContentImage] = useState<string | null>(null)
  const [styleImage, setStyleImage] = useState<string | null>(null)
  const [outputImage, setOutputImage] = useState<string | null>(null)
  const [weights, setWeights] = useState<FirstPoolWeights | null>(null)
  const [contentTensor, setContentTensor] = useState<number[] | null>(null)
  const [styleTensor, setStyleTensor] = useState<number[] | null>(null)
  const [inputTensor, setInputTensor] = useState<number[] | null>(null)

  const workerRef = useRef<Worker | null>(null)
  const hasWebGpuOnMainThread = useMemo<boolean>(() => 'gpu' in navigator, [])

  useEffect(() => {
    void fetch('/vgg19-first-pool/vgg19_first_pool_weights.json')
      .then(async (response) => await response.json())
      .then((json: FirstPoolWeightsArtifact) => {
        setWeights(toFirstPoolWeights(json))
      })
      .catch(() => {
        setWorkerStatus('Failed to load first-pool VGG weights.')
      })
  }, [])

  useEffect(() => {
    const worker = new Worker(new URL('./styleTransfer.worker.ts', import.meta.url), { type: 'module' })
    workerRef.current = worker

    const onMessage = (event: MessageEvent<WorkerResponse>): void => {
      const payload = event.data
      if (payload.type === 'pong') setWorkerStatus(`Worker ping OK @ ${new Date(payload.timestamp).toISOString()}`)
      if (payload.type === 'webgpu-init-result') setGpuStatus(payload.ok ? payload.message : `Fallback: ${payload.message}`)
      if (payload.type === 'error') setWorkerStatus(payload.message)

      if (payload.type === 'run-first-pool-optimizer-result') {
        if (!payload.ok) {
          setWorkerStatus(payload.message)
          setIsRunning(false)
          return
        }
        setInputTensor(payload.finalValues)
        setOutputImage(tensorValuesToDataUrl(payload.finalValues, resolution))
        setIterations((value) => value + stepsPerChunk)
        if (payload.losses.length > 0) setLastLoss(payload.losses[payload.losses.length - 1])
      }
    }

    worker.addEventListener('message', onMessage)
    worker.postMessage({ type: 'ping', id: createMessageId() } satisfies WorkerRequest)
    worker.postMessage({ type: 'init-webgpu', id: createMessageId() } satisfies WorkerRequest)

    return () => {
      worker.removeEventListener('message', onMessage)
      worker.terminate()
      workerRef.current = null
    }
  }, [resolution, stepsPerChunk])

  useEffect(() => {
    void (async (): Promise<void> => {
      if (!hasWebGpuOnMainThread) {
        setGpuInfo('No WebGPU adapter on main thread.')
        return
      }
      const adapter = await navigator.gpu.requestAdapter()
      if (adapter === null) {
        setGpuInfo('Adapter request failed.')
        return
      }
      const maybeAdapter = adapter as GPUAdapter & {
        requestAdapterInfo?: () => Promise<{ vendor?: string; architecture?: string; device?: string; description?: string }>
      }
      if (maybeAdapter.requestAdapterInfo === undefined) {
        setGpuInfo('Adapter info API unavailable in this browser build.')
        return
      }
      const info = await maybeAdapter.requestAdapterInfo()
      setGpuInfo(`Vendor: ${info.vendor ?? 'unknown'} | Architecture: ${info.architecture ?? 'unknown'} | Device: ${info.device ?? 'unknown'} | Description: ${info.description ?? 'unknown'}`)
    })()
  }, [hasWebGpuOnMainThread])

  useEffect(() => {
    if (!isRunning || workerRef.current === null || weights === null || contentTensor === null || styleTensor === null || inputTensor === null) return

    workerRef.current.postMessage({
      type: 'run-first-pool-optimizer',
      id: createMessageId(),
      inputShape: [1, 3, resolution, resolution],
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
      contentImageValues: contentTensor,
      styleImageValues: styleTensor,
      initialInputValues: inputTensor,
      conv1Weight: weights.conv1Weight,
      conv1Bias: weights.conv1Bias,
      conv2Weight: weights.conv2Weight,
      conv2Bias: weights.conv2Bias,
      conv3Weight: weights.conv3Weight,
      conv3Bias: weights.conv3Bias,
      contentWeight,
      styleWeightConv1: styleWeight,
      styleWeightConv3: styleWeight,
      learningRate,
      steps: stepsPerChunk,
    } satisfies WorkerRequest)
  }, [contentTensor, contentWeight, inputTensor, isRunning, iterations, learningRate, resolution, stepsPerChunk, styleTensor, styleWeight, weights])

  const onUpload = async (event: React.ChangeEvent<HTMLInputElement>, target: 'content' | 'style'): Promise<void> => {
    const file = event.target.files?.[0]
    if (file === undefined) return

    const image = await readFileAsImage(file)
    const previewUrl = URL.createObjectURL(file)
    const tensor = imageToTensorValues(image, resolution)

    if (target === 'content') {
      setContentImage(previewUrl)
      setContentTensor(tensor)
      setInputTensor(tensor)
      setOutputImage(tensorValuesToDataUrl(tensor, resolution))
    } else {
      setStyleImage(previewUrl)
      setStyleTensor(tensor)
    }

    setIterations(0)
    setLastLoss(null)
  }

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-6 px-6 py-8 text-slate-100">
      <h1 className="text-3xl font-semibold">WebGPU Style Transfer — Phase 6 UI</h1>

      <section className="grid gap-4 rounded-xl border border-slate-700 bg-slate-900/60 p-4 md:grid-cols-2">
        <p>{workerStatus}</p>
        <p>{gpuStatus}</p>
        <p className="md:col-span-2 text-slate-300">{gpuInfo}</p>
      </section>

      <section className="grid gap-4 rounded-xl border border-slate-700 bg-slate-900/60 p-4 md:grid-cols-3">
        <label className="flex flex-col gap-1">Content image<input type="file" accept="image/*" onChange={(event) => void onUpload(event, 'content')} /></label>
        <label className="flex flex-col gap-1">Style image<input type="file" accept="image/*" onChange={(event) => void onUpload(event, 'style')} /></label>
        <label className="flex flex-col gap-1">Resolution<select value={resolution} onChange={(event) => setResolution(Number(event.target.value) as ResolutionPreset)}><option value={128}>128</option><option value={256}>256</option></select></label>
        <label className="flex flex-col gap-1">Chunk steps<input type="number" min={1} max={20} value={stepsPerChunk} onChange={(event) => setStepsPerChunk(Number(event.target.value))} /></label>
        <label className="flex flex-col gap-1">Style weight<input type="number" value={styleWeight} onChange={(event) => setStyleWeight(Number(event.target.value))} /></label>
        <label className="flex flex-col gap-1">Content weight<input type="number" value={contentWeight} onChange={(event) => setContentWeight(Number(event.target.value))} /></label>
        <label className="flex flex-col gap-1">Learning rate<input type="number" step={0.001} value={learningRate} onChange={(event) => setLearningRate(Number(event.target.value))} /></label>
      </section>

      <section className="flex items-center gap-4">
        <button className="rounded bg-emerald-500 px-4 py-2 font-semibold text-black" onClick={() => setIsRunning(true)} disabled={isRunning || contentTensor === null || styleTensor === null || weights === null}>▶ Play</button>
        <button className="rounded bg-amber-500 px-4 py-2 font-semibold text-black" onClick={() => setIsRunning(false)} disabled={!isRunning}>⏸ Pause</button>
        <p>Iterations: {iterations}</p>
        <p>Loss: {lastLoss === null ? '—' : lastLoss.toFixed(4)}</p>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <img className="aspect-square w-full rounded border border-slate-700 object-cover" src={contentImage ?? undefined} alt="Content" />
        <img className="aspect-square w-full rounded border border-slate-700 object-cover" src={styleImage ?? undefined} alt="Style" />
        <img className="aspect-square w-full rounded border border-slate-700 object-cover" src={outputImage ?? undefined} alt="Output" />
      </section>

      <section className="h-72 overflow-hidden rounded-xl border border-slate-700 bg-black/40">
        <Canvas camera={{ position: [0, 0, 3] }}>
          {contentImage !== null ? <Plane url={contentImage} x={-2} /> : null}
          {styleImage !== null ? <Plane url={styleImage} x={0} /> : null}
          {outputImage !== null ? <Plane url={outputImage} x={2} /> : null}
        </Canvas>
      </section>

      <section className="rounded-xl border border-dashed border-sky-400/50 bg-sky-400/5 p-4 text-sky-200">
        📝 Reserved: add “How we made it” blog post link/teaser here.
      </section>
    </main>
  )
}

export default App
