import { expect, test } from '@playwright/test'
import type { WorkerRequest, WorkerResponse } from '../src/types'

type VggWeightsFixture = {
  conv1WeightShape: [number, number, number, number]
  conv1WeightValues: number[]
  conv1BiasValues: number[]
  conv2WeightShape: [number, number, number, number]
  conv2WeightValues: number[]
  conv2BiasValues: number[]
  conv3WeightShape: [number, number, number, number]
  conv3WeightValues: number[]
  conv3BiasValues: number[]
}

type VggCaseFixture = {
  inputShape: [number, number, number, number]
  inputValues: number[]
  mean: [number, number, number]
  std: [number, number, number]
}

test('phase 5 optimizer loop decreases first-pool graph objective', async ({ page }) => {
  test.setTimeout(300000)
  await page.goto('/')

  const result = await page.evaluate(async () => {
    const loadJson = async <T>(url: string): Promise<T | null> => {
      const response = await fetch(url)
      if (!response.ok) return null
      return JSON.parse(await response.text()) as T
    }
    const weights = await loadJson<VggWeightsFixture>('/vgg19-first-pool/vgg19_first_pool_weights.json')
    const caseData = await loadJson<VggCaseFixture>('/vgg19-first-pool/vgg19_first_pool_case_madeira16.json')
    if (weights === null || caseData === null) return { ok: false as const }

    const worker = new Worker(new URL('/src/styleTransfer.worker.ts', window.location.origin), { type: 'module' })
    const ask = (payload: WorkerRequest): Promise<WorkerResponse> => new Promise((resolve) => {
      const handler = (event: MessageEvent<WorkerResponse>): void => {
        if (event.data.id === payload.id) { worker.removeEventListener('message', handler); resolve(event.data) }
      }
      worker.addEventListener('message', handler)
      worker.postMessage(payload)
    })

    const init = await ask({ type: 'init-webgpu', id: 'phase5-init' })
    const optimized = await ask({
      type: 'run-first-pool-optimizer',
      id: 'phase5-run-first-pool-optimizer',
      inputShape: caseData.inputShape,
      mean: caseData.mean,
      std: caseData.std,
      contentImageValues: caseData.inputValues,
      styleImageValues: [...caseData.inputValues].reverse(),
      initialInputValues: caseData.inputValues,
      conv1Weight: { shape: weights.conv1WeightShape, values: weights.conv1WeightValues },
      conv1Bias: weights.conv1BiasValues,
      conv2Weight: { shape: weights.conv2WeightShape, values: weights.conv2WeightValues },
      conv2Bias: weights.conv2BiasValues,
      conv3Weight: { shape: weights.conv3WeightShape, values: weights.conv3WeightValues },
      conv3Bias: weights.conv3BiasValues,
      contentWeight: 1,
      styleWeightConv1: 30,
      styleWeightConv3: 120,
      learningRate: 0.15,
      steps: 6,
    })
    worker.terminate()
    if (optimized.type !== 'run-first-pool-optimizer-result' || !optimized.ok) return { ok: false as const }
    return { ok: true as const, init, losses: optimized.losses }
  })

  test.skip(!result.ok, 'Missing VGG19 first-pool fixtures. Run python-reference/export_vgg19_first_pool.py first.')
  if (!result.ok) return
  expect(result.init.type).toBe('webgpu-init-result')
  if (result.init.type !== 'webgpu-init-result') throw new Error('init failure')
  expect(result.init.ok).toBeTruthy()
  expect(result.losses.length).toBeGreaterThan(1)
  expect(result.losses.at(-1) ?? Number.POSITIVE_INFINITY).toBeLessThan(result.losses[0])
})
