import { expect, test } from '@playwright/test'
import type { WorkerRequest, WorkerResponse } from '../src/types'

type VggWeightsFixture = {
  conv1WeightShape: [number, number, number, number]
  conv1WeightValues: number[]
  conv1BiasValues: number[]
  conv2WeightShape: [number, number, number, number]
  conv2WeightValues: number[]
  conv2BiasValues: number[]
}

type VggCaseFixture = {
  inputValues: number[]
  normalizedValues: number[]
  mean: [number, number, number]
  std: [number, number, number]
  expectedShape: [number, number, number, number]
  expectedValues: number[]
}

test('phase 2 vgg19 truncated parity through first pool', async ({ page }) => {
  await page.goto('/')
  const artifact = await page.evaluate(async () => {
    const loadJson = async (url: string): Promise<unknown | null> => {
      const response = await fetch(url)
      if (!response.ok) return null
      const text = await response.text()
      try {
        return JSON.parse(text)
      } catch (_error) {
        return null
      }
    }
    const weights = await loadJson('/vgg19-first-pool/vgg19_first_pool_weights.json')
    if (weights === null) return { ok: false as const, reason: 'weights-missing' as const }
    const caseData = await loadJson('/vgg19-first-pool/vgg19_first_pool_case_madeira16.json')
    if (caseData === null) return { ok: false as const, reason: 'case-missing' as const }
    return {
      ok: true as const,
      weights,
      caseData,
    }
  })

  test.skip(!artifact.ok, 'Missing VGG19 export fixtures. Run python-reference/export_vgg19_first_pool.py first.')
  if (!artifact.ok) return

  const { weights, caseData } = artifact as { ok: true; weights: VggWeightsFixture; caseData: VggCaseFixture }
  const result = await page.evaluate(async ({ weightsArg, caseArg }) => {
    const worker = new Worker(new URL('/src/styleTransfer.worker.ts', window.location.origin), { type: 'module' })
    const ask = (payload: WorkerRequest): Promise<WorkerResponse> => new Promise((resolve) => {
      const handler = (event: MessageEvent<WorkerResponse>): void => {
        if (event.data.id === payload.id) { worker.removeEventListener('message', handler); resolve(event.data) }
      }
      worker.addEventListener('message', handler)
      worker.postMessage(payload)
    })

    const init = await ask({ type: 'init-webgpu', id: 'phase2-vgg-init' })

    const vectorValues = (response: WorkerResponse, label: string): number[] => {
      if (response.type !== 'tensor-op-result' || !response.ok || !('values' in response)) {
        throw new Error(`${label} did not return tensor values`)
      }
      return response.values
    }
    const norm = await ask({
      type: 'tensor-op', id: 'phase2-vgg-norm', op: 'normalize-forward',
      input: { shape: caseArg.inputShape, values: caseArg.inputValues }, mean: caseArg.mean, std: caseArg.std,
    })

    const conv1 = await ask({
      type: 'tensor-op', id: 'phase2-vgg-conv1', op: 'conv2d-forward',
      input: { shape: caseArg.inputShape, values: caseArg.normalizedValues },
      weight: { shape: weightsArg.conv1WeightShape, values: weightsArg.conv1WeightValues },
      bias: weightsArg.conv1BiasValues,
    })
    const conv1Values = vectorValues(conv1, 'conv1')
    const relu1 = await ask({ type: 'tensor-op', id: 'phase2-vgg-relu1', op: 'relu-forward', input: { shape: [1, 64, 16, 16], values: conv1Values } })
    const relu1Values = vectorValues(relu1, 'relu1')
    const conv2 = await ask({
      type: 'tensor-op', id: 'phase2-vgg-conv2', op: 'conv2d-forward',
      input: { shape: [1, 64, 16, 16], values: relu1Values },
      weight: { shape: weightsArg.conv2WeightShape, values: weightsArg.conv2WeightValues },
      bias: weightsArg.conv2BiasValues,
    })
    const conv2Values = vectorValues(conv2, 'conv2')
    const relu2 = await ask({ type: 'tensor-op', id: 'phase2-vgg-relu2', op: 'relu-forward', input: { shape: [1, 64, 16, 16], values: conv2Values } })
    const relu2Values = vectorValues(relu2, 'relu2')
    const pool1 = await ask({ type: 'tensor-op', id: 'phase2-vgg-pool1', op: 'maxpool2d-forward', input: { shape: [1, 64, 16, 16], values: relu2Values } })

    worker.terminate()
    return { init, norm, pool1 }
  }, { weightsArg: weights, caseArg: caseData })

  expect(result.init.type).toBe('webgpu-init-result')
  if (result.norm.type !== 'tensor-op-result' || !result.norm.ok || !('values' in result.norm)) throw new Error('normalize-forward failed')
  if (result.pool1.type !== 'tensor-op-result' || !result.pool1.ok || !('values' in result.pool1)) throw new Error('vgg run failed')

  for (let i = 0; i < caseData.expectedValues.length; i += 1) {
    expect(result.pool1.values[i]).toBeCloseTo(caseData.expectedValues[i], 4)
  }
})
