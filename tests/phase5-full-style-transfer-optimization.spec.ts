import { expect, test } from '@playwright/test'
import type { WorkerRequest, WorkerResponse } from '../src/types'

test('phase 5 full style transfer endpoint returns losses', async ({ page }) => {
  test.setTimeout(300000)
  await page.goto('/')
  const result = await page.evaluate(async () => {
    const loadJson = async <T>(url: string): Promise<T | null> => {
      const response = await fetch(url)
      if (!response.ok) return null
      const text = await response.text()
      try { return JSON.parse(text) as T } catch (_error) { return null }
    }
    const weights = await loadJson<Record<string, number[] | [number, number, number, number]>>('/vgg19-phase3-full-pass/vgg19_conv0_to_conv28_weights.json')
    const fixture = await loadJson<any>('/vgg19-phase3-full-pass/vgg19_phase3_full_pass_fixture.json')
    if (weights === null || fixture === null) return { ok: false as const, reason: 'missing-fixtures' as const }
    const worker = new Worker(new URL('/src/styleTransfer.worker.ts', window.location.origin), { type: 'module' })
    const ask = (payload: WorkerRequest): Promise<WorkerResponse> => new Promise((resolve) => {
      const handler = (event: MessageEvent<WorkerResponse>): void => {
        if (event.data.id === payload.id) { worker.removeEventListener('message', handler); resolve(event.data) }
      }
      worker.addEventListener('message', handler)
      worker.postMessage(payload)
    })
    await ask({ type: 'init-webgpu', id: 'phase5-full-init' })
    const out = await ask({ type: 'run-style-transfer', id: 'phase5-full-run', inputShape: fixture.inputShape, inputImageValues: fixture.inputImageValues, contentImageValues: fixture.contentImageValues, styleImageValues: fixture.styleImageValues, mean: fixture.mean, std: fixture.std, styleLayerIndices: fixture.styleLayerIndices, contentLayerIndex: fixture.contentLayerIndex, weights, contentWeight: 1, styleWeight: 1, learningRate: 0.01, steps: 5 })
    worker.terminate()
    if (out.type !== 'run-style-transfer-result') return { ok: false as const, reason: 'wrong-response' as const }
    if (!out.ok) return { ok: false as const, reason: 'worker-failed' as const, message: out.message }
    return { ok: true as const, losses: out.losses }
  })
  test.skip(!result.ok && result.reason === 'missing-fixtures', 'Missing phase3 full-pass fixtures. Run python-reference/export_vgg19_phase3_full_pass.py first.')
  if (!result.ok && result.reason !== 'missing-fixtures') throw new Error(result.reason === 'worker-failed' ? result.message : result.reason)
  if (!result.ok) return
  expect(result.losses.length).toBe(5)
  expect(result.losses.every((loss) => Number.isFinite(loss))).toBeTruthy()
})

test('phase 5 run-style-transfer first-step gradient matches pytorch oracle', async ({ page }) => {
  test.setTimeout(300000)
  await page.goto('/')
  const result = await page.evaluate(async () => {
    const loadJson = async <T>(url: string): Promise<T | null> => {
      const response = await fetch(url)
      if (!response.ok) return null
      const text = await response.text()
      try { return JSON.parse(text) as T } catch (_error) { return null }
    }
    const weights = await loadJson<Record<string, number[] | [number, number, number, number]>>('/vgg19-phase3-full-pass/vgg19_conv0_to_conv28_weights.json')
    const fixture = await loadJson<any>('/vgg19-phase3-full-pass/vgg19_phase3_full_pass_fixture.json')
    if (weights === null || fixture === null) return { ok: false as const, reason: 'missing-fixtures' as const }
    if (fixture.expectedGradients?.total === undefined) return { ok: false as const, reason: 'missing-gradient-fixture' as const }
    const worker = new Worker(new URL('/src/styleTransfer.worker.ts', window.location.origin), { type: 'module' })
    const ask = (payload: WorkerRequest): Promise<WorkerResponse> => new Promise((resolve) => {
      const handler = (event: MessageEvent<WorkerResponse>): void => {
        if (event.data.id === payload.id) { worker.removeEventListener('message', handler); resolve(event.data) }
      }
      worker.addEventListener('message', handler)
      worker.postMessage(payload)
    })
    await ask({ type: 'init-webgpu', id: 'phase5-grad-init' })
    const learningRate = 1e-5
    const out = await ask({ type: 'run-style-transfer', id: 'phase5-grad-run', inputShape: fixture.inputShape, inputImageValues: fixture.inputImageValues, contentImageValues: fixture.contentImageValues, styleImageValues: fixture.styleImageValues, mean: fixture.mean, std: fixture.std, styleLayerIndices: fixture.styleLayerIndices, contentLayerIndex: fixture.contentLayerIndex, weights, contentWeight: 1, styleWeight: 1, learningRate, steps: 1 })
    worker.terminate()
    if (out.type !== 'run-style-transfer-result') return { ok: false as const, reason: 'wrong-response' as const }
    if (!out.ok) return { ok: false as const, reason: 'worker-failed' as const, message: out.message }
    const gradOracle = fixture.expectedGradients.total as number[]
    const gradObserved = out.finalValues.map((value: number, index: number) => (fixture.inputImageValues[index] - value) / learningRate)
    let maxDiff = 0
    let meanDiff = 0
    for (let i = 0; i < gradObserved.length; i += 1) {
      const diff = Math.abs(gradObserved[i] - gradOracle[i])
      maxDiff = Math.max(maxDiff, diff)
      meanDiff += diff
    }
    meanDiff /= gradObserved.length
    return { ok: true as const, maxDiff, meanDiff }
  })
  test.skip(!result.ok && result.reason === 'missing-fixtures', 'Missing phase3 full-pass fixtures. Run python-reference/export_vgg19_phase3_full_pass.py first.')
  test.skip(!result.ok && result.reason === 'missing-gradient-fixture', 'Fixture is missing gradients. Re-run python-reference/export_vgg19_phase3_full_pass.py.')
  if (!result.ok && result.reason !== 'missing-fixtures' && result.reason !== 'missing-gradient-fixture') throw new Error(result.reason === 'worker-failed' ? result.message : result.reason)
  if (!result.ok) return
  expect(result.maxDiff).toBeLessThan(2e-1)
  expect(result.meanDiff).toBeLessThan(2e-2)
})


test('phase 5 fused conv+relu op matches separate ops', async ({ page }) => {
  test.setTimeout(120000)
  await page.goto('/')
  const result = await page.evaluate(async () => {
    const worker = new Worker(new URL('/src/styleTransfer.worker.ts', window.location.origin), { type: 'module' })
    const ask = (payload: WorkerRequest): Promise<WorkerResponse> => new Promise((resolve) => {
      const handler = (event: MessageEvent<WorkerResponse>): void => {
        if (event.data.id === payload.id) { worker.removeEventListener('message', handler); resolve(event.data) }
      }
      worker.addEventListener('message', handler)
      worker.postMessage(payload)
    })
    await ask({ type: 'init-webgpu', id: 'phase5-fused-init' })
    const input = Array.from({ length: 3 * 8 * 8 }, (_, i) => Math.sin(i * 0.13) * 0.5)
    const weight = Array.from({ length: 4 * 3 * 3 * 3 }, (_, i) => Math.cos(i * 0.07) * 0.1)
    const bias = [0.01, -0.02, 0.03, -0.04]
    const conv = await ask({ type: 'tensor-op', id: 'phase5-fused-conv', op: 'conv2d-forward', input: { shape: [1, 3, 8, 8], values: input }, weight: { shape: [4, 3, 3, 3], values: weight }, bias })
    if (conv.type !== 'tensor-op-result' || !conv.ok || !('values' in conv)) return { ok: false as const, reason: 'conv-failed' as const }
    const relu = await ask({ type: 'tensor-op', id: 'phase5-fused-relu', op: 'relu-forward', input: { shape: [1, 4, 8, 8], values: conv.values } })
    const fused = await ask({ type: 'tensor-op', id: 'phase5-fused-fused', op: 'conv2d-relu-forward', input: { shape: [1, 3, 8, 8], values: input }, weight: { shape: [4, 3, 3, 3], values: weight }, bias })
    worker.terminate()
    if (relu.type !== 'tensor-op-result' || !relu.ok || !('values' in relu)) return { ok: false as const, reason: 'relu-failed' as const }
    if (fused.type !== 'tensor-op-result' || !fused.ok || !('values' in fused)) return { ok: false as const, reason: 'fused-failed' as const }
    let maxDiff = 0
    for (let i = 0; i < relu.values.length; i += 1) maxDiff = Math.max(maxDiff, Math.abs(relu.values[i] - fused.values[i]))
    return { ok: true as const, maxDiff }
  })
  if (!result.ok) throw new Error(result.reason)
  expect(result.maxDiff).toBeLessThan(1e-4)
})


test('phase 5 super fused forward path matches baseline for single step', async ({ page }) => {
  test.setTimeout(300000)
  await page.goto('/')
  const result = await page.evaluate(async () => {
    const loadJson = async <T>(url: string): Promise<T | null> => {
      const response = await fetch(url)
      if (!response.ok) return null
      const text = await response.text()
      try { return JSON.parse(text) as T } catch (_error) { return null }
    }
    const weights = await loadJson<Record<string, number[] | [number, number, number, number]>>('/vgg19-phase3-full-pass/vgg19_conv0_to_conv28_weights.json')
    const fixture = await loadJson<any>('/vgg19-phase3-full-pass/vgg19_phase3_full_pass_fixture.json')
    if (weights === null || fixture === null) return { ok: false as const, reason: 'missing-fixtures' as const }
    const worker = new Worker(new URL('/src/styleTransfer.worker.ts', window.location.origin), { type: 'module' })
    const ask = (payload: WorkerRequest): Promise<WorkerResponse> => new Promise((resolve) => {
      const handler = (event: MessageEvent<WorkerResponse>): void => {
        if (event.data.id === payload.id) { worker.removeEventListener('message', handler); resolve(event.data) }
      }
      worker.addEventListener('message', handler)
      worker.postMessage(payload)
    })
    await ask({ type: 'init-webgpu', id: 'phase5-super-init' })
    const common = { type: 'run-style-transfer' as const, inputShape: fixture.inputShape, inputImageValues: fixture.inputImageValues, contentImageValues: fixture.contentImageValues, styleImageValues: fixture.styleImageValues, mean: fixture.mean, std: fixture.std, styleLayerIndices: fixture.styleLayerIndices, contentLayerIndex: fixture.contentLayerIndex, weights, contentWeight: 1, styleWeight: 1, learningRate: 1e-5, steps: 1, optimizer: 'sgd' as const }
    const base = await ask({ ...common, id: 'phase5-super-base', fusedOps: false, superFusedOps: false })
    const superFused = await ask({ ...common, id: 'phase5-super-fused', fusedOps: true, superFusedOps: true })
    worker.terminate()
    if (base.type !== 'run-style-transfer-result') return { ok: false as const, reason: 'base-wrong-response' as const, responseType: base.type }
    if (!base.ok) return { ok: false as const, reason: 'base-failed' as const, message: base.message }
    if (superFused.type !== 'run-style-transfer-result') return { ok: false as const, reason: 'super-wrong-response' as const, responseType: superFused.type }
    if (!superFused.ok) return { ok: false as const, reason: 'super-failed' as const, message: superFused.message }
    let maxDiff = 0
    let maxDiffIndex = -1
    for (let i = 0; i < base.finalValues.length; i += 1) {
      const diff = Math.abs(base.finalValues[i] - superFused.finalValues[i])
      if (diff > maxDiff) { maxDiff = diff; maxDiffIndex = i }
    }
    return { ok: true as const, maxDiff, maxDiffIndex, baseLoss: base.losses[0] ?? null, superLoss: superFused.losses[0] ?? null, baseStats: base.stats, superStats: superFused.stats }
  })
  test.skip(!result.ok && result.reason === 'missing-fixtures', 'Missing phase3 full-pass fixtures.')
  if (!result.ok && result.reason !== 'missing-fixtures') {
    throw new Error(`${result.reason}${'message' in result && result.message !== undefined ? `: ${result.message}` : ''}${'responseType' in result && result.responseType !== undefined ? ` (responseType=${String(result.responseType)})` : ''}`)
  }
  if (!result.ok) return
  expect(result.maxDiff).toBeLessThan(2e-3)
})
