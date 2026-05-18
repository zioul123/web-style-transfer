import { expect, test } from '@playwright/test'
import type { WorkerRequest, WorkerResponse } from '../src/types'

type Phase3Fixture = {
  inputShape: [number, number, number, number]
  inputValues: number[]
  targetShape: [number, number, number, number]
  targetValues: number[]
  expectedGram: number[]
  expectedContentLoss: number
  expectedStyleLoss: number
}

test('phase 3 reshape/gram/content/style-loss parity', async ({ page }) => {
  await page.goto('/')
  const fixture: Phase3Fixture = {
    inputShape: [1, 2, 2, 2],
    inputValues: [1, 2, 3, 4, 5, 6, 7, 8],
    targetShape: [1, 2, 2, 2],
    targetValues: [1.5, 1, 2.5, 3, 4, 5, 6, 7],
    expectedGram: [3.75, 8.75, 8.75, 21.75],
    expectedContentLoss: 0.8125,
    expectedStyleLoss: 13.6494140625,
  }

  const result = await page.evaluate(async (fixtureArg) => {
    const worker = new Worker(new URL('/src/styleTransfer.worker.ts', window.location.origin), { type: 'module' })
    const ask = (payload: WorkerRequest): Promise<WorkerResponse> => new Promise((resolve) => {
      const handler = (event: MessageEvent<WorkerResponse>): void => {
        if (event.data.id === payload.id) { worker.removeEventListener('message', handler); resolve(event.data) }
      }
      worker.addEventListener('message', handler)
      worker.postMessage(payload)
    })

    const init = await ask({ type: 'init-webgpu', id: 'phase3-init' })
    const reshape = await ask({ type: 'tensor-op', id: 'phase3-reshape', op: 'reshape-chw-flatten', input: { shape: fixtureArg.inputShape, values: fixtureArg.inputValues } })
    const gram = await ask({ type: 'tensor-op', id: 'phase3-gram', op: 'gram-matrix', input: { shape: fixtureArg.inputShape, values: fixtureArg.inputValues } })
    const contentLoss = await ask({ type: 'tensor-op', id: 'phase3-content-loss', op: 'content-loss', input: { shape: fixtureArg.inputShape, values: fixtureArg.inputValues }, target: { shape: fixtureArg.targetShape, values: fixtureArg.targetValues } })
    const styleLoss = await ask({ type: 'tensor-op', id: 'phase3-style-loss', op: 'style-loss', input: { shape: fixtureArg.inputShape, values: fixtureArg.inputValues }, target: { shape: fixtureArg.targetShape, values: fixtureArg.targetValues } })
    worker.terminate()
    return { init, reshape, gram, contentLoss, styleLoss }
  }, fixture)

  expect(result.init.type).toBe('webgpu-init-result')
  if (result.init.type !== 'webgpu-init-result') throw new Error('Expected webgpu-init-result')
  expect(result.init.ok).toBeTruthy()

  if (result.reshape.type !== 'tensor-op-result' || !result.reshape.ok || !('values' in result.reshape)) throw new Error('reshape failed')
  expect(result.reshape.values).toEqual(fixture.inputValues)

  if (result.gram.type !== 'tensor-op-result' || !result.gram.ok || !('values' in result.gram)) throw new Error('gram failed')
  fixture.expectedGram.forEach((value, index) => expect(result.gram.values[index]).toBeCloseTo(value, 5))

  if (result.contentLoss.type !== 'tensor-op-result' || !result.contentLoss.ok || !('scalar' in result.contentLoss)) throw new Error('content loss failed')
  expect(result.contentLoss.scalar).toBeCloseTo(fixture.expectedContentLoss, 6)

  if (result.styleLoss.type !== 'tensor-op-result' || !result.styleLoss.ok || !('scalar' in result.styleLoss)) throw new Error('style loss failed')
  expect(result.styleLoss.scalar).toBeCloseTo(fixture.expectedStyleLoss, 6)
})
