import { expect, test } from '@playwright/test'
import type { WorkerRequest, WorkerResponse } from '../src/types'

test('phase 3 full vgg19 pass parity through conv5_1 style + conv4_2 content losses', async ({ page }) => {
  test.setTimeout(300000)
  await page.goto('/')
  const artifactReady = await page.evaluate(async () => {
    const check = async (url: string): Promise<boolean> => (await fetch(url)).ok
    return await check('/vgg19-phase3-full-pass/vgg19_conv0_to_conv28_weights.json')
      && await check('/vgg19-phase3-full-pass/vgg19_phase3_full_pass_fixture.json')
  })

  test.skip(!artifactReady, 'Missing phase3 full-pass fixtures. Run python-reference/export_vgg19_phase3_full_pass.py first.')
  if (!artifactReady) return

  const result = await page.evaluate(async () => {
    const loadJson = async <T>(url: string): Promise<T> => JSON.parse(await (await fetch(url)).text()) as T
    const weights = await loadJson<Record<string, number[] | [number, number, number, number]>>('/vgg19-phase3-full-pass/vgg19_conv0_to_conv28_weights.json')
    const fixture = await loadJson<{
      inputShape: [number, number, number, number]
      inputImageValues: number[]
      styleImageValues: number[]
      mean: [number, number, number]
      std: [number, number, number]
      styleLayerIndices: number[]
      contentLayerIndex: number
      expectedStyleLossByLayer: Record<string, number>
      expectedStyleLossTotal: number
      expectedContentLoss: number
      expectedTotalLoss: number
    }>('/vgg19-phase3-full-pass/vgg19_phase3_full_pass_fixture.json')

    const worker = new Worker(new URL('/src/styleTransfer.worker.ts', window.location.origin), { type: 'module' })
    const ask = (payload: WorkerRequest): Promise<WorkerResponse> => new Promise((resolve) => {
      const handler = (event: MessageEvent<WorkerResponse>): void => {
        if (event.data.id === payload.id) { worker.removeEventListener('message', handler); resolve(event.data) }
      }
      worker.addEventListener('message', handler)
      worker.postMessage(payload)
    })

    const getValues = (response: WorkerResponse): number[] => {
      if (response.type !== 'tensor-op-result' || !response.ok || !('values' in response)) throw new Error('Expected tensor values response')
      return response.values
    }
    const getScalar = (response: WorkerResponse): number => {
      if (response.type !== 'tensor-op-result' || !response.ok || !('scalar' in response)) throw new Error('Expected scalar response')
      return response.scalar
    }

    const init = await ask({ type: 'init-webgpu', id: 'phase3-full-init' })
    const normalizedInput = await ask({ type: 'tensor-op', id: 'phase3-full-norm-input', op: 'normalize-forward', input: { shape: fixture.inputShape, values: fixture.inputImageValues }, mean: fixture.mean, std: fixture.std })
    const normalizedStyle = await ask({ type: 'tensor-op', id: 'phase3-full-norm-style', op: 'normalize-forward', input: { shape: fixture.inputShape, values: fixture.styleImageValues }, mean: fixture.mean, std: fixture.std })

    let currentInputShape: [number, number, number, number] = fixture.inputShape
    let currentStyleShape: [number, number, number, number] = fixture.inputShape
    let currentInputValues: number[] = getValues(normalizedInput)
    let currentStyleValues: number[] = getValues(normalizedStyle)
    const styleLossByLayer: Record<string, number> = {}
    let contentLoss = 0

    for (let layerIndex = 0; layerIndex <= 28; layerIndex += 1) {
      const weightShape = weights[`conv${layerIndex}.weightShape`]
      const weightValues = weights[`conv${layerIndex}.weightValues`]
      const biasValues = weights[`conv${layerIndex}.biasValues`]

      if (Array.isArray(weightValues) && Array.isArray(biasValues) && Array.isArray(weightShape)) {
        const convInput = await ask({ type: 'tensor-op', id: `conv-i-${layerIndex}`, op: 'conv2d-forward', input: { shape: currentInputShape, values: currentInputValues }, weight: { shape: weightShape as [number, number, number, number], values: weightValues }, bias: biasValues })
        const convStyle = await ask({ type: 'tensor-op', id: `conv-s-${layerIndex}`, op: 'conv2d-forward', input: { shape: currentStyleShape, values: currentStyleValues }, weight: { shape: weightShape as [number, number, number, number], values: weightValues }, bias: biasValues })
        currentInputValues = getValues(convInput)
        currentStyleValues = getValues(convStyle)
        currentInputShape = [1, weightShape[0], currentInputShape[2], currentInputShape[3]]
        currentStyleShape = [1, weightShape[0], currentStyleShape[2], currentStyleShape[3]]

        if (fixture.styleLayerIndices.includes(layerIndex)) {
          styleLossByLayer[`conv${layerIndex}`] = getScalar(await ask({ type: 'tensor-op', id: `style-${layerIndex}`, op: 'style-loss', input: { shape: currentInputShape, values: currentInputValues }, target: { shape: currentStyleShape, values: currentStyleValues } }))
        }
        if (layerIndex === fixture.contentLayerIndex) {
          contentLoss = getScalar(await ask({ type: 'tensor-op', id: 'content-21', op: 'content-loss', input: { shape: currentInputShape, values: currentInputValues }, target: { shape: currentInputShape, values: currentInputValues } }))
        }
      }
      if ([1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26].includes(layerIndex)) {
        currentInputValues = getValues(await ask({ type: 'tensor-op', id: `relu-i-${layerIndex}`, op: 'relu-forward', input: { shape: currentInputShape, values: currentInputValues } }))
        currentStyleValues = getValues(await ask({ type: 'tensor-op', id: `relu-s-${layerIndex}`, op: 'relu-forward', input: { shape: currentStyleShape, values: currentStyleValues } }))
      }
      if ([4, 9, 18, 27].includes(layerIndex)) {
        currentInputValues = getValues(await ask({ type: 'tensor-op', id: `pool-i-${layerIndex}`, op: 'maxpool2d-forward', input: { shape: currentInputShape, values: currentInputValues } }))
        currentStyleValues = getValues(await ask({ type: 'tensor-op', id: `pool-s-${layerIndex}`, op: 'maxpool2d-forward', input: { shape: currentStyleShape, values: currentStyleValues } }))
        currentInputShape = [1, currentInputShape[1], Math.floor(currentInputShape[2] / 2), Math.floor(currentInputShape[3] / 2)]
        currentStyleShape = [1, currentStyleShape[1], Math.floor(currentStyleShape[2] / 2), Math.floor(currentStyleShape[3] / 2)]
      }
    }

    worker.terminate()
    const styleTotal = Object.values(styleLossByLayer).reduce((sum, value) => sum + value, 0)
    return { init, styleLossByLayer, styleTotal, contentLoss, totalLoss: styleTotal + contentLoss, expected: fixture }
  })

  expect(result.init.type).toBe('webgpu-init-result')
  if (result.init.type !== 'webgpu-init-result') throw new Error('Expected init result')
  expect(result.init.ok).toBeTruthy()
  for (const [layerName, expectedLoss] of Object.entries(result.expected.expectedStyleLossByLayer)) expect(result.styleLossByLayer[layerName]).toBeCloseTo(expectedLoss, 4)
  expect(result.styleTotal).toBeCloseTo(result.expected.expectedStyleLossTotal, 4)
  expect(result.contentLoss).toBeCloseTo(result.expected.expectedContentLoss, 6)
  expect(result.totalLoss).toBeCloseTo(result.expected.expectedTotalLoss, 4)
})
