import { expect, test } from '@playwright/test'
import { cpuAdd, cpuClamp, cpuDiv, cpuMse, cpuMul, cpuSub, createTensor } from '../src/ml'

test('phase 1 tensor roundtrip and ops parity', async ({ page }) => {
  await page.goto('/')

  const result = await page.evaluate(async () => {
    const worker = new Worker(new URL('/src/styleTransfer.worker.ts', window.location.origin), {
      type: 'module',
    })

    const ask = (payload: { id: string; [key: string]: unknown }): Promise<unknown> =>
      new Promise((resolve) => {
        const handler = (event: MessageEvent<unknown>): void => {
          const data = event.data as { id?: string }
          if (data.id === payload.id) {
            worker.removeEventListener('message', handler)
            resolve(event.data)
          }
        }
        worker.addEventListener('message', handler)
        worker.postMessage(payload)
      })

    const shape = [1, 1, 2, 2] as const
    const tensor = { shape, values: [1, 2, 3, 4] }

    const init = (await ask({ type: 'init-webgpu', id: `${Date.now()}-init` })) as { ok: boolean; message: string }
    const roundtrip = (await ask({ type: 'tensor-roundtrip', id: `${Date.now()}-rt`, tensor })) as {
      ok: boolean
      tensor?: { values: number[] }
    }
    const add = (await ask({ type: 'tensor-op', id: `${Date.now()}-add`, op: 'add', a: tensor, b: { shape, values: [5, 6, 7, 8] } })) as { ok: boolean; values?: number[] }
    const sub = (await ask({ type: 'tensor-op', id: `${Date.now()}-sub`, op: 'sub', a: tensor, b: { shape, values: [1, 1, 1, 1] } })) as { ok: boolean; values?: number[] }
    const mul = (await ask({ type: 'tensor-op', id: `${Date.now()}-mul`, op: 'mul', a: tensor, b: { shape, values: [2, 2, 2, 2] } })) as { ok: boolean; values?: number[] }
    const div = (await ask({ type: 'tensor-op', id: `${Date.now()}-div`, op: 'div', a: tensor, b: { shape, values: [2, 2, 2, 2] } })) as { ok: boolean; values?: number[] }
    const clamp = (await ask({ type: 'tensor-op', id: `${Date.now()}-clamp`, op: 'clamp', a: { shape, values: [-1, 0.5, 1.5, 0.25] }, clampMin: 0, clampMax: 1 })) as { ok: boolean; values?: number[] }
    const mse = (await ask({ type: 'tensor-op', id: `${Date.now()}-mse`, op: 'mse', a: tensor, b: { shape, values: [1, 1, 1, 1] } })) as { ok: boolean; scalar?: number }

    worker.terminate()

    return { init, roundtrip, add, sub, mul, div, clamp, mse }
  })

  expect(result.init.ok).toBeTruthy()
  expect(result.roundtrip.ok).toBeTruthy()
  expect(result.roundtrip.tensor?.values).toEqual([1, 2, 3, 4])

  expect(result.add.values).toEqual(Array.from(cpuAdd(createTensor([1, 1, 2, 2], [1, 2, 3, 4]), createTensor([1, 1, 2, 2], [5, 6, 7, 8])).values))
  expect(result.sub.values).toEqual(Array.from(cpuSub(createTensor([1, 1, 2, 2], [1, 2, 3, 4]), createTensor([1, 1, 2, 2], [1, 1, 1, 1])).values))
  expect(result.mul.values).toEqual(Array.from(cpuMul(createTensor([1, 1, 2, 2], [1, 2, 3, 4]), createTensor([1, 1, 2, 2], [2, 2, 2, 2])).values))
  expect(result.div.values).toEqual(Array.from(cpuDiv(createTensor([1, 1, 2, 2], [1, 2, 3, 4]), createTensor([1, 1, 2, 2], [2, 2, 2, 2])).values))
  expect(result.clamp.values).toEqual(Array.from(cpuClamp(createTensor([1, 1, 2, 2], [-1, 0.5, 1.5, 0.25]), 0, 1).values))
  expect(result.mse.scalar).toBeCloseTo(cpuMse(createTensor([1, 1, 2, 2], [1, 2, 3, 4]), createTensor([1, 1, 2, 2], [1, 1, 1, 1])))
})
