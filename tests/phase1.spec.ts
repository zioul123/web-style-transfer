import { expect, test } from '@playwright/test'
import { cpuAdd, cpuClamp, cpuDiv, cpuMse, cpuMul, cpuSub, createTensor } from '../src/ml'
import {
  isWorkerTensorScalarOpResponse,
  isWorkerTensorVectorOpResponse,
  type WorkerRequest,
  type WorkerResponse,
} from '../src/types'

const shape = [1, 1, 2, 2] as const
const tensorValues = [1, 2, 3, 4]

type Phase1EvaluateResult = {
  init: WorkerResponse
  roundtrip: WorkerResponse
  add: WorkerResponse
  sub: WorkerResponse
  mul: WorkerResponse
  div: WorkerResponse
  mulScalar: WorkerResponse
  scalarSubTensor: WorkerResponse
  clamp: WorkerResponse
  scalarClamp: WorkerResponse
  mse: WorkerResponse
}

const makeWorkerMseTensor = (length: number): { shape: readonly [number, number, number, number]; values: number[] } => ({
  shape: [1, 1, 1, length] as const,
  values: Array.from({ length }, (_unused: unknown, index: number): number => ((index % 11) - 5) / 3),
})

test('phase 1 tensor roundtrip and ops parity', async ({ page }) => {
  await page.goto('/')

  const result: Phase1EvaluateResult = await page.evaluate(async () => {
    const worker = new Worker(new URL('/src/styleTransfer.worker.ts', window.location.origin), { type: 'module' })

    const ask = (payload: WorkerRequest): Promise<WorkerResponse> =>
      new Promise((resolve) => {
        const handler = (event: MessageEvent<WorkerResponse>): void => {
          if (event.data.id === payload.id) {
            worker.removeEventListener('message', handler)
            resolve(event.data)
          }
        }
        worker.addEventListener('message', handler)
        worker.postMessage(payload)
      })

    const workerShape = [1, 1, 2, 2] as const
    const workerTensorValues = [1, 2, 3, 4]
    const tensor = { shape: workerShape, values: workerTensorValues }

    const init = await ask({ type: 'init-webgpu', id: `${Date.now()}-init` })
    const roundtrip = await ask({ type: 'tensor-roundtrip', id: `${Date.now()}-rt`, tensor })
    const add = await ask({ type: 'tensor-op', id: `${Date.now()}-add`, op: 'add', a: { kind: 'tensor', tensor }, b: { kind: 'tensor', tensor: { shape: workerShape, values: [5, 6, 7, 8] } } })
    const sub = await ask({ type: 'tensor-op', id: `${Date.now()}-sub`, op: 'sub', a: { kind: 'tensor', tensor }, b: { kind: 'tensor', tensor: { shape: workerShape, values: [1, 1, 1, 1] } } })
    const mul = await ask({ type: 'tensor-op', id: `${Date.now()}-mul`, op: 'mul', a: { kind: 'tensor', tensor }, b: { kind: 'tensor', tensor: { shape: workerShape, values: [2, 2, 2, 2] } } })
    const div = await ask({ type: 'tensor-op', id: `${Date.now()}-div`, op: 'div', a: { kind: 'tensor', tensor }, b: { kind: 'tensor', tensor: { shape: workerShape, values: [2, 2, 2, 2] } } })

    const mulScalar = await ask({ type: 'tensor-op', id: `${Date.now()}-mul-scalar`, op: 'mul', a: { kind: 'tensor', tensor }, b: { kind: 'scalar', scalar: 2 } })
    const scalarSubTensor = await ask({ type: 'tensor-op', id: `${Date.now()}-scalar-sub-tensor`, op: 'sub', a: { kind: 'scalar', scalar: 10 }, b: { kind: 'tensor', tensor } })
    const clamp = await ask({ type: 'tensor-op', id: `${Date.now()}-clamp`, op: 'clamp', a: { kind: 'tensor', tensor: { shape: workerShape, values: [-1, 0.5, 1.5, 0.25] } }, clampMin: 0, clampMax: 1 })
    const scalarClamp = await ask({ type: 'tensor-op', id: `${Date.now()}-scalar-clamp`, op: 'clamp', a: { kind: 'scalar', scalar: 2.5 }, clampMin: 0, clampMax: 1 })
    const mse = await ask({ type: 'tensor-op', id: `${Date.now()}-mse`, op: 'mse', a: { kind: 'tensor', tensor }, b: { kind: 'tensor', tensor: { shape: workerShape, values: [1, 1, 1, 1] } } })

    worker.terminate()

    return { init, roundtrip, add, sub, mul, div, mulScalar, scalarSubTensor, clamp, scalarClamp, mse }
  })

  expect(result.init.type).toBe('webgpu-init-result')
  if (result.init.type !== 'webgpu-init-result') throw new Error('Expected webgpu-init-result')
  expect(result.init.ok).toBeTruthy()

  expect(result.roundtrip.type).toBe('tensor-roundtrip-result')
  if (result.roundtrip.type !== 'tensor-roundtrip-result') throw new Error('Expected tensor-roundtrip-result')
  expect(result.roundtrip.ok).toBeTruthy()
  if (!result.roundtrip.ok) throw new Error(result.roundtrip.message)
  expect(result.roundtrip.tensor.values).toEqual(tensorValues)


  if (result.add.type !== 'tensor-op-result') throw new Error('add must be tensor-op-result')
  if (result.sub.type !== 'tensor-op-result') throw new Error('sub must be tensor-op-result')
  if (result.mul.type !== 'tensor-op-result') throw new Error('mul must be tensor-op-result')
  if (result.div.type !== 'tensor-op-result') throw new Error('div must be tensor-op-result')
  if (result.mulScalar.type !== 'tensor-op-result') throw new Error('mulScalar must be tensor-op-result')
  if (result.scalarSubTensor.type !== 'tensor-op-result') throw new Error('scalarSubTensor must be tensor-op-result')
  if (result.clamp.type !== 'tensor-op-result') throw new Error('clamp must be tensor-op-result')
  if (result.scalarClamp.type !== 'tensor-op-result') throw new Error('scalarClamp must be tensor-op-result')
  if (result.mse.type !== 'tensor-op-result') throw new Error('mse must be tensor-op-result')

  expect(isWorkerTensorVectorOpResponse(result.add)).toBeTruthy()
  expect(isWorkerTensorVectorOpResponse(result.sub)).toBeTruthy()
  expect(isWorkerTensorVectorOpResponse(result.mul)).toBeTruthy()
  expect(isWorkerTensorVectorOpResponse(result.div)).toBeTruthy()
  expect(isWorkerTensorVectorOpResponse(result.mulScalar)).toBeTruthy()
  expect(isWorkerTensorVectorOpResponse(result.scalarSubTensor)).toBeTruthy()
  expect(isWorkerTensorVectorOpResponse(result.clamp)).toBeTruthy()
  expect(isWorkerTensorVectorOpResponse(result.scalarClamp)).toBeTruthy()
  expect(isWorkerTensorScalarOpResponse(result.mse)).toBeTruthy()

  if (!isWorkerTensorVectorOpResponse(result.add)) throw new Error('add must be vector response')
  if (!isWorkerTensorVectorOpResponse(result.sub)) throw new Error('sub must be vector response')
  if (!isWorkerTensorVectorOpResponse(result.mul)) throw new Error('mul must be vector response')
  if (!isWorkerTensorVectorOpResponse(result.div)) throw new Error('div must be vector response')
  if (!isWorkerTensorVectorOpResponse(result.mulScalar)) throw new Error('mulScalar must be vector response')
  if (!isWorkerTensorVectorOpResponse(result.scalarSubTensor)) throw new Error('scalarSubTensor must be vector response')
  if (!isWorkerTensorVectorOpResponse(result.clamp)) throw new Error('clamp must be vector response')
  if (!isWorkerTensorVectorOpResponse(result.scalarClamp)) throw new Error('scalarClamp must be vector response')
  if (!isWorkerTensorScalarOpResponse(result.mse)) throw new Error('mse must be scalar response')

  expect(result.add.values).toEqual(Array.from(cpuAdd(createTensor(shape, tensorValues), createTensor(shape, [5, 6, 7, 8])).values))
  expect(result.sub.values).toEqual(Array.from(cpuSub(createTensor(shape, tensorValues), createTensor(shape, [1, 1, 1, 1])).values))
  expect(result.mul.values).toEqual(Array.from(cpuMul(createTensor(shape, tensorValues), createTensor(shape, [2, 2, 2, 2])).values))
  expect(result.div.values).toEqual(Array.from(cpuDiv(createTensor(shape, tensorValues), createTensor(shape, [2, 2, 2, 2])).values))
  expect(result.mulScalar.values).toEqual(Array.from(cpuMul(createTensor(shape, tensorValues), createTensor(shape, [2, 2, 2, 2])).values))
  expect(result.scalarSubTensor.values).toEqual([9, 8, 7, 6])
  expect(result.clamp.values).toEqual(Array.from(cpuClamp(createTensor(shape, [-1, 0.5, 1.5, 0.25]), 0, 1).values))
  expect(result.scalarClamp.values).toEqual([1])
  expect(result.mse.scalar).toBeCloseTo(cpuMse(createTensor(shape, tensorValues), createTensor(shape, [1, 1, 1, 1])))
})

test('phase 1 mse GPU reduction parity for >64 lengths', async ({ page }) => {
  await page.goto('/')

  const mseLengths: readonly number[] = [64, 64 * 5, 64 * 5 + 1, 64 * 5 + 32, 64 * 5 + 63]

  const phase1MseResult: { init: WorkerResponse; responses: Array<{ length: number; response: WorkerResponse }> } = await page.evaluate(async (lengths: readonly number[]) => {
    const worker = new Worker(new URL('/src/styleTransfer.worker.ts', window.location.origin), { type: 'module' })
    const ask = (payload: WorkerRequest): Promise<WorkerResponse> =>
      new Promise((resolve) => {
        const handler = (event: MessageEvent<WorkerResponse>): void => {
          if (event.data.id === payload.id) {
            worker.removeEventListener('message', handler)
            resolve(event.data)
          }
        }
        worker.addEventListener('message', handler)
        worker.postMessage(payload)
      })

    const init = await ask({ type: 'init-webgpu', id: `${Date.now()}-mse-init` })
    const responses: Array<{ length: number; response: WorkerResponse }> = []
    for (const length of lengths) {
      const aValues: number[] = Array.from({ length }, (_unused: unknown, index: number): number => ((index % 11) - 5) / 3)
      const bValues: number[] = Array.from({ length }, (_unused: unknown, index: number): number => ((index % 7) - 3) / 2)
      const response = await ask({
        type: 'tensor-op',
        id: `${Date.now()}-mse-${length}`,
        op: 'mse',
        a: { kind: 'tensor', tensor: { shape: [1, 1, 1, length] as const, values: aValues } },
        b: { kind: 'tensor', tensor: { shape: [1, 1, 1, length] as const, values: bValues } },
      })
      responses.push({ length, response })
    }

    worker.terminate()
    return { init, responses }
  }, mseLengths)

  expect(phase1MseResult.init.type).toBe('webgpu-init-result')
  if (phase1MseResult.init.type !== 'webgpu-init-result') throw new Error('Expected webgpu-init-result')
  expect(phase1MseResult.init.ok).toBeTruthy()

  for (const result of phase1MseResult.responses) {
    expect(result.response.type).toBe('tensor-op-result')
    if (result.response.type !== 'tensor-op-result') throw new Error(`Expected tensor-op-result for length ${result.length}.`)
    expect(isWorkerTensorScalarOpResponse(result.response)).toBeTruthy()
    if (!isWorkerTensorScalarOpResponse(result.response)) throw new Error(`Expected scalar MSE response for length ${result.length}.`)

    const aTensor = makeWorkerMseTensor(result.length)
    const bTensor = {
      shape: [1, 1, 1, result.length] as const,
      values: Array.from({ length: result.length }, (_unused: unknown, index: number): number => ((index % 7) - 3) / 2),
    }
    const expectedMse: number = cpuMse(createTensor(aTensor.shape, aTensor.values), createTensor(bTensor.shape, bTensor.values))
    expect(result.response.scalar).toBeCloseTo(expectedMse, 6)
  }
})
