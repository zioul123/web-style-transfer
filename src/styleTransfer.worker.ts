/// <reference lib="webworker" />

import type { WorkerRequest, WorkerResponse, WorkerTensorOperand } from './types'
import { createTensor } from './ml'

let gpuDevice: GPUDevice | null = null

// Numeric bitflags are used because `GPUBufferUsage`/`GPUMapMode` ambient globals are not consistently
// available in TS worker builds. Flag values reference:
// https://developer.mozilla.org/en-US/docs/Web/API/GPUBuffer/usage
const BUFFER_USAGE_STORAGE_COPY_DST: number = 0x0080 | 0x0008
const BUFFER_USAGE_STORAGE_COPY_SRC: number = 0x0080 | 0x0004
const BUFFER_USAGE_MAP_READ_COPY_DST: number = 0x0001 | 0x0008
const MAP_MODE_READ: number = 0x0001

const BINARY_SHADER_SNIPPETS: Record<'add' | 'sub' | 'mul' | 'div', { tensorTensor: string; tensorScalar: string; scalarTensor: string }> = {
  add: { tensorTensor: 'out[i] = a[i] + b[i];', tensorScalar: 'out[i] = a[i] + b[0];', scalarTensor: 'out[i] = b[0] + a[i];' },
  sub: { tensorTensor: 'out[i] = a[i] - b[i];', tensorScalar: 'out[i] = a[i] - b[0];', scalarTensor: 'out[i] = b[0] - a[i];' },
  mul: { tensorTensor: 'out[i] = a[i] * b[i];', tensorScalar: 'out[i] = a[i] * b[0];', scalarTensor: 'out[i] = b[0] * a[i];' },
  div: { tensorTensor: 'out[i] = a[i] / b[i];', tensorScalar: 'out[i] = a[i] / b[0];', scalarTensor: 'out[i] = b[0] / a[i];' },
}

const makeBinaryOpShader = (op: 'add' | 'sub' | 'mul' | 'div', count: number, mode: 'tensorTensor' | 'tensorScalar' | 'scalarTensor'): string => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  ${BINARY_SHADER_SNIPPETS[op][mode]}
}`


const makeClampShader = (count: number, min: number, max: number): string => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = clamp(a[i], ${min}, ${max});
}`

const makeMseDiffShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let d = a[i] - b[i];
  out[i] = d * d;
}`

const makeReduceSumShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
var<workgroup> cache: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let i = gid.x;
  let localIndex = lid.x;
  if (i < ${count}u) {
    cache[localIndex] = inputValues[i];
  } else {
    cache[localIndex] = 0.0;
  }
  workgroupBarrier();

  var step: u32 = 32u;
  loop {
    if (localIndex < step) {
      cache[localIndex] = cache[localIndex] + cache[localIndex + step];
    }
    workgroupBarrier();
    if (step == 1u) {
      break;
    }
    step = step / 2u;
  }

  if (localIndex == 0u) {
    out[wid.x] = cache[0];
  }
}`

const postResponse = (response: WorkerResponse): void => {
  self.postMessage(response)
}

const initWebGpu = async (id: string): Promise<void> => {
  if (!('gpu' in navigator)) {
    postResponse({ type: 'webgpu-init-result', id, ok: false, message: 'navigator.gpu is unavailable in this worker context.' })
    return
  }
  const adapter: GPUAdapter | null = await navigator.gpu.requestAdapter()
  if (adapter === null) {
    postResponse({ type: 'webgpu-init-result', id, ok: false, message: 'WebGPU adapter request returned null.' })
    return
  }
  gpuDevice = await adapter.requestDevice()
  postResponse({ type: 'webgpu-init-result', id, ok: true, message: `WebGPU device initialized: ${gpuDevice.label || 'unnamed-device'}` })
}

const runBinaryOp = async (op: 'add' | 'sub' | 'mul' | 'div', a: Float32Array, b: Float32Array, mode: 'tensorTensor' | 'tensorScalar' | 'scalarTensor'): Promise<Float32Array> => {
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const device: GPUDevice = gpuDevice
  const count: number = a.length
  const code: string = makeBinaryOpShader(op, count, mode)
  const bytes: number = count * 4
  const aBuffer: GPUBuffer = device.createBuffer({ size: bytes, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  const bBuffer: GPUBuffer = device.createBuffer({ size: bytes, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  const outBuffer: GPUBuffer = device.createBuffer({ size: bytes, usage: BUFFER_USAGE_STORAGE_COPY_SRC })
  const readBuffer: GPUBuffer = device.createBuffer({ size: bytes, usage: BUFFER_USAGE_MAP_READ_COPY_DST })
  device.queue.writeBuffer(aBuffer, 0, a)
  device.queue.writeBuffer(bBuffer, 0, b)
  const pipeline: GPUComputePipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } })
  const bindGroup: GPUBindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuffer } },
      { binding: 1, resource: { buffer: bBuffer } },
      { binding: 2, resource: { buffer: outBuffer } },
    ],
  })
  const encoder: GPUCommandEncoder = device.createCommandEncoder()
  const pass: GPUComputePassEncoder = encoder.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(Math.ceil(count / 64))
  pass.end()
  encoder.copyBufferToBuffer(outBuffer, 0, readBuffer, 0, bytes)
  device.queue.submit([encoder.finish()])
  await readBuffer.mapAsync(MAP_MODE_READ)
  const arr: Float32Array = new Float32Array(readBuffer.getMappedRange().slice(0))
  readBuffer.unmap()
  return arr
}

const runScalarBinaryOp = async (op: 'add' | 'sub' | 'mul' | 'div', tensor: Float32Array, scalar: number, scalarOnLeft: boolean): Promise<Float32Array> => {
  const scalarTensor: Float32Array = new Float32Array([scalar])
  if (scalarOnLeft) {
    return runBinaryOp(op, tensor, scalarTensor, 'scalarTensor')
  }
  return runBinaryOp(op, tensor, scalarTensor, 'tensorScalar')
}

const getTensorFromOperand = (operand: WorkerTensorOperand): { ok: true; values: Float32Array } | { ok: false } => {
  if (operand.kind !== 'tensor') return { ok: false }
  const tensor = createTensor(operand.tensor.shape, operand.tensor.values)
  return { ok: true, values: tensor.values }
}

const getValuesFromOperand = (operand: WorkerTensorOperand): Float32Array => {
  if (operand.kind === 'scalar') return new Float32Array([operand.scalar])
  return createTensor(operand.tensor.shape, operand.tensor.values).values
}

const runClamp = async (a: Float32Array, min: number, max: number): Promise<Float32Array> => {
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const device: GPUDevice = gpuDevice
  const count: number = a.length
  const code: string = makeClampShader(count, min, max)
  const bytes: number = count * 4
  const aBuffer: GPUBuffer = device.createBuffer({ size: bytes, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  const outBuffer: GPUBuffer = device.createBuffer({ size: bytes, usage: BUFFER_USAGE_STORAGE_COPY_SRC })
  const readBuffer: GPUBuffer = device.createBuffer({ size: bytes, usage: BUFFER_USAGE_MAP_READ_COPY_DST })
  device.queue.writeBuffer(aBuffer, 0, a)
  const pipeline: GPUComputePipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } })
  const bindGroup: GPUBindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: aBuffer } }, { binding: 1, resource: { buffer: outBuffer } }] })
  const encoder: GPUCommandEncoder = device.createCommandEncoder()
  const pass: GPUComputePassEncoder = encoder.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(Math.ceil(count / 64))
  pass.end()
  encoder.copyBufferToBuffer(outBuffer, 0, readBuffer, 0, bytes)
  device.queue.submit([encoder.finish()])
  await readBuffer.mapAsync(MAP_MODE_READ)
  const arr: Float32Array = new Float32Array(readBuffer.getMappedRange().slice(0))
  readBuffer.unmap()
  return arr
}

const runMse = async (a: Float32Array, b: Float32Array): Promise<number> => {
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const device: GPUDevice = gpuDevice
  const count: number = a.length
  const bytes: number = count * 4

  const aBuffer: GPUBuffer = device.createBuffer({ size: bytes, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  const bBuffer: GPUBuffer = device.createBuffer({ size: bytes, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  let currentBuffer: GPUBuffer = device.createBuffer({ size: bytes, usage: BUFFER_USAGE_STORAGE_COPY_SRC })

  device.queue.writeBuffer(aBuffer, 0, a)
  device.queue.writeBuffer(bBuffer, 0, b)

  const diffPipeline: GPUComputePipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: makeMseDiffShader(count) }), entryPoint: 'main' } })
  const diffBindGroup: GPUBindGroup = device.createBindGroup({
    layout: diffPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuffer } },
      { binding: 1, resource: { buffer: bBuffer } },
      { binding: 2, resource: { buffer: currentBuffer } },
    ],
  })

  {
    const encoder: GPUCommandEncoder = device.createCommandEncoder()
    const pass: GPUComputePassEncoder = encoder.beginComputePass()
    pass.setPipeline(diffPipeline)
    pass.setBindGroup(0, diffBindGroup)
    pass.dispatchWorkgroups(Math.ceil(count / 64))
    pass.end()
    device.queue.submit([encoder.finish()])
  }

  let currentCount: number = count
  // Repeatedly reduce the squared-error buffer by summing each 64-value chunk into one output value.
  // After each pass, `currentBuffer` holds a smaller partial-sum vector and `currentCount` shrinks to
  // `ceil(previousCount / 64)`. When `currentCount` reaches 1, the single value is the total SSE.
  while (currentCount > 1) {
    const nextCount: number = Math.ceil(currentCount / 64)
    const nextBytes: number = nextCount * 4
    const nextBuffer: GPUBuffer = device.createBuffer({ size: nextBytes, usage: BUFFER_USAGE_STORAGE_COPY_SRC })
    const reducePipeline: GPUComputePipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: makeReduceSumShader(currentCount) }), entryPoint: 'main' } })
    const reduceBindGroup: GPUBindGroup = device.createBindGroup({
      layout: reducePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: currentBuffer } },
        { binding: 1, resource: { buffer: nextBuffer } },
      ],
    })
    const encoder: GPUCommandEncoder = device.createCommandEncoder()
    const pass: GPUComputePassEncoder = encoder.beginComputePass()
    pass.setPipeline(reducePipeline)
    pass.setBindGroup(0, reduceBindGroup)
    pass.dispatchWorkgroups(nextCount)
    pass.end()
    device.queue.submit([encoder.finish()])
    currentBuffer = nextBuffer
    currentCount = nextCount
  }

  const readBuffer: GPUBuffer = device.createBuffer({ size: 4, usage: BUFFER_USAGE_MAP_READ_COPY_DST })
  const encoder: GPUCommandEncoder = device.createCommandEncoder()
  encoder.copyBufferToBuffer(currentBuffer, 0, readBuffer, 0, 4)
  device.queue.submit([encoder.finish()])
  await readBuffer.mapAsync(MAP_MODE_READ)
  const sumSquaredError: number = new Float32Array(readBuffer.getMappedRange().slice(0))[0]
  readBuffer.unmap()
  return sumSquaredError / count
}

self.onmessage = (event: MessageEvent<WorkerRequest>): void => {
  const payload: WorkerRequest = event.data
  switch (payload.type) {
    case 'ping':
      postResponse({ type: 'pong', id: payload.id, timestamp: Date.now() })
      break
    case 'init-webgpu':
      void initWebGpu(payload.id).catch((error: unknown) => postResponse({ type: 'error', id: payload.id, message: error instanceof Error ? error.message : 'Unknown worker error' }))
      break
    case 'tensor-roundtrip': {
      try {
        const tensor = createTensor(payload.tensor.shape, payload.tensor.values)
        postResponse({ type: 'tensor-roundtrip-result', id: payload.id, ok: true, tensor: { shape: tensor.shape, values: Array.from(tensor.values) } })
      } catch (error: unknown) {
        postResponse({ type: 'tensor-roundtrip-result', id: payload.id, ok: false, message: error instanceof Error ? error.message : 'Tensor roundtrip failed.' })
      }
      break
    }
    case 'tensor-op': {
      void (async (): Promise<void> => {
        try {
          if (payload.op === 'mse') {
            const aOperand = getTensorFromOperand(payload.a)
            if (!aOperand.ok) throw new Error('MSE requires both operands to be tensors.')
            const bOperand = getTensorFromOperand(payload.b)
            if (!bOperand.ok) throw new Error('MSE requires both operands to be tensors.')
            const mse = await runMse(aOperand.values, bOperand.values)
            postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, scalar: mse })
            return
          }
          if (payload.op === 'clamp') {
            const v = await runClamp(getValuesFromOperand(payload.a), payload.clampMin, payload.clampMax)
            postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, values: Array.from(v) })
            return
          }
          if (payload.a.kind === 'scalar' && payload.b.kind === 'scalar') {
            throw new Error('At least one operand must be a tensor for binary tensor ops.')
          }
          let v: Float32Array
          if (payload.a.kind === 'tensor' && payload.b.kind === 'tensor') {
            v = await runBinaryOp(
              payload.op,
              createTensor(payload.a.tensor.shape, payload.a.tensor.values).values,
              createTensor(payload.b.tensor.shape, payload.b.tensor.values).values,
              'tensorTensor',
            )
          } else if (payload.a.kind === 'tensor' && payload.b.kind === 'scalar') {
            v = await runScalarBinaryOp(payload.op, createTensor(payload.a.tensor.shape, payload.a.tensor.values).values, payload.b.scalar, false)
          } else if (payload.a.kind === 'scalar' && payload.b.kind === 'tensor') {
            v = await runScalarBinaryOp(payload.op, createTensor(payload.b.tensor.shape, payload.b.tensor.values).values, payload.a.scalar, true)
          } else {
            throw new Error('At least one operand must be a tensor for binary tensor ops.')
          }
          postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, values: Array.from(v) })
        } catch (error: unknown) {
          postResponse({ type: 'tensor-op-result', id: payload.id, ok: false, message: error instanceof Error ? error.message : 'Tensor op failed.' })
        }
      })()
      break
    }
    default: {
      const exhaustivenessCheck: never = payload
      postResponse({ type: 'error', id: 'unknown', message: `Received unsupported message type: ${String(exhaustivenessCheck)}` })
    }
  }
}
