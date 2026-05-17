/// <reference lib="webworker" />

import type { WorkerRequest, WorkerResponse } from './types'
import { createTensor, cpuMse } from './ml'

let gpuDevice: GPUDevice | null = null
const BUFFER_USAGE_STORAGE_COPY_DST: number = 0x0080 | 0x0008
const BUFFER_USAGE_STORAGE_COPY_SRC: number = 0x0080 | 0x0004
const BUFFER_USAGE_MAP_READ_COPY_DST: number = 0x0001 | 0x0008
const MAP_MODE_READ: number = 0x0001

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

const runBinaryOp = async (op: 'add' | 'sub' | 'mul' | 'div', a: Float32Array, b: Float32Array): Promise<Float32Array> => {
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const device: GPUDevice = gpuDevice
  const count = a.length
  const code = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  ${op === 'add' ? 'out[i] = a[i] + b[i];' : ''}
  ${op === 'sub' ? 'out[i] = a[i] - b[i];' : ''}
  ${op === 'mul' ? 'out[i] = a[i] * b[i];' : ''}
  ${op === 'div' ? 'out[i] = a[i] / b[i];' : ''}
}`
  const bytes: number = count * 4
  const aBuffer = device.createBuffer({ size: bytes, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  const bBuffer = device.createBuffer({ size: bytes, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  const outBuffer = device.createBuffer({ size: bytes, usage: BUFFER_USAGE_STORAGE_COPY_SRC })
  const readBuffer = device.createBuffer({ size: bytes, usage: BUFFER_USAGE_MAP_READ_COPY_DST })
  device.queue.writeBuffer(aBuffer, 0, a)
  device.queue.writeBuffer(bBuffer, 0, b)
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } })
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuffer } },
      { binding: 1, resource: { buffer: bBuffer } },
      { binding: 2, resource: { buffer: outBuffer } },
    ],
  })
  const encoder = device.createCommandEncoder()
  const pass = encoder.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(Math.ceil(count / 64))
  pass.end()
  encoder.copyBufferToBuffer(outBuffer, 0, readBuffer, 0, bytes)
  device.queue.submit([encoder.finish()])
  await readBuffer.mapAsync(MAP_MODE_READ)
  const arr = new Float32Array(readBuffer.getMappedRange().slice(0))
  readBuffer.unmap()
  return arr
}

const runClamp = async (a: Float32Array, min: number, max: number): Promise<Float32Array> => {
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const device: GPUDevice = gpuDevice
  const count = a.length
  const code = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = clamp(a[i], ${min}, ${max});
}`
  const bytes: number = count * 4
  const aBuffer = device.createBuffer({ size: bytes, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  const outBuffer = device.createBuffer({ size: bytes, usage: BUFFER_USAGE_STORAGE_COPY_SRC })
  const readBuffer = device.createBuffer({ size: bytes, usage: BUFFER_USAGE_MAP_READ_COPY_DST })
  device.queue.writeBuffer(aBuffer, 0, a)
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } })
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: aBuffer } }, { binding: 1, resource: { buffer: outBuffer } }] })
  const encoder = device.createCommandEncoder()
  const pass = encoder.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(Math.ceil(count / 64))
  pass.end()
  encoder.copyBufferToBuffer(outBuffer, 0, readBuffer, 0, bytes)
  device.queue.submit([encoder.finish()])
  await readBuffer.mapAsync(MAP_MODE_READ)
  const arr = new Float32Array(readBuffer.getMappedRange().slice(0))
  readBuffer.unmap()
  return arr
}

self.onmessage = (event: MessageEvent<WorkerRequest>): void => {
  const payload = event.data
  switch (payload.type) {
    case 'ping': postResponse({ type: 'pong', id: payload.id, timestamp: Date.now() }); break
    case 'init-webgpu': void initWebGpu(payload.id).catch((error: unknown) => postResponse({ type: 'error', id: payload.id, message: error instanceof Error ? error.message : 'Unknown worker error' })); break
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
          const a = createTensor(payload.a.shape, payload.a.values)
          if (payload.op === 'mse') {
            const b = createTensor(payload.b.shape, payload.b.values)
            postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, scalar: cpuMse(a, b) })
            return
          }
          if (payload.op === 'clamp') {
            const v = await runClamp(a.values, payload.clampMin, payload.clampMax)
            postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, values: Array.from(v) })
            return
          }
          const b = createTensor(payload.b.shape, payload.b.values)
          const v = await runBinaryOp(payload.op, a.values, b.values)
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
