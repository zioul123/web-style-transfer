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
const BUFFER_USAGE_UNIFORM_COPY_DST: number = 0x0040 | 0x0008
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

const makeReluShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = max(0.0, inputValues[i]);
}`

const makeNormalizeShader = (count: number): string => `
struct NormalizeUniforms {
  channels: u32,
  height: u32,
  width: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> meanValues: array<f32>;
@group(0) @binding(2) var<storage, read> stdValues: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: NormalizeUniforms;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let spatial = uniforms.height * uniforms.width;
  let c = (i / spatial) % uniforms.channels;
  out[i] = (inputValues[i] - meanValues[c]) / stdValues[c];
}`

const makeMaxPool2dShader = (count: number): string => `
struct MaxPoolUniforms {
  channels: u32,
  inHeight: u32,
  inWidth: u32,
  outHeight: u32,
  outWidth: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: MaxPoolUniforms;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let channelPlane = uniforms.outHeight * uniforms.outWidth;
  let c = (i / channelPlane) % uniforms.channels;
  let outOffset = i % channelPlane;
  let oh = outOffset / uniforms.outWidth;
  let ow = outOffset % uniforms.outWidth;
  let ih = oh * 2u;
  let iw = ow * 2u;
  let inputPlane = uniforms.inHeight * uniforms.inWidth;
  let base = c * inputPlane;
  let idx00 = base + ih * uniforms.inWidth + iw;
  let idx01 = idx00 + 1u;
  let idx10 = idx00 + uniforms.inWidth;
  let idx11 = idx10 + 1u;
  let m0 = max(inputValues[idx00], inputValues[idx01]);
  let m1 = max(inputValues[idx10], inputValues[idx11]);
  out[i] = max(m0, m1);
}`

const makeConv2dShader = (count: number): string => `
struct Conv2dUniforms {
  inChannels: u32,
  outChannels: u32,
  height: u32,
  width: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> weightValues: array<f32>;
@group(0) @binding(2) var<storage, read> biasValues: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Conv2dUniforms;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let hw = uniforms.height * uniforms.width;
  let oc = (i / hw) % uniforms.outChannels;
  let outOffset = i % hw;
  let oh = outOffset / uniforms.width;
  let ow = outOffset % uniforms.width;
  var sum = biasValues[oc];
  for (var ic: u32 = 0u; ic < uniforms.inChannels; ic = ic + 1u) {
    let inputBase = ic * hw;
    let weightBase = (oc * uniforms.inChannels + ic) * 9u;
    for (var kh: u32 = 0u; kh < 3u; kh = kh + 1u) {
      let khSigned = i32(kh) - 1;
      let ihSigned = i32(oh) + khSigned;
      if (ihSigned < 0 || ihSigned >= i32(uniforms.height)) { continue; }
      for (var kw: u32 = 0u; kw < 3u; kw = kw + 1u) {
        let kwSigned = i32(kw) - 1;
        let iwSigned = i32(ow) + kwSigned;
        if (iwSigned < 0 || iwSigned >= i32(uniforms.width)) { continue; }
        let ih = u32(ihSigned);
        let iw = u32(iwSigned);
        let inputIndex = inputBase + ih * uniforms.width + iw;
        let weightIndex = weightBase + kh * 3u + kw;
        sum = sum + inputValues[inputIndex] * weightValues[weightIndex];
      }
    }
  }
  out[i] = sum;
}`

const isBinaryTensorOpPayload = (
  payload: WorkerRequest,
): payload is Extract<WorkerRequest, { type: 'tensor-op'; op: 'add' | 'sub' | 'mul' | 'div' | 'mse'; a: WorkerTensorOperand; b: WorkerTensorOperand }> =>
  payload.type === 'tensor-op'
  && (payload.op === 'add' || payload.op === 'sub' || payload.op === 'mul' || payload.op === 'div' || payload.op === 'mse')

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
  return runBinaryOp(op, tensor, scalarTensor, scalarOnLeft ? 'scalarTensor' : 'tensorScalar')
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

const runUnaryShader = async (code: string, input: Float32Array, outCount: number, extraEntries: GPUBindGroupEntry[] = []): Promise<Float32Array> => {
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const device: GPUDevice = gpuDevice
  const inputBuffer: GPUBuffer = device.createBuffer({ size: input.byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  const outBuffer: GPUBuffer = device.createBuffer({ size: outCount * 4, usage: BUFFER_USAGE_STORAGE_COPY_SRC })
  const readBuffer: GPUBuffer = device.createBuffer({ size: outCount * 4, usage: BUFFER_USAGE_MAP_READ_COPY_DST })
  device.queue.writeBuffer(inputBuffer, 0, input)
  const pipeline: GPUComputePipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } })
  const bindGroup: GPUBindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: inputBuffer } }, ...extraEntries, { binding: extraEntries.length + 1, resource: { buffer: outBuffer } }],
  })
  const encoder: GPUCommandEncoder = device.createCommandEncoder()
  const pass: GPUComputePassEncoder = encoder.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(Math.ceil(outCount / 64))
  pass.end()
  encoder.copyBufferToBuffer(outBuffer, 0, readBuffer, 0, outCount * 4)
  device.queue.submit([encoder.finish()])
  await readBuffer.mapAsync(MAP_MODE_READ)
  const out: Float32Array = new Float32Array(readBuffer.getMappedRange().slice(0))
  readBuffer.unmap()
  return out
}

const runReluForward = async (input: Float32Array): Promise<Float32Array> => runUnaryShader(makeReluShader(input.length), input, input.length)

const runMaxPool2dForward = async (input: Float32Array, shape: readonly [number, number, number, number]): Promise<Float32Array> => {
  if (shape[0] !== 1) throw new Error('maxpool2d-forward currently expects batch size 1.')
  const channels: number = shape[1]
  const inHeight: number = shape[2]
  const inWidth: number = shape[3]
  const outHeight: number = Math.floor(inHeight / 2)
  const outWidth: number = Math.floor(inWidth / 2)
  const outCount: number = channels * outHeight * outWidth
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 5 * 4, usage: BUFFER_USAGE_UNIFORM_COPY_DST })
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([channels, inHeight, inWidth, outHeight, outWidth]))
  return runUnaryShader(makeMaxPool2dShader(outCount), input, outCount, [{ binding: 1, resource: { buffer: uniformBuffer } }])
}

const runNormalizeForward = async (input: Float32Array, shape: readonly [number, number, number, number], mean: readonly number[], std: readonly number[]): Promise<Float32Array> => {
  const channels: number = shape[1]
  if (mean.length !== channels || std.length !== channels) throw new Error('Mean/std lengths must match input channels.')
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const meanBuffer: GPUBuffer = gpuDevice.createBuffer({ size: channels * 4, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  const stdBuffer: GPUBuffer = gpuDevice.createBuffer({ size: channels * 4, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 3 * 4, usage: BUFFER_USAGE_UNIFORM_COPY_DST })
  gpuDevice.queue.writeBuffer(meanBuffer, 0, new Float32Array(mean))
  gpuDevice.queue.writeBuffer(stdBuffer, 0, new Float32Array(std))
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([channels, shape[2], shape[3]]))
  return runUnaryShader(makeNormalizeShader(input.length), input, input.length, [
    { binding: 1, resource: { buffer: meanBuffer } },
    { binding: 2, resource: { buffer: stdBuffer } },
    { binding: 3, resource: { buffer: uniformBuffer } },
  ])
}

const runConv2dForward = async (input: Float32Array, inputShape: readonly [number, number, number, number], weight: Float32Array, weightShape: readonly [number, number, number, number], bias: readonly number[]): Promise<Float32Array> => {
  const [batch, inChannels, height, width] = inputShape
  const [outChannels, weightInChannels, kernelHeight, kernelWidth] = weightShape
  if (batch !== 1 || kernelHeight !== 3 || kernelWidth !== 3 || inChannels !== weightInChannels || bias.length !== outChannels) throw new Error('conv2d-forward only supports VGG-compatible params.')
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const outCount: number = outChannels * height * width
  const weightBuffer: GPUBuffer = gpuDevice.createBuffer({ size: weight.byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  const biasBuffer: GPUBuffer = gpuDevice.createBuffer({ size: outChannels * 4, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 4 * 4, usage: BUFFER_USAGE_UNIFORM_COPY_DST })
  gpuDevice.queue.writeBuffer(weightBuffer, 0, weight)
  gpuDevice.queue.writeBuffer(biasBuffer, 0, new Float32Array(bias))
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([inChannels, outChannels, height, width]))
  return runUnaryShader(makeConv2dShader(outCount), input, outCount, [
    { binding: 1, resource: { buffer: weightBuffer } },
    { binding: 2, resource: { buffer: biasBuffer } },
    { binding: 3, resource: { buffer: uniformBuffer } },
  ])
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

          if (payload.op === 'conv2d-forward') {
            const input = createTensor(payload.input.shape, payload.input.values)
            const weight = createTensor(payload.weight.shape, payload.weight.values)
            const output = await runConv2dForward(input.values, input.shape, weight.values, weight.shape, payload.bias)
            postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, values: Array.from(output) })
            return
          }
          if (payload.op === 'relu-forward') {
            const input = createTensor(payload.input.shape, payload.input.values)
            const output = await runReluForward(input.values)
            postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, values: Array.from(output) })
            return
          }
          if (payload.op === 'maxpool2d-forward') {
            const input = createTensor(payload.input.shape, payload.input.values)
            const output = await runMaxPool2dForward(input.values, input.shape)
            postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, values: Array.from(output) })
            return
          }
          if (payload.op === 'normalize-forward') {
            const input = createTensor(payload.input.shape, payload.input.values)
            const output = await runNormalizeForward(input.values, input.shape, payload.mean, payload.std)
            postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, values: Array.from(output) })
            return
          }
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
          if (!isBinaryTensorOpPayload(payload)) {
            throw new Error(`Unsupported tensor op: ${payload.op}`)
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
