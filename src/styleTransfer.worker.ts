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


const makeReluBackwardShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> gradOutValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = select(0.0, gradOutValues[i], inputValues[i] > 0.0);
}`

const makeMaxPool2dBackwardShader = (count: number): string => `
struct MaxPoolBackwardUniforms {
  channels: u32,
  inHeight: u32,
  inWidth: u32,
  outHeight: u32,
  outWidth: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> gradOutValues: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: MaxPoolBackwardUniforms;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let hw = uniforms.inHeight * uniforms.inWidth;
  let c = (i / hw) % uniforms.channels;
  let inOffset = i % hw;
  let ih = inOffset / uniforms.inWidth;
  let iw = inOffset % uniforms.inWidth;
  let oh = ih / 2u;
  let ow = iw / 2u;
  if (oh >= uniforms.outHeight || ow >= uniforms.outWidth) { out[i] = 0.0; return; }
  let outIdx = c * uniforms.outHeight * uniforms.outWidth + oh * uniforms.outWidth + ow;
  var maxH = oh * 2u;
  var maxW = ow * 2u;
  var maxVal = inputValues[c * hw + maxH * uniforms.inWidth + maxW];
  for (var kh: u32 = 0u; kh < 2u; kh = kh + 1u) {
    for (var kw: u32 = 0u; kw < 2u; kw = kw + 1u) {
      let candH = oh * 2u + kh;
      let candW = ow * 2u + kw;
      let v = inputValues[c * hw + candH * uniforms.inWidth + candW];
      if (v > maxVal) { maxVal = v; maxH = candH; maxW = candW; }
    }
  }
  out[i] = select(0.0, gradOutValues[outIdx], ih == maxH && iw == maxW);
}`

const makeNormalizeBackwardShader = (count: number): string => `
struct NormalizeBackwardUniforms { channels: u32, height: u32, width: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> gradOutValues: array<f32>;
@group(0) @binding(1) var<storage, read> stdValues: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: NormalizeBackwardUniforms;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let spatial = uniforms.height * uniforms.width;
  let c = (i / spatial) % uniforms.channels;
  out[i] = gradOutValues[i] / stdValues[c];
}`

const makeConv2dBackwardInputShader = (count: number): string => `
struct Conv2dBackwardInputUniforms { inChannels: u32, outChannels: u32, height: u32, width: u32 }
@group(0) @binding(0) var<storage, read> gradOutValues: array<f32>;
@group(0) @binding(1) var<storage, read> weightValues: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: Conv2dBackwardInputUniforms;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let hw = uniforms.height * uniforms.width;
  let ic = (i / hw) % uniforms.inChannels;
  let inOffset = i % hw;
  let ih = inOffset / uniforms.width;
  let iw = inOffset % uniforms.width;
  var sum = 0.0;
  for (var oc: u32 = 0u; oc < uniforms.outChannels; oc = oc + 1u) {
    for (var kh: u32 = 0u; kh < 3u; kh = kh + 1u) {
      for (var kw: u32 = 0u; kw < 3u; kw = kw + 1u) {
        let ohSigned = i32(ih) - i32(kh) + 1;
        let owSigned = i32(iw) - i32(kw) + 1;
        if (ohSigned < 0 || ohSigned >= i32(uniforms.height) || owSigned < 0 || owSigned >= i32(uniforms.width)) { continue; }
        let oh = u32(ohSigned);
        let ow = u32(owSigned);
        let gradOutIndex = oc * hw + oh * uniforms.width + ow;
        let weightIndex = ((oc * uniforms.inChannels + ic) * 9u) + kh * 3u + kw;
        sum = sum + gradOutValues[gradOutIndex] * weightValues[weightIndex];
      }
    }
  }
  out[i] = sum;
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


const makeMseBackwardShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> targetValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = (2.0 * (inputValues[i] - targetValues[i])) / ${count}.0;
}`

const makeGramBackwardShader = (count: number): string => `
struct GramBackwardUniforms {
  channels: u32,
  spatial: u32,
  norm: u32,
  _pad: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> gradOutValues: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: GramBackwardUniforms;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let a = i / uniforms.spatial;
  let p = i % uniforms.spatial;
  var sum: f32 = 0.0;
  for (var b: u32 = 0u; b < uniforms.channels; b = b + 1u) {
    let gradAB = gradOutValues[a * uniforms.channels + b];
    let gradBA = gradOutValues[b * uniforms.channels + a];
    let inputBP = inputValues[b * uniforms.spatial + p];
    sum = sum + ((gradAB + gradBA) * inputBP);
  }
  out[i] = sum / f32(uniforms.norm);
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

const makeGramMatrixShader = (count: number): string => `
struct GramUniforms {
  channels: u32,
  spatial: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: GramUniforms;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let row = i / uniforms.channels;
  let col = i % uniforms.channels;
  var sum = 0.0;
  let rowBase = row * uniforms.spatial;
  let colBase = col * uniforms.spatial;
  for (var s: u32 = 0u; s < uniforms.spatial; s = s + 1u) {
    sum = sum + inputValues[rowBase + s] * inputValues[colBase + s];
  }
  let norm = f32(uniforms.channels * uniforms.spatial);
  out[i] = sum / norm;
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


const runReluBackward = async (input: Float32Array, gradOut: Float32Array): Promise<Float32Array> => {
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const gradOutBuffer: GPUBuffer = gpuDevice.createBuffer({ size: gradOut.byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  gpuDevice.queue.writeBuffer(gradOutBuffer, 0, gradOut)
  return runUnaryShader(makeReluBackwardShader(input.length), input, input.length, [{ binding: 1, resource: { buffer: gradOutBuffer } }])
}

const runMaxPool2dBackward = async (input: Float32Array, shape: readonly [number, number, number, number], gradOut: Float32Array): Promise<Float32Array> => {
  if (shape[0] !== 1) throw new Error('maxpool2d-backward currently expects batch size 1.')
  const channels = shape[1]
  const inHeight = shape[2]
  const inWidth = shape[3]
  const outHeight = Math.floor(inHeight / 2)
  const outWidth = Math.floor(inWidth / 2)
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const gradOutBuffer: GPUBuffer = gpuDevice.createBuffer({ size: gradOut.byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 8 * 4, usage: BUFFER_USAGE_UNIFORM_COPY_DST })
  gpuDevice.queue.writeBuffer(gradOutBuffer, 0, gradOut)
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([channels, inHeight, inWidth, outHeight, outWidth, 0, 0, 0]))
  return runUnaryShader(makeMaxPool2dBackwardShader(input.length), input, input.length, [{ binding: 1, resource: { buffer: gradOutBuffer } }, { binding: 2, resource: { buffer: uniformBuffer } }])
}

const runNormalizeBackward = async (gradOut: Float32Array, shape: readonly [number, number, number, number], std: readonly number[]): Promise<Float32Array> => {
  const channels = shape[1]
  if (std.length !== channels) throw new Error('Std length must match input channels.')
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const stdBuffer: GPUBuffer = gpuDevice.createBuffer({ size: channels * 4, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 4 * 4, usage: BUFFER_USAGE_UNIFORM_COPY_DST })
  gpuDevice.queue.writeBuffer(stdBuffer, 0, new Float32Array(std))
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([channels, shape[2], shape[3], 0]))
  return runUnaryShader(makeNormalizeBackwardShader(gradOut.length), gradOut, gradOut.length, [{ binding: 1, resource: { buffer: stdBuffer } }, { binding: 2, resource: { buffer: uniformBuffer } }])
}

const runConv2dBackwardInput = async (inputShape: readonly [number, number, number, number], gradOut: Float32Array, weight: Float32Array, weightShape: readonly [number, number, number, number]): Promise<Float32Array> => {
  const [batch, inChannels, height, width] = inputShape
  const [outChannels, weightInChannels, kernelHeight, kernelWidth] = weightShape
  if (batch !== 1 || kernelHeight !== 3 || kernelWidth !== 3 || inChannels !== weightInChannels) throw new Error('conv2d-backward-input only supports VGG-compatible params.')
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const outCount = inChannels * height * width
  const gradOutBuffer: GPUBuffer = gpuDevice.createBuffer({ size: gradOut.byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  const weightBuffer: GPUBuffer = gpuDevice.createBuffer({ size: weight.byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 4 * 4, usage: BUFFER_USAGE_UNIFORM_COPY_DST })
  gpuDevice.queue.writeBuffer(gradOutBuffer, 0, gradOut)
  gpuDevice.queue.writeBuffer(weightBuffer, 0, weight)
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([inChannels, outChannels, height, width]))
  return runUnaryShader(makeConv2dBackwardInputShader(outCount), gradOut, outCount, [{ binding: 1, resource: { buffer: weightBuffer } }, { binding: 2, resource: { buffer: uniformBuffer } }])
}
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
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 4 * 4, usage: BUFFER_USAGE_UNIFORM_COPY_DST })
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

const runGramMatrix = async (input: Float32Array, shape: readonly [number, number, number, number]): Promise<Float32Array> => {
  if (shape[0] !== 1) throw new Error('gram-matrix expects batch size 1.')
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const channels: number = shape[1]
  const spatial: number = shape[2] * shape[3]
  const outCount: number = channels * channels
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 2 * 4, usage: BUFFER_USAGE_UNIFORM_COPY_DST })
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([channels, spatial]))
  return runUnaryShader(makeGramMatrixShader(outCount), input, outCount, [{ binding: 1, resource: { buffer: uniformBuffer } }])
}


const runContentLossBackward = async (input: Float32Array, target: Float32Array): Promise<Float32Array> => {
  if (input.length !== target.length) return new Float32Array(input.length)
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const targetBuffer: GPUBuffer = gpuDevice.createBuffer({ size: target.byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  gpuDevice.queue.writeBuffer(targetBuffer, 0, target)
  return runUnaryShader(makeMseBackwardShader(input.length), input, input.length, [{ binding: 1, resource: { buffer: targetBuffer } }])
}

const runGramBackward = async (input: Float32Array, shape: readonly [number, number, number, number], gradOut: Float32Array): Promise<Float32Array> => {
  const channels: number = shape[1]
  const spatial: number = shape[2] * shape[3]
  const norm: number = shape[0] * shape[1] * shape[2] * shape[3]
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const gradOutBuffer: GPUBuffer = gpuDevice.createBuffer({ size: gradOut.byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 4 * 4, usage: BUFFER_USAGE_UNIFORM_COPY_DST })
  gpuDevice.queue.writeBuffer(gradOutBuffer, 0, gradOut)
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([channels, spatial, norm, 0]))
  return runUnaryShader(makeGramBackwardShader(input.length), input, input.length, [{ binding: 1, resource: { buffer: gradOutBuffer } }, { binding: 2, resource: { buffer: uniformBuffer } }])
}

const runStyleLossBackward = async (input: Float32Array, shape: readonly [number, number, number, number], target: Float32Array): Promise<Float32Array> => {
  const inputGram = await runGramMatrix(input, shape)
  const targetGram = await runGramMatrix(target, shape)
  const gramGrad: Float32Array = await runContentLossBackward(inputGram, targetGram)
  return await runGramBackward(input, shape, gramGrad)
}
const runFirstPoolOptimizer = async (payload: Extract<WorkerRequest, { type: 'run-first-pool-optimizer' }>): Promise<{ losses: number[]; finalValues: number[] }> => {
  const convForward = async (inputValues: Float32Array, inputShape: readonly [number, number, number, number], weightValues: Float32Array, weightShape: readonly [number, number, number, number], bias: number[]): Promise<Float32Array> =>
    await runConv2dForward(inputValues, inputShape, weightValues, weightShape, bias)

  const conv1Weight = createTensor(payload.conv1Weight.shape, payload.conv1Weight.values)
  const conv2Weight = createTensor(payload.conv2Weight.shape, payload.conv2Weight.values)
  const conv3Weight = createTensor(payload.conv3Weight.shape, payload.conv3Weight.values)

  const contentNorm = await runNormalizeForward(new Float32Array(payload.contentImageValues), payload.inputShape, payload.mean, payload.std)
  const styleNorm = await runNormalizeForward(new Float32Array(payload.styleImageValues), payload.inputShape, payload.mean, payload.std)

  const styleConv1 = await convForward(styleNorm, payload.inputShape, conv1Weight.values, conv1Weight.shape, payload.conv1Bias)
  const styleRelu1 = await runReluForward(styleConv1)
  const styleConv2 = await convForward(styleRelu1, [1, 64, 16, 16], conv2Weight.values, conv2Weight.shape, payload.conv2Bias)
  const styleRelu2 = await runReluForward(styleConv2)
  const stylePool = await runMaxPool2dForward(styleRelu2, [1, 64, 16, 16])
  const styleConv3 = await convForward(stylePool, [1, 64, 8, 8], conv3Weight.values, conv3Weight.shape, payload.conv3Bias)
  const styleRelu3 = await runReluForward(styleConv3)

  const contentConv1 = await convForward(contentNorm, payload.inputShape, conv1Weight.values, conv1Weight.shape, payload.conv1Bias)
  const contentRelu1 = await runReluForward(contentConv1)
  const contentConv2 = await convForward(contentRelu1, [1, 64, 16, 16], conv2Weight.values, conv2Weight.shape, payload.conv2Bias)
  const contentRelu2 = await runReluForward(contentConv2)

  let inputValues = new Float32Array(payload.initialInputValues)
  const losses: number[] = []
  for (let step = 0; step < payload.steps; step += 1) {
    const norm = await runNormalizeForward(inputValues, payload.inputShape, payload.mean, payload.std)
    const conv1 = await convForward(norm, payload.inputShape, conv1Weight.values, conv1Weight.shape, payload.conv1Bias)
    const relu1 = await runReluForward(conv1)
    const conv2 = await convForward(relu1, [1, 64, 16, 16], conv2Weight.values, conv2Weight.shape, payload.conv2Bias)
    const relu2 = await runReluForward(conv2)
    const pool = await runMaxPool2dForward(relu2, [1, 64, 16, 16])
    const conv3 = await convForward(pool, [1, 64, 8, 8], conv3Weight.values, conv3Weight.shape, payload.conv3Bias)
    const relu3 = await runReluForward(conv3)

    const styleLoss1 = (await runMse(await runGramMatrix(relu1, [1, 64, 16, 16]), await runGramMatrix(styleRelu1, [1, 64, 16, 16]))) * payload.styleWeightConv1
    const styleLoss3 = (await runMse(await runGramMatrix(relu3, [1, 128, 8, 8]), await runGramMatrix(styleRelu3, [1, 128, 8, 8]))) * payload.styleWeightConv3
    const contentLoss = (await runMse(relu2, contentRelu2)) * payload.contentWeight
    losses.push(styleLoss1 + styleLoss3 + contentLoss)

    const gStyle1Relu = await runScalarBinaryOp('mul', await runStyleLossBackward(relu1, [1, 64, 16, 16], styleRelu1), payload.styleWeightConv1, false)
    const gStyle3Relu = await runScalarBinaryOp('mul', await runStyleLossBackward(relu3, [1, 128, 8, 8], styleRelu3), payload.styleWeightConv3, false)
    const gContentRelu = await runScalarBinaryOp('mul', await runContentLossBackward(relu2, contentRelu2), payload.contentWeight, false)
    const gStyle3 = await runReluBackward(conv3, gStyle3Relu)
    const gContent = await runReluBackward(conv2, gContentRelu)
    const gStyle1 = await runReluBackward(conv1, gStyle1Relu)

    const gPool = await runConv2dBackwardInput([1, 64, 8, 8], gStyle3, conv3Weight.values, conv3Weight.shape)
    const gRelu2 = await runMaxPool2dBackward(relu2, [1, 64, 16, 16], gPool)
    const gConv2FromPool = await runReluBackward(conv2, gRelu2)
    const gConv2Total = await runBinaryOp('add', gConv2FromPool, gContent, 'tensorTensor')
    const gRelu1 = await runConv2dBackwardInput([1, 64, 16, 16], gConv2Total, conv2Weight.values, conv2Weight.shape)
    const gConv1FromConv2 = await runReluBackward(conv1, gRelu1)
    const gConv1Total = await runBinaryOp('add', gConv1FromConv2, gStyle1, 'tensorTensor')
    const gNorm = await runConv2dBackwardInput(payload.inputShape, gConv1Total, conv1Weight.values, conv1Weight.shape)
    const gInput = await runNormalizeBackward(gNorm, payload.inputShape, payload.std)

    const next = new Float32Array(inputValues.length)
    for (let i = 0; i < inputValues.length; i += 1) next[i] = Math.max(0, Math.min(1, inputValues[i] - payload.learningRate * gInput[i]))
    inputValues = next
  }
  return { losses, finalValues: Array.from(inputValues) }
}


const runStyleTransfer = async (payload: Extract<WorkerRequest, { type: 'run-style-transfer' }>): Promise<{ losses: number[]; finalValues: number[]; stats: { elapsedMs: number; avgStepMs: number; forwardMs: number; backwardMs: number; lossMs: number; updateMs: number; clampMs: number; steps: number } }> => {
  const reluLayers: readonly number[] = [1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29]
  const poolLayers: readonly number[] = [4, 9, 18, 27]
  const reluLayerSet = new Set(reluLayers)
  const poolLayerSet = new Set(poolLayers)
  const convWeightCache: Record<number, { shape: [number, number, number, number]; values: Float32Array } | undefined> = {}
  for (let layerIndex = 0; layerIndex <= 29; layerIndex += 1) {
    const weightShapeRaw = payload.weights[`conv${layerIndex}.weightShape`]
    const weightValues = payload.weights[`conv${layerIndex}.weightValues`]
    if (Array.isArray(weightShapeRaw) && Array.isArray(weightValues)) convWeightCache[layerIndex] = { shape: weightShapeRaw as [number, number, number, number], values: new Float32Array(weightValues) }
  }
  const contentNorm = await runNormalizeForward(new Float32Array(payload.contentImageValues), payload.inputShape, payload.mean, payload.std)
  const styleNorm = await runNormalizeForward(new Float32Array(payload.styleImageValues), payload.inputShape, payload.mean, payload.std)
  const runForward = async (start: Float32Array): Promise<{ reluOut: Record<number, Float32Array>; reluShape: Record<number, readonly [number, number, number, number]>; convInShape: Record<number, readonly [number, number, number, number]>; reluPre: Record<number, Float32Array>; layerInput: Record<number, Float32Array>; layerInputShape: Record<number, readonly [number, number, number, number]>; final: Float32Array }> => {
    let shape: readonly [number, number, number, number] = payload.inputShape
    let values = start
    const reluOut: Record<number, Float32Array> = {}
    const reluShape: Record<number, readonly [number, number, number, number]> = {}
    const convInShape: Record<number, readonly [number, number, number, number]> = {}
    const reluPre: Record<number, Float32Array> = {}
    const layerInput: Record<number, Float32Array> = {}
    const layerInputShape: Record<number, readonly [number, number, number, number]> = {}
    for (let layerIndex = 0; layerIndex <= 29; layerIndex += 1) {
      layerInput[layerIndex] = values
      layerInputShape[layerIndex] = shape
      const convWeight = convWeightCache[layerIndex]
      const biasValues = payload.weights[`conv${layerIndex}.biasValues`]
      if (convWeight !== undefined && Array.isArray(biasValues)) {
        convInShape[layerIndex] = shape
        values = await runConv2dForward(values, shape, convWeight.values, convWeight.shape, biasValues)
        shape = [1, convWeight.shape[0], shape[2], shape[3]]
      }
      if (reluLayerSet.has(layerIndex)) {
        reluPre[layerIndex] = values
        values = await runReluForward(values)
        reluOut[layerIndex] = values
        reluShape[layerIndex] = shape
      }
      if (poolLayerSet.has(layerIndex)) {
        values = await runMaxPool2dForward(values, shape)
        shape = [1, shape[1], Math.floor(shape[2] / 2), Math.floor(shape[3] / 2)]
      }
    }
    return { reluOut, reluShape, convInShape, reluPre, layerInput, layerInputShape, final: values }
  }

  const contentTargets = await runForward(contentNorm)
  const styleTargets = await runForward(styleNorm)
  let inputValues: Float32Array<ArrayBufferLike> = new Float32Array(payload.inputImageValues)
  const losses: number[] = []
  const optimizer = payload.optimizer ?? 'sgd'
  const adamBeta1 = payload.adamBeta1 ?? 0.9
  const adamBeta2 = payload.adamBeta2 ?? 0.999
  const adamEps = payload.adamEpsilon ?? 1e-8
  const lbfgsMemory = payload.lbfgsMemory ?? 10
  const lbfgsEps = payload.lbfgsEpsilon ?? 1e-8
  const adamM = new Float32Array(inputValues.length)
  const adamV = new Float32Array(inputValues.length)
  const lbfgsHistory: { s: Float32Array; y: Float32Array; rho: number }[] = []
  let previousInput: Float32Array | null = null
  let previousGrad: Float32Array | null = null

  let forwardMs = 0
  let backwardMs = 0
  let lossMs = 0
  let updateMs = 0
  let clampMs = 0
  const totalStart = performance.now()

  const updateInput = async (grad: Float32Array, step: number): Promise<Float32Array<ArrayBufferLike>> => {
    const nextValues = new Float32Array(inputValues.length)
    if (optimizer === 'sgd') {
      for (let i = 0; i < inputValues.length; i += 1) nextValues[i] = inputValues[i] - payload.learningRate * grad[i]
    } else if (optimizer === 'adam') {
      const t = step + 1
      const b1Correction = 1 - Math.pow(adamBeta1, t)
      const b2Correction = 1 - Math.pow(adamBeta2, t)
      for (let i = 0; i < inputValues.length; i += 1) {
        const g = grad[i]
        adamM[i] = adamBeta1 * adamM[i] + (1 - adamBeta1) * g
        adamV[i] = adamBeta2 * adamV[i] + (1 - adamBeta2) * g * g
        const mHat = adamM[i] / b1Correction
        const vHat = adamV[i] / b2Correction
        nextValues[i] = inputValues[i] - payload.learningRate * (mHat / (Math.sqrt(vHat) + adamEps))
      }
    } else {
      if (previousInput !== null && previousGrad !== null) {
        const s = new Float32Array(inputValues.length)
        const y = new Float32Array(inputValues.length)
        let sy = 0
        for (let i = 0; i < inputValues.length; i += 1) {
          s[i] = inputValues[i] - previousInput[i]
          y[i] = grad[i] - previousGrad[i]
          sy += s[i] * y[i]
        }
        if (sy > lbfgsEps) {
          lbfgsHistory.push({ s, y, rho: 1 / sy })
          if (lbfgsHistory.length > lbfgsMemory) lbfgsHistory.shift()
        }
      }
      const q = new Float32Array(grad)
      const alphas = new Float32Array(lbfgsHistory.length)
      for (let i = lbfgsHistory.length - 1; i >= 0; i -= 1) {
        const { s, y, rho } = lbfgsHistory[i]
        let sq = 0
        for (let j = 0; j < q.length; j += 1) sq += s[j] * q[j]
        const alpha = rho * sq
        alphas[i] = alpha
        for (let j = 0; j < q.length; j += 1) q[j] -= alpha * y[j]
      }
      let scale = 1
      if (lbfgsHistory.length > 0) {
        const last = lbfgsHistory[lbfgsHistory.length - 1]
        let ys = 0
        let yy = 0
        for (let i = 0; i < q.length; i += 1) {
          ys += last.y[i] * last.s[i]
          yy += last.y[i] * last.y[i]
        }
        if (yy > lbfgsEps) scale = ys / yy
      }
      for (let i = 0; i < q.length; i += 1) q[i] *= scale
      for (let i = 0; i < lbfgsHistory.length; i += 1) {
        const { s, y, rho } = lbfgsHistory[i]
        let yq = 0
        for (let j = 0; j < q.length; j += 1) yq += y[j] * q[j]
        const beta = rho * yq
        for (let j = 0; j < q.length; j += 1) q[j] += s[j] * (alphas[i] - beta)
      }
      for (let i = 0; i < inputValues.length; i += 1) nextValues[i] = inputValues[i] - payload.learningRate * q[i]
    }
    const clampStart = performance.now()
    const clamped = new Float32Array(await runClamp(nextValues, 0, 1))
    clampMs += performance.now() - clampStart
    previousInput = new Float32Array(inputValues)
    previousGrad = new Float32Array(grad)
    return clamped
  }

  for (let step = 0; step < payload.steps; step += 1) {
    const forwardStart = performance.now()
    const norm = await runNormalizeForward(inputValues, payload.inputShape, payload.mean, payload.std)
    const run = await runForward(norm)
    forwardMs += performance.now() - forwardStart
    const lossStart = performance.now()
    let totalStyle = 0
    for (const reluLayerIndex of payload.styleLayerIndices) {
      const inputRelu = run.reluOut[reluLayerIndex]
      const styleRelu = styleTargets.reluOut[reluLayerIndex]
      if (inputRelu === undefined || styleRelu === undefined) throw new Error(`Missing ReLU activation for style layer index ${reluLayerIndex}.`)
      const reluShape = run.reluShape[reluLayerIndex]
      if (reluShape === undefined) throw new Error(`Unsupported ReLU style layer index ${reluLayerIndex}.`)
      totalStyle += await runMse(await runGramMatrix(inputRelu, reluShape), await runGramMatrix(styleRelu, reluShape))
    }
    const contentRelu = run.reluOut[payload.contentLayerIndex]
    const contentTargetRelu = contentTargets.reluOut[payload.contentLayerIndex]
    if (contentRelu === undefined || contentTargetRelu === undefined) throw new Error(`Missing ReLU activation for content layer index ${payload.contentLayerIndex}.`)
    const contentLoss = await runMse(contentRelu, contentTargetRelu)
    losses.push(payload.styleWeight * totalStyle + payload.contentWeight * contentLoss)
    lossMs += performance.now() - lossStart
    const gradByReluLayer: Record<number, Float32Array> = {}
    for (const reluLayerIndex of payload.styleLayerIndices) {
      const inputRelu = run.reluOut[reluLayerIndex]
      const styleRelu = styleTargets.reluOut[reluLayerIndex]
      if (inputRelu === undefined || styleRelu === undefined) throw new Error(`Missing ReLU activation for style gradient layer index ${reluLayerIndex}.`)
      const reluShape = run.reluShape[reluLayerIndex]
      if (reluShape === undefined) throw new Error(`Unsupported ReLU style gradient layer index ${reluLayerIndex}.`)
      const styleGrad = await runStyleLossBackward(inputRelu, reluShape, styleRelu)
      const weightedStyleGrad = payload.styleWeight === 1 ? styleGrad : await runScalarBinaryOp('mul', styleGrad, payload.styleWeight, false)
      gradByReluLayer[reluLayerIndex] = gradByReluLayer[reluLayerIndex] === undefined ? weightedStyleGrad : await runBinaryOp('add', gradByReluLayer[reluLayerIndex], weightedStyleGrad, 'tensorTensor')
    }
    const contentGrad = await runContentLossBackward(contentRelu, contentTargetRelu)
    const weightedContentGrad = payload.contentWeight === 1 ? contentGrad : await runScalarBinaryOp('mul', contentGrad, payload.contentWeight, false)
    gradByReluLayer[payload.contentLayerIndex] = gradByReluLayer[payload.contentLayerIndex] === undefined ? weightedContentGrad : await runBinaryOp('add', gradByReluLayer[payload.contentLayerIndex], weightedContentGrad, 'tensorTensor')

    const backwardStart = performance.now()
    let grad: Float32Array | null = null
    for (let layerIndex = 29; layerIndex >= 0; layerIndex -= 1) {
      if (poolLayerSet.has(layerIndex)) {
        if (grad !== null) grad = await runMaxPool2dBackward(run.layerInput[layerIndex], run.layerInputShape[layerIndex], grad)
      }
      if (reluLayerSet.has(layerIndex)) {
        const tapGrad = gradByReluLayer[layerIndex]
        if (tapGrad !== undefined) grad = grad === null ? tapGrad : await runBinaryOp('add', grad, tapGrad, 'tensorTensor')
        if (grad !== null) grad = await runReluBackward(run.reluPre[layerIndex], grad)
      }
      const convWeight = convWeightCache[layerIndex]
      if (convWeight !== undefined && grad !== null) {
        grad = await runConv2dBackwardInput(run.convInShape[layerIndex], grad, convWeight.values, convWeight.shape)
      }
    }
    if (grad === null) throw new Error('No gradient path reached the input image.')
    const gInput = await runNormalizeBackward(grad, payload.inputShape, payload.std)
    backwardMs += performance.now() - backwardStart

    const updateStart = performance.now()
    inputValues = await updateInput(gInput, step)
    updateMs += performance.now() - updateStart
  }
  const elapsedMs = performance.now() - totalStart
  return { losses, finalValues: Array.from(inputValues), stats: { elapsedMs, avgStepMs: payload.steps === 0 ? 0 : elapsedMs / payload.steps, forwardMs, backwardMs, lossMs, updateMs, clampMs, steps: payload.steps } }
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
    case 'run-first-pool-optimizer': {
      void (async (): Promise<void> => {
        try {
          const result = await runFirstPoolOptimizer(payload)
          postResponse({ type: 'run-first-pool-optimizer-result', id: payload.id, ok: true, losses: result.losses, finalValues: result.finalValues })
        } catch (error: unknown) {
          postResponse({ type: 'run-first-pool-optimizer-result', id: payload.id, ok: false, message: error instanceof Error ? error.message : 'First-pool optimizer failed.' })
        }
      })()
      break
    }
    case 'run-style-transfer': {
      void (async (): Promise<void> => {
        try {
          const result = await runStyleTransfer(payload)
          postResponse({ type: 'run-style-transfer-result', id: payload.id, ok: true, losses: result.losses, finalValues: result.finalValues, stats: result.stats })
        } catch (error: unknown) {
          postResponse({ type: 'run-style-transfer-result', id: payload.id, ok: false, message: error instanceof Error ? error.message : 'Style transfer run failed.' })
        }
      })()
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
          if (payload.op === 'reshape-chw-flatten') {
            const input = createTensor(payload.input.shape, payload.input.values)
            postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, values: Array.from(input.values) })
            return
          }
          if (payload.op === 'gram-matrix') {
            const input = createTensor(payload.input.shape, payload.input.values)
            const output = await runGramMatrix(input.values, input.shape)
            postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, values: Array.from(output) })
            return
          }
          if (payload.op === 'content-loss') {
            const input = createTensor(payload.input.shape, payload.input.values)
            const target = createTensor(payload.target.shape, payload.target.values)
            const sameShape: boolean = input.shape.every((value, index) => value === target.shape[index])
            if (!sameShape) {
              postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, scalar: 0 })
              return
            }
            const mse = await runMse(input.values, target.values)
            const contentWeight: number = payload.contentWeight ?? 1
            postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, scalar: mse * contentWeight })
            return
          }
          if (payload.op === 'relu-backward') {
            const input = createTensor(payload.input.shape, payload.input.values)
            const gradOut = createTensor(payload.gradOut.shape, payload.gradOut.values)
            const gradIn = await runReluBackward(input.values, gradOut.values)
            postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, values: Array.from(gradIn) })
            return
          }
          if (payload.op === 'maxpool2d-backward') {
            const input = createTensor(payload.input.shape, payload.input.values)
            const gradOut = createTensor(payload.gradOut.shape, payload.gradOut.values)
            const gradIn = await runMaxPool2dBackward(input.values, input.shape, gradOut.values)
            postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, values: Array.from(gradIn) })
            return
          }
          if (payload.op === 'normalize-backward') {
            const gradOut = createTensor(payload.gradOut.shape, payload.gradOut.values)
            const gradIn = await runNormalizeBackward(gradOut.values, gradOut.shape, payload.std)
            postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, values: Array.from(gradIn) })
            return
          }
          if (payload.op === 'conv2d-backward-input') {
            const gradOut = createTensor(payload.gradOut.shape, payload.gradOut.values)
            const weight = createTensor(payload.weight.shape, payload.weight.values)
            const gradIn = await runConv2dBackwardInput(payload.inputShape, gradOut.values, weight.values, weight.shape)
            postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, values: Array.from(gradIn) })
            return
          }
          if (payload.op === 'content-loss-backward') {
            const input = createTensor(payload.input.shape, payload.input.values)
            const target = createTensor(payload.target.shape, payload.target.values)
            const gradIn = await runContentLossBackward(input.values, target.values)
            const contentWeight: number = payload.contentWeight ?? 1
            const weightedGradIn = contentWeight === 1 ? gradIn : await runScalarBinaryOp('mul', gradIn, contentWeight, false)
            postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, values: Array.from(weightedGradIn) })
            return
          }
          if (payload.op === 'gram-backward') {
            const input = createTensor(payload.input.shape, payload.input.values)
            const gradOut = createTensor(payload.gradOut.shape, payload.gradOut.values)
            const gradIn = await runGramBackward(input.values, input.shape, gradOut.values)
            postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, values: Array.from(gradIn) })
            return
          }
          if (payload.op === 'style-loss-backward') {
            const input = createTensor(payload.input.shape, payload.input.values)
            const target = createTensor(payload.target.shape, payload.target.values)
            const gradIn = await runStyleLossBackward(input.values, input.shape, target.values)
            const styleWeight: number = payload.styleWeight ?? 1
            const weightedGradIn = styleWeight === 1 ? gradIn : await runScalarBinaryOp('mul', gradIn, styleWeight, false)
            postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, values: Array.from(weightedGradIn) })
            return
          }
          if (payload.op === 'style-loss') {
            const input = createTensor(payload.input.shape, payload.input.values)
            const target = createTensor(payload.target.shape, payload.target.values)
            const inputGram = await runGramMatrix(input.values, input.shape)
            const targetGram = await runGramMatrix(target.values, target.shape)
            const mse = await runMse(inputGram, targetGram)
            const styleWeight: number = payload.styleWeight ?? 1
            postResponse({ type: 'tensor-op-result', id: payload.id, ok: true, scalar: mse * styleWeight })
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
