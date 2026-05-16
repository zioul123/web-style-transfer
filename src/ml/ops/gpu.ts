import { Tensor } from '../core/tensor';
import { cpuOps } from './cpu';

interface BinaryUniform {
  length: number;
}

interface ClampUniform {
  length: number;
  min: number;
  max: number;
}

interface MseUniform {
  length: number;
}

const WORKGROUP_SIZE = 64;

class WebGPUOpsEngine {
  private devicePromise: Promise<GPUDevice> | null = null;
  private binaryPipelines = new Map<string, GPUComputePipeline>();
  private clampPipeline: GPUComputePipeline | null = null;
  private mseDiffPipeline: GPUComputePipeline | null = null;
  private mseReducePipeline: GPUComputePipeline | null = null;

  private async getDevice(): Promise<GPUDevice> {
    if (!this.devicePromise) {
      this.devicePromise = (async () => {
        const nav = globalThis.navigator as Navigator & { gpu?: GPU };
        if (!nav?.gpu) throw new Error('WebGPU unavailable');
        const adapter = await nav.gpu.requestAdapter();
        if (!adapter) throw new Error('No WebGPU adapter found');
        return adapter.requestDevice();
      })();
    }
    return this.devicePromise;
  }

  private createBuffer(device: GPUDevice, data: Float32Array, usage: GPUBufferUsageFlags): GPUBuffer {
    const buffer = device.createBuffer({
      size: Math.max(4, data.byteLength),
      usage,
      mappedAtCreation: true
    });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
  }

  private async readFloatArray(device: GPUDevice, buffer: GPUBuffer, length: number): Promise<Float32Array> {
    const readBuffer = device.createBuffer({
      size: Math.max(4, length * Float32Array.BYTES_PER_ELEMENT),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, Math.max(4, length * 4));
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await readBuffer.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    readBuffer.destroy();
    return copy;
  }

  private getBinaryPipeline(device: GPUDevice, op: 'add' | 'sub' | 'mul' | 'div'): GPUComputePipeline {
    const existing = this.binaryPipelines.get(op);
    if (existing) return existing;

    const operator = op === 'add' ? '+' : op === 'sub' ? '-' : op === 'mul' ? '*' : '/';
    const module = device.createShaderModule({
      code: `
struct Params { length: u32; };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.length) { return; }
  out[idx] = a[idx] ${operator} b[idx];
}`
    });

    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    this.binaryPipelines.set(op, pipeline);
    return pipeline;
  }

  private getClampPipeline(device: GPUDevice): GPUComputePipeline {
    if (this.clampPipeline) return this.clampPipeline;
    const module = device.createShaderModule({
      code: `
struct Params { length: u32; minValue: f32; maxValue: f32; };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.length) { return; }
  out[idx] = clamp(a[idx], params.minValue, params.maxValue);
}`
    });
    this.clampPipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    return this.clampPipeline;
  }

  private getMseDiffPipeline(device: GPUDevice): GPUComputePipeline {
    if (this.mseDiffPipeline) return this.mseDiffPipeline;
    const module = device.createShaderModule({
      code: `
struct Params { length: u32; };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.length) { return; }
  let d = a[idx] - b[idx];
  out[idx] = d * d;
}`
    });
    this.mseDiffPipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    return this.mseDiffPipeline;
  }

  private getMseReducePipeline(device: GPUDevice): GPUComputePipeline {
    if (this.mseReducePipeline) return this.mseReducePipeline;
    const module = device.createShaderModule({
      code: `
struct Params { length: u32; };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> partial: array<f32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let idx = gid.x;
  let localIdx = lid.x;
  if (idx < params.length) {
    partial[localIdx] = src[idx];
  } else {
    partial[localIdx] = 0.0;
  }
  workgroupBarrier();

  var stride = ${WORKGROUP_SIZE / 2}u;
  loop {
    if (localIdx < stride) {
      partial[localIdx] += partial[localIdx + stride];
    }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride = stride / 2u;
  }

  if (localIdx == 0u) {
    dst[wid.x] = partial[0];
  }
}`
    });
    this.mseReducePipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    return this.mseReducePipeline;
  }

  async binaryOp(op: 'add' | 'sub' | 'mul' | 'div', a: Tensor, b: Tensor): Promise<Tensor> {
    if (a.data.length !== b.data.length) throw new Error('Shape mismatch');
    const device = await this.getDevice();
    const pipeline = this.getBinaryPipeline(device, op);
    const length = a.data.length;

    const aBuffer = this.createBuffer(device, a.data, GPUBufferUsage.STORAGE);
    const bBuffer = this.createBuffer(device, b.data, GPUBufferUsage.STORAGE);
    const outBuffer = device.createBuffer({ size: Math.max(4, length * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
    new Uint32Array(uniform.getMappedRange())[0] = ( { length } as BinaryUniform).length;
    uniform.unmap();

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuffer } },
        { binding: 1, resource: { buffer: bBuffer } },
        { binding: 2, resource: { buffer: outBuffer } },
        { binding: 3, resource: { buffer: uniform } }
      ]
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(length / WORKGROUP_SIZE));
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    const out = await this.readFloatArray(device, outBuffer, length);
    aBuffer.destroy(); bBuffer.destroy(); outBuffer.destroy(); uniform.destroy();
    return new Tensor(out, a.shape);
  }

  async clamp(a: Tensor, min = 0, max = 1): Promise<Tensor> {
    const device = await this.getDevice();
    const pipeline = this.getClampPipeline(device);
    const length = a.data.length;
    const inBuffer = this.createBuffer(device, a.data, GPUBufferUsage.STORAGE);
    const outBuffer = device.createBuffer({ size: Math.max(4, length * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
    const uniformData = new ArrayBuffer(16);
    new Uint32Array(uniformData)[0] = ({ length, min, max } as ClampUniform).length;
    new Float32Array(uniformData)[1] = min;
    new Float32Array(uniformData)[2] = max;
    new Uint8Array(uniform.getMappedRange()).set(new Uint8Array(uniformData));
    uniform.unmap();

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inBuffer } },
        { binding: 1, resource: { buffer: outBuffer } },
        { binding: 2, resource: { buffer: uniform } }
      ]
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(length / WORKGROUP_SIZE));
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    const out = await this.readFloatArray(device, outBuffer, length);
    inBuffer.destroy(); outBuffer.destroy(); uniform.destroy();
    return new Tensor(out, a.shape);
  }

  async mse(a: Tensor, b: Tensor): Promise<number> {
    if (a.data.length !== b.data.length) throw new Error('Shape mismatch');
    const device = await this.getDevice();
    const length = a.data.length;

    let srcBuffer = device.createBuffer({ size: Math.max(4, length * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const aBuffer = this.createBuffer(device, a.data, GPUBufferUsage.STORAGE);
    const bBuffer = this.createBuffer(device, b.data, GPUBufferUsage.STORAGE);
    const diffUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
    new Uint32Array(diffUniform.getMappedRange())[0] = ({ length } as MseUniform).length;
    diffUniform.unmap();

    const diffPipeline = this.getMseDiffPipeline(device);
    const diffBind = device.createBindGroup({
      layout: diffPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuffer } },
        { binding: 1, resource: { buffer: bBuffer } },
        { binding: 2, resource: { buffer: srcBuffer } },
        { binding: 3, resource: { buffer: diffUniform } }
      ]
    });

    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(diffPipeline);
      pass.setBindGroup(0, diffBind);
      pass.dispatchWorkgroups(Math.ceil(length / WORKGROUP_SIZE));
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    let currentLength = length;
    const reducePipeline = this.getMseReducePipeline(device);
    while (currentLength > 1) {
      const nextLength = Math.ceil(currentLength / WORKGROUP_SIZE);
      const dstBuffer = device.createBuffer({ size: Math.max(4, nextLength * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const reduceUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
      new Uint32Array(reduceUniform.getMappedRange())[0] = currentLength;
      reduceUniform.unmap();

      const reduceBind = device.createBindGroup({
        layout: reducePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: srcBuffer } },
          { binding: 1, resource: { buffer: dstBuffer } },
          { binding: 2, resource: { buffer: reduceUniform } }
        ]
      });

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(reducePipeline);
      pass.setBindGroup(0, reduceBind);
      pass.dispatchWorkgroups(nextLength);
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();

      srcBuffer.destroy();
      reduceUniform.destroy();
      srcBuffer = dstBuffer;
      currentLength = nextLength;
    }

    const result = await this.readFloatArray(device, srcBuffer, 1);
    aBuffer.destroy(); bBuffer.destroy(); srcBuffer.destroy(); diffUniform.destroy();
    return result[0] / length;
  }
}

const engine = new WebGPUOpsEngine();

export type GPUBackend = 'webgpu' | 'cpu-fallback';

export interface GPUExecutionMeta {
  backend: GPUBackend;
  operation: string;
  fallbackReason?: string;
}

let lastExecutionMeta: GPUExecutionMeta | null = null;

async function withFallback<T>(operation: string, run: () => Promise<T>, fallback: () => T): Promise<T> {
  try {
    const result = await run();
    lastExecutionMeta = { backend: 'webgpu', operation };
    return result;
  } catch (error) {
    lastExecutionMeta = { backend: 'cpu-fallback', operation, fallbackReason: String(error) };
    return fallback();
  }
}

export function getLastGPUExecutionMeta(): GPUExecutionMeta | null {
  return lastExecutionMeta;
}

export const gpuOps = {
  add: async (a: Tensor, b: Tensor) => withFallback('add', () => engine.binaryOp('add', a, b), () => cpuOps.add(a, b)),
  sub: async (a: Tensor, b: Tensor) => withFallback('sub', () => engine.binaryOp('sub', a, b), () => cpuOps.sub(a, b)),
  mul: async (a: Tensor, b: Tensor) => withFallback('mul', () => engine.binaryOp('mul', a, b), () => cpuOps.mul(a, b)),
  div: async (a: Tensor, b: Tensor) => withFallback('div', () => engine.binaryOp('div', a, b), () => cpuOps.div(a, b)),
  clamp: async (a: Tensor, min = 0, max = 1) => withFallback('clamp', () => engine.clamp(a, min, max), () => cpuOps.clamp(a, min, max)),
  mse: async (a: Tensor, b: Tensor) => withFallback('mse', () => engine.mse(a, b), () => cpuOps.mse(a, b))
};
