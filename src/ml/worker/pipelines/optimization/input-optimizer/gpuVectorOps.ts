import {
  ownedBuffer,
  releaseOwnedBuffer,
  type GpuBufferRef,
  type OwnedGpuBuffer,
} from "../../../runtime/bufferKernels";
import {
  BUFFER_USAGE_MAP_READ_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_SRC,
  MAP_MODE_READ,
} from "../../../runtime/gpuFlags";
import { runBinaryOpToBuffer } from "../../../runtime/shaderRunner";
import type {
  InputOptimizerConfig,
  InputOptimizerVectorOps,
} from "./types";

const makeCloneShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = input[i];
}`;

const makeScaleShader = (count: number, scalar: number): string => `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = input[i] * ${scalar};
}`;

const makeAddScaledShader = (count: number, scalar: number): string => `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> direction: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = input[i] + direction[i] * ${scalar};
}`;

const makeUpdateClampShader = (count: number, learningRate: number): string => `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> direction: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = clamp(input[i] - ${learningRate} * direction[i], 0.0, 1.0);
}`;

const makeAdamUpdateClampShader = (
  count: number,
  learningRate: number,
  beta1: number,
  beta2: number,
  oneMinusBeta1: number,
  oneMinusBeta2: number,
  beta1Correction: number,
  beta2Correction: number,
  epsilon: number,
): string => `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let g = grad[i];
  let nextM = ${beta1} * m[i] + ${oneMinusBeta1} * g;
  let nextV = ${beta2} * v[i] + ${oneMinusBeta2} * g * g;
  m[i] = nextM;
  v[i] = nextV;
  let mHat = nextM / ${beta1Correction};
  let vHat = nextV / ${beta2Correction};
  out[i] = clamp(input[i] - ${learningRate} * (mHat / (sqrt(vHat) + ${epsilon})), 0.0, 1.0);
}`;

const makeDotProductShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = a[i] * b[i];
}`;

const makeReduceSumShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let groupStart = gid.x * 64u;
  var sum = 0.0;
  for (var offset = 0u; offset < 64u; offset = offset + 1u) {
    let i = groupStart + offset;
    if (i < ${count}u) {
      sum = sum + input[i];
    }
  }
  out[gid.x] = sum;
}`;

const runTwoInputShader = (
  device: GPUDevice,
  shader: string,
  a: GpuBufferRef,
  b: GpuBufferRef,
  count: number,
): OwnedGpuBuffer => {
  const outBuffer = device.createBuffer({
    size: count * Float32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_STORAGE_COPY_SRC,
  });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: shader }), entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: a.buffer } },
      { binding: 1, resource: { buffer: b.buffer } },
      { binding: 2, resource: { buffer: outBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
  return ownedBuffer(outBuffer);
};

const runSingleInputShader = (
  device: GPUDevice,
  shader: string,
  input: GpuBufferRef,
  count: number,
): OwnedGpuBuffer => {
  const outBuffer = device.createBuffer({
    size: count * Float32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_STORAGE_COPY_SRC,
  });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: shader }), entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.buffer } },
      { binding: 1, resource: { buffer: outBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
  return ownedBuffer(outBuffer);
};

const runAdamShader = (
  device: GPUDevice,
  input: GpuBufferRef,
  grad: GpuBufferRef,
  m: GpuBufferRef,
  v: GpuBufferRef,
  count: number,
  step: number,
  config: InputOptimizerConfig,
): OwnedGpuBuffer => {
  const t = step + 1;
  const outBuffer = device.createBuffer({
    size: count * Float32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_STORAGE_COPY_SRC,
  });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: makeAdamUpdateClampShader(
          count,
          config.learningRate,
          config.adamBeta1,
          config.adamBeta2,
          1 - config.adamBeta1,
          1 - config.adamBeta2,
          1 - Math.pow(config.adamBeta1, t),
          1 - Math.pow(config.adamBeta2, t),
          config.adamEpsilon,
        ),
      }),
      entryPoint: "main",
    },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.buffer } },
      { binding: 1, resource: { buffer: grad.buffer } },
      { binding: 2, resource: { buffer: m.buffer } },
      { binding: 3, resource: { buffer: v.buffer } },
      { binding: 4, resource: { buffer: outBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
  return ownedBuffer(outBuffer);
};

const readScalar = async (
  device: GPUDevice,
  sourceBuffer: GPUBuffer,
): Promise<number> => {
  const readBuffer = device.createBuffer({
    size: Float32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_MAP_READ_COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(
    sourceBuffer,
    0,
    readBuffer,
    0,
    Float32Array.BYTES_PER_ELEMENT,
  );
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(MAP_MODE_READ);
  const scalar = new Float32Array(readBuffer.getMappedRange().slice(0))[0];
  readBuffer.unmap();
  readBuffer.destroy();
  return scalar;
};

const dotProduct = async (
  device: GPUDevice,
  a: GpuBufferRef,
  b: GpuBufferRef,
  count: number,
): Promise<number> => {
  let current = runTwoInputShader(device, makeDotProductShader(count), a, b, count);
  let currentCount = count;
  while (currentCount > 1) {
    const nextCount = Math.ceil(currentCount / 64);
    const next = runSingleInputShader(
      device,
      makeReduceSumShader(currentCount),
      current,
      nextCount,
    );
    releaseOwnedBuffer(current);
    current = next;
    currentCount = nextCount;
  }
  const scalar = await readScalar(device, current.buffer);
  releaseOwnedBuffer(current);
  return scalar;
};

export const createGpuVectorOps = (
  device: GPUDevice,
  count: number,
): InputOptimizerVectorOps<GpuBufferRef> => ({
  clone: async (input: GpuBufferRef): Promise<GpuBufferRef> =>
    runSingleInputShader(device, makeCloneShader(count), input, count),

  zeros: async (): Promise<GpuBufferRef> =>
    ownedBuffer(
      device.createBuffer({
        size: count * Float32Array.BYTES_PER_ELEMENT,
        usage: BUFFER_USAGE_STORAGE_COPY_SRC,
      }),
    ),

  sub: async (a: GpuBufferRef, b: GpuBufferRef): Promise<GpuBufferRef> =>
    ownedBuffer(
      runBinaryOpToBuffer(
        device,
        "sub",
        a.buffer,
        b.buffer,
        count,
        "tensorTensor",
      ),
    ),

  scale: async (
    input: GpuBufferRef,
    scalar: number,
  ): Promise<GpuBufferRef> =>
    runSingleInputShader(device, makeScaleShader(count, scalar), input, count),

  addScaled: async (
    input: GpuBufferRef,
    direction: GpuBufferRef,
    scalar: number,
  ): Promise<GpuBufferRef> =>
    runTwoInputShader(
      device,
      makeAddScaledShader(count, scalar),
      input,
      direction,
      count,
    ),

  dot: async (a: GpuBufferRef, b: GpuBufferRef): Promise<number> =>
    dotProduct(device, a, b, count),

  updateClamp: async (
    input: GpuBufferRef,
    direction: GpuBufferRef,
    learningRate: number,
  ): Promise<GpuBufferRef> =>
    runTwoInputShader(
      device,
      makeUpdateClampShader(count, learningRate),
      input,
      direction,
      count,
    ),

  adamUpdateClamp: async (
    input: GpuBufferRef,
    grad: GpuBufferRef,
    m: GpuBufferRef,
    v: GpuBufferRef,
    step: number,
    config: InputOptimizerConfig,
  ): Promise<GpuBufferRef> =>
    runAdamShader(device, input, grad, m, v, count, step, config),

  dispose: (input: GpuBufferRef): void => {
    releaseOwnedBuffer(input);
  },
});
