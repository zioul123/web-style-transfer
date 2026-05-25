import {
  ownedBuffer,
  releaseOwnedBuffer,
  type GpuBufferRef,
  type OwnedGpuBuffer,
} from "../../../runtime/bufferKernels";
import {
  BUFFER_USAGE_MAP_READ_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_SRC,
  BUFFER_USAGE_STORAGE_COPY_DST,
  MAP_MODE_READ,
} from "../../../runtime/gpuFlags";
import { getOrCreateComputePipeline } from "../../../runtime/computePipelineCache";
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

const makeScaleShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> scalar: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = input[i] * scalar[0];
}`;

const makeAddScaledShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> direction: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> scalar: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = input[i] + direction[i] * scalar[0];
}`;

const makeUpdateClampShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> direction: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> learningRate: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = clamp(input[i] - learningRate[0] * direction[i], 0.0, 1.0);
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

const makeReduceSumShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>) {
  let groupStart = workgroupId.x * 64u;
  var sum = 0.0;
  for (var offset = 0u; offset < 64u; offset = offset + 1u) {
    let i = groupStart + offset;
    if (i < ${count}u) {
      sum = sum + input[i];
    }
  }
  out[workgroupId.x] = sum;
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

const makeDotPairProductsShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> leftA: array<f32>;
@group(0) @binding(1) var<storage, read> leftB: array<f32>;
@group(0) @binding(2) var<storage, read> right: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let rv = right[i];
  out[i] = leftA[i] * rv;
  out[${count}u + i] = leftB[i] * rv;
}`;

const makeReducePairSumShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>) {
  let groupStart = workgroupId.x * 64u;
  var sumA = 0.0;
  var sumB = 0.0;
  for (var offset = 0u; offset < 64u; offset = offset + 1u) {
    let i = groupStart + offset;
    if (i < ${count}u) {
      sumA = sumA + input[i];
      sumB = sumB + input[${count}u + i];
    }
  }
  out[workgroupId.x] = sumA;
  out[${Math.ceil(count / 64)}u + workgroupId.x] = sumB;
}`;

const makeAbsShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = abs(input[i]);
}`;

const createPipelineCache = (
  device: GPUDevice,
): ((key: string, shader: string) => GPUComputePipeline) =>
  (key: string, shader: string): GPUComputePipeline =>
    getOrCreateComputePipeline(device, key, shader);

const writeScalarParameter = (
  device: GPUDevice,
  scalarBuffer: GPUBuffer,
  scalar: number,
): void => {
  device.queue.writeBuffer(scalarBuffer, 0, new Float32Array([scalar]));
};

const runTwoInputShader = (
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  a: GpuBufferRef,
  b: GpuBufferRef,
  count: number,
  extraEntries: GPUBindGroupEntry[] = [],
): OwnedGpuBuffer => {
  const outBuffer = device.createBuffer({
    size: count * Float32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_STORAGE_COPY_SRC,
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: a.buffer } },
      { binding: 1, resource: { buffer: b.buffer } },
      { binding: 2, resource: { buffer: outBuffer } },
      ...extraEntries,
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
  pipeline: GPUComputePipeline,
  input: GpuBufferRef,
  count: number,
  extraEntries: GPUBindGroupEntry[] = [],
): OwnedGpuBuffer => {
  const outBuffer = device.createBuffer({
    size: count * Float32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_STORAGE_COPY_SRC,
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.buffer } },
      { binding: 1, resource: { buffer: outBuffer } },
      ...extraEntries,
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

const runSingleInputReductionShader = (
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  input: GpuBufferRef,
  inputCount: number,
  outputCount: number,
): OwnedGpuBuffer => {
  const outBuffer = device.createBuffer({
    size: outputCount * Float32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_STORAGE_COPY_SRC,
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
  pass.dispatchWorkgroups(Math.ceil(inputCount / 64));
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

const runThreeInputShader = (
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  a: GpuBufferRef,
  b: GpuBufferRef,
  c: GpuBufferRef,
  count: number,
  outCount: number,
): OwnedGpuBuffer => {
  const outBuffer = device.createBuffer({
    size: outCount * Float32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_STORAGE_COPY_SRC,
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: a.buffer } },
      { binding: 1, resource: { buffer: b.buffer } },
      { binding: 2, resource: { buffer: c.buffer } },
      { binding: 3, resource: { buffer: outBuffer } },
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
  getPipeline: (key: string, shader: string) => GPUComputePipeline,
  a: GpuBufferRef,
  b: GpuBufferRef,
  count: number,
): Promise<number> => {
  let current = runTwoInputShader(
    device,
    getPipeline(`dot-product:${count}`, makeDotProductShader(count)),
    a,
    b,
    count,
  );
  let currentCount = count;
  while (currentCount > 1) {
    const nextCount = Math.ceil(currentCount / 64);
    const next = runSingleInputReductionShader(
      device,
      getPipeline(`reduce-sum:${currentCount}`, makeReduceSumShader(currentCount)),
      current,
      currentCount,
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

const absSum = async (
  device: GPUDevice,
  getPipeline: (key: string, shader: string) => GPUComputePipeline,
  input: GpuBufferRef,
  count: number,
): Promise<number> => {
  let current = runSingleInputShader(
    device,
    getPipeline(`abs:${count}`, makeAbsShader(count)),
    input,
    count,
  );
  let currentCount = count;
  while (currentCount > 1) {
    const nextCount = Math.ceil(currentCount / 64);
    const next = runSingleInputReductionShader(
      device,
      getPipeline(`reduce-sum:${currentCount}`, makeReduceSumShader(currentCount)),
      current,
      currentCount,
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

const dotPairWithRight = async (
  device: GPUDevice,
  getPipeline: (key: string, shader: string) => GPUComputePipeline,
  leftA: GpuBufferRef,
  leftB: GpuBufferRef,
  right: GpuBufferRef,
  count: number,
): Promise<readonly [number, number]> => {
  let current = runThreeInputShader(
    device,
    getPipeline(`dot-pair-products:${count}`, makeDotPairProductsShader(count)),
    leftA,
    leftB,
    right,
    count,
    count * 2,
  );
  let currentCount = count;
  while (currentCount > 1) {
    const nextCount = Math.ceil(currentCount / 64);
    const next = runSingleInputReductionShader(
      device,
      getPipeline(`reduce-pair-sum:${currentCount}`, makeReducePairSumShader(currentCount)),
      current,
      currentCount,
      nextCount * 2,
    );
    releaseOwnedBuffer(current);
    current = next;
    currentCount = nextCount;
  }
  const readBuffer = device.createBuffer({
    size: 2 * Float32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_MAP_READ_COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(current.buffer, 0, readBuffer, 0, 2 * Float32Array.BYTES_PER_ELEMENT);
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(MAP_MODE_READ);
  const pair = new Float32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  readBuffer.destroy();
  releaseOwnedBuffer(current);
  return [pair[0], pair[1]] as const;
};

export const createGpuVectorOps = (
  device: GPUDevice,
  count: number,
): InputOptimizerVectorOps<GpuBufferRef> => {
  const getPipeline = createPipelineCache(device);
  const clonePipeline = getPipeline("clone", makeCloneShader(count));
  const scalePipeline = getPipeline("scale", makeScaleShader(count));
  const addScaledPipeline = getPipeline("add-scaled", makeAddScaledShader(count));
  const updateClampPipeline = getPipeline("update-clamp", makeUpdateClampShader(count));
  const scalarParameterBuffer = device.createBuffer({
    size: Float32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_STORAGE_COPY_DST,
  });
  const learningRateParameterBuffer = device.createBuffer({
    size: Float32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_STORAGE_COPY_DST,
  });

  return {
    clone: async (input: GpuBufferRef): Promise<GpuBufferRef> =>
      runSingleInputShader(device, clonePipeline, input, count),

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
    ): Promise<GpuBufferRef> => {
      writeScalarParameter(device, scalarParameterBuffer, scalar);
      const out = runSingleInputShader(
        device,
        scalePipeline,
        input,
        count,
        [{ binding: 2, resource: { buffer: scalarParameterBuffer } }],
      );
      return out;
    },

    addScaled: async (
      input: GpuBufferRef,
      direction: GpuBufferRef,
      scalar: number,
    ): Promise<GpuBufferRef> => {
      writeScalarParameter(device, scalarParameterBuffer, scalar);
      const out = runTwoInputShader(
        device,
        addScaledPipeline,
        input,
        direction,
        count,
        [{ binding: 3, resource: { buffer: scalarParameterBuffer } }],
      );
      return out;
    },

    dot: async (a: GpuBufferRef, b: GpuBufferRef): Promise<number> =>
      dotProduct(device, getPipeline, a, b, count),

    dotPairWithRight: async (
      leftA: GpuBufferRef,
      leftB: GpuBufferRef,
      right: GpuBufferRef,
    ): Promise<readonly [number, number]> =>
      dotPairWithRight(device, getPipeline, leftA, leftB, right, count),

    absSum: async (input: GpuBufferRef): Promise<number> =>
      absSum(device, getPipeline, input, count),

    updateClamp: async (
      input: GpuBufferRef,
      direction: GpuBufferRef,
      learningRate: number,
    ): Promise<GpuBufferRef> => {
      writeScalarParameter(
        device,
        learningRateParameterBuffer,
        learningRate,
      );
      const out = runTwoInputShader(
        device,
        updateClampPipeline,
        input,
        direction,
        count,
        [{ binding: 3, resource: { buffer: learningRateParameterBuffer } }],
      );
      return out;
    },

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
  };
};
