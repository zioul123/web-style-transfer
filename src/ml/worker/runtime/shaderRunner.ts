import { acquireReusableBuffer, releaseReusableBuffer } from "./bufferPool";
import { getOrCreateComputePipeline } from "./computePipelineCache";
import {
  BUFFER_USAGE_MAP_READ_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_SRC,
  MAP_MODE_READ,
} from "./gpuFlags";

const BINARY_SHADER_SNIPPETS: Record<
  "add" | "sub" | "mul" | "div",
  { tensorTensor: string; tensorScalar: string; scalarTensor: string }
> = {
  add: {
    tensorTensor: "out[i] = a[i] + b[i];",
    tensorScalar: "out[i] = a[i] + b[0];",
    scalarTensor: "out[i] = b[0] + a[i];",
  },
  sub: {
    tensorTensor: "out[i] = a[i] - b[i];",
    tensorScalar: "out[i] = a[i] - b[0];",
    scalarTensor: "out[i] = b[0] - a[i];",
  },
  mul: {
    tensorTensor: "out[i] = a[i] * b[i];",
    tensorScalar: "out[i] = a[i] * b[0];",
    scalarTensor: "out[i] = b[0] * a[i];",
  },
  div: {
    tensorTensor: "out[i] = a[i] / b[i];",
    tensorScalar: "out[i] = a[i] / b[0];",
    scalarTensor: "out[i] = b[0] / a[i];",
  },
};

export const makeBinaryOpShader = (
  op: "add" | "sub" | "mul" | "div",
  count: number,
  mode: "tensorTensor" | "tensorScalar" | "scalarTensor",
): string => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  ${BINARY_SHADER_SNIPPETS[op][mode]}
}`;

export const makeBinaryOpVec4Shader = (
  op: "add" | "sub" | "mul" | "div",
  vecCount: number,
  mode: "tensorTensor" | "tensorScalar" | "scalarTensor",
): string => {
  const expr = mode === "tensorTensor"
    ? `out[i] = a[i] ${op === "add" ? "+" : op === "sub" ? "-" : op === "mul" ? "*" : "/"} b[i];`
    : mode === "tensorScalar"
      ? `out[i] = a[i] ${op === "add" ? "+" : op === "sub" ? "-" : op === "mul" ? "*" : "/"} b[0];`
      : `out[i] = b[0] ${op === "add" ? "+" : op === "sub" ? "-" : op === "mul" ? "*" : "/"} a[i];`;
  return `
@group(0) @binding(0) var<storage, read> a: array<vec4f>;
@group(0) @binding(1) var<storage, read> b: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${vecCount}u) { return; }
  ${expr}
}`;
};

export const makeClampShader = (
  count: number,
  min: number,
  max: number,
): string => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = clamp(a[i], ${min}, ${max});
}`;


export const runBinaryOpToBuffer = (
  gpuDevice: GPUDevice | null,
  op: "add" | "sub" | "mul" | "div",
  aBuffer: GPUBuffer,
  bBuffer: GPUBuffer,
  count: number,
  mode: "tensorTensor" | "tensorScalar" | "scalarTensor",
  useVec4: boolean = false,
): GPUBuffer => {
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const shouldUseVec4 = useVec4 && count % 4 === 0;
  const dispatchCount = shouldUseVec4 ? count / 4 : count;
  const outBuffer = gpuDevice.createBuffer({
    size: count * 4,
    usage: BUFFER_USAGE_STORAGE_COPY_SRC,
  });
  const shaderCode = shouldUseVec4
    ? makeBinaryOpVec4Shader(op, dispatchCount, mode)
    : makeBinaryOpShader(op, count, mode);
  const pipeline = getOrCreateComputePipeline(
    gpuDevice,
    shouldUseVec4
      ? `binary-vec4:${op}:${dispatchCount}:${mode}`
      : `binary:${op}:${count}:${mode}`,
    shaderCode,
  );
  const bindGroup = gpuDevice.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuffer } },
      { binding: 1, resource: { buffer: bBuffer } },
      { binding: 2, resource: { buffer: outBuffer } },
    ],
  });
  const encoder = gpuDevice.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(dispatchCount / 64));
  pass.end();
  gpuDevice.queue.submit([encoder.finish()]);
  return outBuffer;
};
export const runBinaryOp = async (
  gpuDevice: GPUDevice | null,
  op: "add" | "sub" | "mul" | "div",
  a: Float32Array,
  b: Float32Array,
  mode: "tensorTensor" | "tensorScalar" | "scalarTensor",
): Promise<Float32Array> => {
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const count = a.length;
  const bytes = count * 4;
  const aBuffer = acquireReusableBuffer(
    gpuDevice,
    bytes,
    BUFFER_USAGE_STORAGE_COPY_DST,
  );
  const bBuffer = acquireReusableBuffer(
    gpuDevice,
    bytes,
    BUFFER_USAGE_STORAGE_COPY_DST,
  );
  const outBuffer = acquireReusableBuffer(
    gpuDevice,
    bytes,
    BUFFER_USAGE_STORAGE_COPY_SRC,
  );
  const readBuffer = acquireReusableBuffer(
    gpuDevice,
    bytes,
    BUFFER_USAGE_MAP_READ_COPY_DST,
  );
  gpuDevice.queue.writeBuffer(aBuffer, 0, a);
  gpuDevice.queue.writeBuffer(bBuffer, 0, b);
  const pipeline = gpuDevice.createComputePipeline({
    layout: "auto",
    compute: {
      module: gpuDevice.createShaderModule({
        code: makeBinaryOpShader(op, count, mode),
      }),
      entryPoint: "main",
    },
  });
  const bindGroup = gpuDevice.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuffer } },
      { binding: 1, resource: { buffer: bBuffer } },
      { binding: 2, resource: { buffer: outBuffer } },
    ],
  });
  const encoder = gpuDevice.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(outBuffer, 0, readBuffer, 0, bytes);
  gpuDevice.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(MAP_MODE_READ);
  const arr = new Float32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  releaseReusableBuffer(bytes, BUFFER_USAGE_STORAGE_COPY_DST, aBuffer);
  releaseReusableBuffer(bytes, BUFFER_USAGE_STORAGE_COPY_DST, bBuffer);
  releaseReusableBuffer(bytes, BUFFER_USAGE_STORAGE_COPY_SRC, outBuffer);
  releaseReusableBuffer(bytes, BUFFER_USAGE_MAP_READ_COPY_DST, readBuffer);
  return arr;
};

export const runScalarBinaryOp = async (
  gpuDevice: GPUDevice | null,
  op: "add" | "sub" | "mul" | "div",
  tensor: Float32Array,
  scalar: number,
  scalarOnLeft: boolean,
): Promise<Float32Array> =>
  runBinaryOp(
    gpuDevice,
    op,
    tensor,
    new Float32Array([scalar]),
    scalarOnLeft ? "scalarTensor" : "tensorScalar",
  );

export const runClamp = async (
  gpuDevice: GPUDevice | null,
  a: Float32Array,
  min: number,
  max: number,
): Promise<Float32Array> => {
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const count = a.length;
  const bytes = count * 4;
  const aBuffer = acquireReusableBuffer(
    gpuDevice,
    bytes,
    BUFFER_USAGE_STORAGE_COPY_DST,
  );
  const outBuffer = acquireReusableBuffer(
    gpuDevice,
    bytes,
    BUFFER_USAGE_STORAGE_COPY_SRC,
  );
  const readBuffer = acquireReusableBuffer(
    gpuDevice,
    bytes,
    BUFFER_USAGE_MAP_READ_COPY_DST,
  );
  gpuDevice.queue.writeBuffer(aBuffer, 0, a);
  const pipeline = gpuDevice.createComputePipeline({
    layout: "auto",
    compute: {
      module: gpuDevice.createShaderModule({
        code: makeClampShader(count, min, max),
      }),
      entryPoint: "main",
    },
  });
  const bindGroup = gpuDevice.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuffer } },
      { binding: 1, resource: { buffer: outBuffer } },
    ],
  });
  const encoder = gpuDevice.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(outBuffer, 0, readBuffer, 0, bytes);
  gpuDevice.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(MAP_MODE_READ);
  const arr = new Float32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  releaseReusableBuffer(bytes, BUFFER_USAGE_STORAGE_COPY_DST, aBuffer);
  releaseReusableBuffer(bytes, BUFFER_USAGE_STORAGE_COPY_SRC, outBuffer);
  releaseReusableBuffer(bytes, BUFFER_USAGE_MAP_READ_COPY_DST, readBuffer);
  return arr;
};

export const runUnaryShader = async (
  gpuDevice: GPUDevice | null,
  code: string,
  input: Float32Array,
  outCount: number,
  extraEntries: GPUBindGroupEntry[] = [],
): Promise<Float32Array> => {
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const inputBuffer = acquireReusableBuffer(
    gpuDevice,
    input.byteLength,
    BUFFER_USAGE_STORAGE_COPY_DST,
  );
  const outBuffer = acquireReusableBuffer(
    gpuDevice,
    outCount * 4,
    BUFFER_USAGE_STORAGE_COPY_SRC,
  );
  const readBuffer = acquireReusableBuffer(
    gpuDevice,
    outCount * 4,
    BUFFER_USAGE_MAP_READ_COPY_DST,
  );
  gpuDevice.queue.writeBuffer(inputBuffer, 0, input);
  const pipeline = gpuDevice.createComputePipeline({
    layout: "auto",
    compute: {
      module: gpuDevice.createShaderModule({ code }),
      entryPoint: "main",
    },
  });
  const bindGroup = gpuDevice.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      ...extraEntries,
      { binding: extraEntries.length + 1, resource: { buffer: outBuffer } },
    ],
  });
  const encoder = gpuDevice.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(outCount / 64));
  pass.end();
  encoder.copyBufferToBuffer(outBuffer, 0, readBuffer, 0, outCount * 4);
  gpuDevice.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(MAP_MODE_READ);
  const out = new Float32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  releaseReusableBuffer(
    input.byteLength,
    BUFFER_USAGE_STORAGE_COPY_DST,
    inputBuffer,
  );
  releaseReusableBuffer(outCount * 4, BUFFER_USAGE_STORAGE_COPY_SRC, outBuffer);
  releaseReusableBuffer(
    outCount * 4,
    BUFFER_USAGE_MAP_READ_COPY_DST,
    readBuffer,
  );
  return out;
};
