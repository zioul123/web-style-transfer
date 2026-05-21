/// <reference lib="webworker" />

import { acquireReusableBuffer, releaseReusableBuffer } from "./bufferPool";
import { getGpuDevice } from "./deviceState";

const requireGpuDevice = (): GPUDevice => {
  const gpuDevice = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  return gpuDevice;
};
import {
  BUFFER_USAGE_MAP_READ_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_DST,
  MAP_MODE_READ,
} from "./gpuFlags";
import { runUnaryShader } from "./shaderRunner";

export type RuntimeTensorShape = readonly [number, number, number, number];

export type RuntimeTensorHandle = {
  readonly kind: "gpu-tensor-handle";
  readonly shape: RuntimeTensorShape;
  readonly elementCount: number;
  readonly byteLength: number;
  readonly usage: number;
  readonly buffer: GPUBuffer;
  readonly owner: "runtime" | "external";
};

const makeHandle = (
  shape: RuntimeTensorShape,
  buffer: GPUBuffer,
  usage: number,
  owner: "runtime" | "external",
): RuntimeTensorHandle => ({
  kind: "gpu-tensor-handle",
  shape,
  elementCount: shape[0] * shape[1] * shape[2] * shape[3],
  byteLength: shape[0] * shape[1] * shape[2] * shape[3] * 4,
  usage,
  buffer,
  owner,
});

export const acquireTensorHandleFromCpu = (
  shape: RuntimeTensorShape,
  values: Float32Array,
): RuntimeTensorHandle => {
  const device = requireGpuDevice();
  const buffer = acquireReusableBuffer(
    device,
    values.byteLength,
    BUFFER_USAGE_STORAGE_COPY_DST,
  );
  device.queue.writeBuffer(buffer, 0, values);
  return makeHandle(shape, buffer, BUFFER_USAGE_STORAGE_COPY_DST, "runtime");
};

export const releaseTensorHandle = (handle: RuntimeTensorHandle): void => {
  if (handle.owner !== "runtime") return;
  releaseReusableBuffer(handle.byteLength, handle.usage, handle.buffer);
};

export const readTensorHandleToCpu = async (
  handle: RuntimeTensorHandle,
): Promise<Float32Array> => {
  const device = requireGpuDevice();
  const readBuffer = acquireReusableBuffer(
    device,
    handle.byteLength,
    BUFFER_USAGE_MAP_READ_COPY_DST,
  );
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(handle.buffer, 0, readBuffer, 0, handle.byteLength);
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(MAP_MODE_READ);
  const out = new Float32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  releaseReusableBuffer(handle.byteLength, BUFFER_USAGE_MAP_READ_COPY_DST, readBuffer);
  return out;
};

export const readScalarBoundary = async (
  handle: RuntimeTensorHandle,
): Promise<number> => {
  const values = await readTensorHandleToCpu(handle);
  return values[0];
};

export const readImageSnapshotBoundary = async (
  handle: RuntimeTensorHandle,
): Promise<Float32Array> => readTensorHandleToCpu(handle);

export const runUnary = (
  code: string,
  input: Float32Array,
  outCount: number,
  extraEntries: GPUBindGroupEntry[] = [],
): Promise<Float32Array> =>
  runUnaryShader(getGpuDevice(), code, input, outCount, extraEntries);

const makeBinaryScalarShader = (
  count: number,
  op: "sub" | "add" | "mul",
  scalar: number,
): string => {
  const expr = op === "sub" ? `a[i] - ${scalar}` : op === "add" ? `a[i] + ${scalar}` : `a[i] * ${scalar}`;
  return `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = ${expr};
}`;
};

const makeClampShader = (count: number): string => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  out[i] = clamp(a[i], 0.0, 1.0);
}`;

export const chainTensorScalarOp = async (
  input: RuntimeTensorHandle,
  op: "sub" | "add" | "mul",
  scalar: number,
): Promise<RuntimeTensorHandle> => {
  const out = await runUnary(
    makeBinaryScalarShader(input.elementCount, op, scalar),
    await readTensorHandleToCpu(input),
    input.elementCount,
  );
  return acquireTensorHandleFromCpu(input.shape, out);
};

export const chainClamp01 = async (
  input: RuntimeTensorHandle,
): Promise<RuntimeTensorHandle> => {
  const out = await runUnary(
    makeClampShader(input.elementCount),
    await readTensorHandleToCpu(input),
    input.elementCount,
  );
  return acquireTensorHandleFromCpu(input.shape, out);
};
