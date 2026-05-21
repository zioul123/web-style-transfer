/// <reference lib="webworker" />

import { acquireReusableBuffer, releaseReusableBuffer } from "./bufferPool";
import { getGpuDevice } from "./deviceState";
import {
  BUFFER_USAGE_MAP_READ_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_DST,
  MAP_MODE_READ,
} from "./gpuFlags";
import { runClamp, runScalarBinaryOp, runUnaryShader } from "./shaderRunner";

/**
 * Canonical shape type used by runtime tensor handles.
 *
 * We keep it as NCHW `[batch, channels, height, width]` to match the worker
 * protocol and the existing op surfaces.
 */
export type RuntimeTensorShape = readonly [number, number, number, number];

/**
 * Lightweight runtime IR for an intermediate tensor that is conceptually
 * "resident on GPU".
 *
 * - `buffer` points at the underlying GPU storage.
 * - `shape` + `elementCount` + `byteLength` let callers reason about size.
 * - `owner` makes lifecycle explicit:
 *   - `runtime`: this handle must be released via `releaseTensorHandle`.
 *   - `external`: buffer lifetime is owned elsewhere.
 */
export type RuntimeTensorHandle = {
  readonly kind: "gpu-tensor-handle";
  readonly shape: RuntimeTensorShape;
  readonly elementCount: number;
  readonly byteLength: number;
  readonly usage: number;
  readonly buffer: GPUBuffer;
  readonly owner: "runtime" | "external";
};

/**
 * Ensures we have an initialized device before runtime operations execute.
 */
const requireGpuDevice = (): GPUDevice => {
  const gpuDevice = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  return gpuDevice;
};

/**
 * Internal constructor for tensor handles.
 */
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

/**
 * Uploads CPU tensor values into a pooled GPU storage buffer and returns a
 * runtime-owned handle for downstream chaining.
 */
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

/**
 * Releases a runtime-owned handle back into the reusable buffer pool.
 *
 * No-op for externally-owned handles.
 */
export const releaseTensorHandle = (handle: RuntimeTensorHandle): void => {
  if (handle.owner !== "runtime") return;
  releaseReusableBuffer(handle.byteLength, handle.usage, handle.buffer);
};

/**
 * Explicit boundary: materialize a GPU tensor handle into CPU memory.
 *
 * This is the primary "GPU -> CPU" transition helper for tensor payloads.
 */
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

/**
 * Explicit boundary: extract a scalar from a tensor handle.
 *
 * Convention: scalar results are stored at index 0.
 */
export const readScalarBoundary = async (
  handle: RuntimeTensorHandle,
): Promise<number> => {
  const values = await readTensorHandleToCpu(handle);
  return values[0];
};

/**
 * Explicit boundary: read tensor values intended for preview/final image
 * snapshots on the main thread.
 */
export const readImageSnapshotBoundary = async (
  handle: RuntimeTensorHandle,
): Promise<Float32Array> => readTensorHandleToCpu(handle);

/**
 * Compatibility helper for existing op modules that still operate on
 * CPU-backed Float32Array payloads.
 */
export const runUnary = (
  code: string,
  input: Float32Array,
  outCount: number,
  extraEntries: GPUBindGroupEntry[] = [],
): Promise<Float32Array> =>
  runUnaryShader(getGpuDevice(), code, input, outCount, extraEntries);

/**
 * Chain helper: apply a scalar op to a tensor-handle value and return a new
 * runtime-owned handle.
 *
 * Implementation note:
 * - This currently reuses `runScalarBinaryOp` from `shaderRunner.ts` to avoid
 *   duplicate WGSL generation in this module.
 * - Today `runScalarBinaryOp` still returns CPU data, so this helper performs
 *   one temporary readback+reupload internally. The boundary is now centralized
 *   here so later fully-GPU chaining can be swapped in without changing callers.
 */
export const chainTensorScalarOp = async (
  input: RuntimeTensorHandle,
  op: "sub" | "add" | "mul",
  scalar: number,
): Promise<RuntimeTensorHandle> => {
  const inputCpu = await readTensorHandleToCpu(input);
  const out = await runScalarBinaryOp(getGpuDevice(), op, inputCpu, scalar, false);
  return acquireTensorHandleFromCpu(input.shape, out);
};

/**
 * Chain helper: clamp tensor values into the image-valid range `[0, 1]` and
 * return a new runtime-owned handle.
 *
 * Uses shared `runClamp` implementation from `shaderRunner.ts` for consistency.
 */
export const chainClamp01 = async (
  input: RuntimeTensorHandle,
): Promise<RuntimeTensorHandle> => {
  const inputCpu = await readTensorHandleToCpu(input);
  const out = await runClamp(getGpuDevice(), inputCpu, 0, 1);
  return acquireTensorHandleFromCpu(input.shape, out);
};
