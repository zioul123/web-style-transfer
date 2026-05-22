/// <reference lib="webworker" />

import { acquireReusableBuffer, releaseReusableBuffer } from "./bufferPool";
import { getGpuDevice } from "./deviceState";
import {
  BUFFER_USAGE_MAP_READ_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_SRC,
  MAP_MODE_READ,
} from "./gpuFlags";
import {
  makeBinaryOpShader,
  makeClampShader,
  runUnaryShader,
} from "./shaderRunner";

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
 * Internal chain primitive for handle -> handle compute dispatch.
 *
 * Unlike `runUnary`, this never maps the output buffer back to CPU.
 * The caller gets a new runtime-owned tensor handle and can keep chaining
 * GPU-side ops until they intentionally cross a readback boundary.
 */
export const runUnaryOnHandle = (
  input: RuntimeTensorHandle,
  code: string,
  extraEntries: GPUBindGroupEntry[] = [],
): RuntimeTensorHandle => {
  const device = requireGpuDevice();
  const outBuffer = acquireReusableBuffer(
    device,
    input.byteLength,
    BUFFER_USAGE_STORAGE_COPY_SRC,
  );
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code }),
      entryPoint: "main",
    },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.buffer } },
      ...extraEntries,
      { binding: extraEntries.length + 1, resource: { buffer: outBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(input.elementCount / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
  return makeHandle(input.shape, outBuffer, BUFFER_USAGE_STORAGE_COPY_SRC, "runtime");
};

/**
 * Chain helper: apply a scalar op to a tensor-handle value and return a new
 * runtime-owned handle.
 *
 * Implementation note:
 * - The scalar is uploaded as a 1-value GPU buffer and dispatched via the
 *   existing `makeBinaryOpShader(..., "tensorScalar")` generator.
 * - Input and output remain GPU-resident; there is no implicit readback here.
 */
export const chainTensorScalarOp = async (
  input: RuntimeTensorHandle,
  op: "sub" | "add" | "mul",
  scalar: number,
): Promise<RuntimeTensorHandle> => {
  const device = requireGpuDevice();
  const scalarBuffer = acquireReusableBuffer(
    device,
    4,
    BUFFER_USAGE_STORAGE_COPY_DST,
  );
  device.queue.writeBuffer(scalarBuffer, 0, new Float32Array([scalar]));
  const outHandle = runUnaryOnHandle(
    input,
    makeBinaryOpShader(op, input.elementCount, "tensorScalar"),
    [{ binding: 1, resource: { buffer: scalarBuffer } }],
  );
  releaseReusableBuffer(4, BUFFER_USAGE_STORAGE_COPY_DST, scalarBuffer);
  return outHandle;
};

/**
 * Chain helper: apply elementwise binary op between two same-shaped handles.
 *
 * This keeps both operands and the result on GPU until the caller chooses an
 * explicit boundary helper.
 */
export const chainTensorBinaryOp = async (
  a: RuntimeTensorHandle,
  b: RuntimeTensorHandle,
  op: "add" | "sub" | "mul" | "div",
): Promise<RuntimeTensorHandle> => {
  if (a.elementCount !== b.elementCount) {
    throw new Error("chainTensorBinaryOp expects tensors with equal element count.");
  }
  return runUnaryOnHandle(
    a,
    makeBinaryOpShader(op, a.elementCount, "tensorTensor"),
    [{ binding: 1, resource: { buffer: b.buffer } }],
  );
};

/**
 * Chain helper: clamp tensor values into the image-valid range `[0, 1]` and
 * return a new runtime-owned handle.
 *
 * Uses shared `makeClampShader` shader generator from `shaderRunner.ts`.
 */
export const chainClamp01 = async (
  input: RuntimeTensorHandle,
): Promise<RuntimeTensorHandle> => {
  return runUnaryOnHandle(input, makeClampShader(input.elementCount, 0, 1));
};
