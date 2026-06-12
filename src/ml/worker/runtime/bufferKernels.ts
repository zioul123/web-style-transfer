import {
  BUFFER_USAGE_MAP_READ_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_SRC,
  BUFFER_USAGE_UNIFORM_COPY_DST,
  MAP_MODE_READ,
} from "./gpuFlags";
import { acquireReusableBuffer, releaseReusableBuffer } from "./bufferPool";
import { getOrCreateComputePipeline } from "./computePipelineCache";
import { runBinaryOpToBuffer } from "./shaderRunner";
import { getGpuDevice } from "./deviceState";

export type OwnedGpuBuffer = { owned: true; buffer: GPUBuffer };
export type BorrowedGpuBuffer = { owned: false; buffer: GPUBuffer };
export type GpuBufferRef = OwnedGpuBuffer | BorrowedGpuBuffer;
type KernelRuntimeOptions = { useCachedPipelines: boolean };
type StorageUploadArray = Float32Array | Int32Array | Uint32Array;
type StorageUploadArrayConstructor<T extends StorageUploadArray> = {
  new (values: ArrayLike<number>): T;
};

const kernelRuntimeOptions: KernelRuntimeOptions = {
  useCachedPipelines: false,
};

export const setKernelRuntimeOptions = (
  options: Partial<KernelRuntimeOptions>,
): void => {
  if (options.useCachedPipelines !== undefined) {
    kernelRuntimeOptions.useCachedPipelines = options.useCachedPipelines;
  }
};

export const ownedBuffer = (buffer: GPUBuffer): OwnedGpuBuffer => ({
  owned: true,
  buffer,
});
export const borrowedBuffer = (buffer: GPUBuffer): BorrowedGpuBuffer => ({
  owned: false,
  buffer,
});

export const releaseOwnedBuffer = (bufferRef: GpuBufferRef): void => {
  if (bufferRef.owned) {
    bufferRef.buffer.destroy();
  }
};

export const uploadToOwnedBuffer = (
  gpuDevice: GPUDevice | null,
  values: Float32Array,
): OwnedGpuBuffer => {
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const buffer: GPUBuffer = gpuDevice.createBuffer({
    size: values.byteLength,
    usage: BUFFER_USAGE_STORAGE_COPY_DST,
  });
  gpuDevice.queue.writeBuffer(buffer, 0, values);
  return ownedBuffer(buffer);
};

const uploadTypedStorageBuffer = <T extends StorageUploadArray>(
  gpuDevice: GPUDevice,
  values: T | readonly number[],
  TypedArray: StorageUploadArrayConstructor<T>,
): GPUBuffer => {
  const data = values instanceof TypedArray ? values : new TypedArray(values);
  const buffer: GPUBuffer = gpuDevice.createBuffer({
    size: data.byteLength,
    usage: BUFFER_USAGE_STORAGE_COPY_DST,
  });
  gpuDevice.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

export const uploadFloat32StorageBuffer = (
  gpuDevice: GPUDevice,
  values: Float32Array | readonly number[],
): GPUBuffer => uploadTypedStorageBuffer(gpuDevice, values, Float32Array);

export const uploadInt32StorageBuffer = (
  gpuDevice: GPUDevice,
  values: Int32Array | readonly number[],
): GPUBuffer => uploadTypedStorageBuffer(gpuDevice, values, Int32Array);

export const uploadUint32StorageBuffer = (
  gpuDevice: GPUDevice,
  values: Uint32Array | readonly number[],
): GPUBuffer => uploadTypedStorageBuffer(gpuDevice, values, Uint32Array);

export const uploadUint32UniformBuffer = (
  gpuDevice: GPUDevice,
  values: readonly number[],
): GPUBuffer => {
  const wordCount = Math.max(4, Math.ceil(values.length / 4) * 4);
  const data = new Uint32Array(wordCount);
  data.set(values);
  const buffer: GPUBuffer = gpuDevice.createBuffer({
    size: data.byteLength,
    usage: BUFFER_USAGE_UNIFORM_COPY_DST,
  });
  gpuDevice.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

export const readGpuBufferToArray = async (
  gpuDevice: GPUDevice | null,
  sourceBuffer: GPUBuffer,
  count: number,
): Promise<Float32Array> => {
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const byteLength: number = count * 4;
  const readBuffer: GPUBuffer = acquireReusableBuffer(
    gpuDevice,
    byteLength,
    BUFFER_USAGE_MAP_READ_COPY_DST,
  );
  const encoder: GPUCommandEncoder = gpuDevice.createCommandEncoder();
  encoder.copyBufferToBuffer(sourceBuffer, 0, readBuffer, 0, byteLength);
  gpuDevice.queue.submit([encoder.finish()]);
  let mapped = false;
  try {
    await readBuffer.mapAsync(MAP_MODE_READ);
    mapped = true;
    const out: Float32Array = new Float32Array(
      readBuffer.getMappedRange().slice(0),
    );
    readBuffer.unmap();
    mapped = false;
    releaseReusableBuffer(
      byteLength,
      BUFFER_USAGE_MAP_READ_COPY_DST,
      readBuffer,
    );
    return out;
  } catch (error) {
    if (mapped) readBuffer.unmap();
    readBuffer.destroy();
    throw error;
  }
};

export const runUnaryShaderToBuffer = (
  gpuDevice: GPUDevice | null,
  code: string,
  inputBuffer: GPUBuffer,
  outCount: number,
  extraEntries: GPUBindGroupEntry[] = [],
): GPUBuffer => {
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const outBuffer: GPUBuffer = gpuDevice.createBuffer({
    size: outCount * 4,
    usage: BUFFER_USAGE_STORAGE_COPY_SRC,
  });
  const pipeline = kernelRuntimeOptions.useCachedPipelines
    ? getOrCreateComputePipeline(gpuDevice, code, code)
    : gpuDevice.createComputePipeline({
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
  const encoder: GPUCommandEncoder = gpuDevice.createCommandEncoder();
  const pass: GPUComputePassEncoder = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(outCount / 64));
  pass.end();
  gpuDevice.queue.submit([encoder.finish()]);
  return outBuffer;
};

export const runUnaryShaderToBufferWithDispatch = (
  gpuDevice: GPUDevice | null,
  code: string,
  inputBuffer: GPUBuffer,
  outputFloatCount: number,
  dispatchElementCount: number,
  extraEntries: GPUBindGroupEntry[] = [],
): GPUBuffer => {
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const outBuffer: GPUBuffer = gpuDevice.createBuffer({
    size: outputFloatCount * Float32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_STORAGE_COPY_SRC,
  });
  const pipeline = kernelRuntimeOptions.useCachedPipelines
    ? getOrCreateComputePipeline(gpuDevice, code, code)
    : gpuDevice.createComputePipeline({
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
  const encoder: GPUCommandEncoder = gpuDevice.createCommandEncoder();
  const pass: GPUComputePassEncoder = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(dispatchElementCount / 64));
  pass.end();
  gpuDevice.queue.submit([encoder.finish()]);
  return outBuffer;
};

export const runScalarMulBuffer = async (
  input: GpuBufferRef,
  count: number,
  scalar: number,
): Promise<GpuBufferRef> => {
  const gpuDevice: GPUDevice | null = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const scalarBuffer = uploadToOwnedBuffer(
    gpuDevice,
    new Float32Array([scalar]),
  );
  const outputBuffer = runBinaryOpToBuffer(
    gpuDevice,
    "mul",
    input.buffer,
    scalarBuffer.buffer,
    count,
    "tensorScalar",
  );
  releaseOwnedBuffer(scalarBuffer);
  return ownedBuffer(outputBuffer);
};

export const runTensorAddBuffer = async (
  a: GpuBufferRef,
  b: GpuBufferRef,
  count: number,
): Promise<GpuBufferRef> => {
  const gpuDevice: GPUDevice | null = getGpuDevice();
  const outputBuffer = runBinaryOpToBuffer(
    gpuDevice,
    "add",
    a.buffer,
    b.buffer,
    count,
    "tensorTensor",
  );
  return ownedBuffer(outputBuffer);
};
