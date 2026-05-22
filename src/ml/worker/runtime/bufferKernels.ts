import {
  BUFFER_USAGE_MAP_READ_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_SRC,
  MAP_MODE_READ,
} from "./gpuFlags";

export type OwnedGpuBuffer = { owned: true; buffer: GPUBuffer };
export type BorrowedGpuBuffer = { owned: false; buffer: GPUBuffer };
export type GpuBufferRef = OwnedGpuBuffer | BorrowedGpuBuffer;

export const ownedBuffer = (buffer: GPUBuffer): OwnedGpuBuffer => ({ owned: true, buffer });
export const borrowedBuffer = (buffer: GPUBuffer): BorrowedGpuBuffer => ({ owned: false, buffer });

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

export const readGpuBufferToArray = async (
  gpuDevice: GPUDevice | null,
  sourceBuffer: GPUBuffer,
  count: number,
): Promise<Float32Array> => {
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const byteLength: number = count * 4;
  const readBuffer: GPUBuffer = gpuDevice.createBuffer({
    size: byteLength,
    usage: BUFFER_USAGE_MAP_READ_COPY_DST,
  });
  const encoder: GPUCommandEncoder = gpuDevice.createCommandEncoder();
  encoder.copyBufferToBuffer(sourceBuffer, 0, readBuffer, 0, byteLength);
  gpuDevice.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(MAP_MODE_READ);
  const out: Float32Array = new Float32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  readBuffer.destroy();
  return out;
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
  const pipeline = gpuDevice.createComputePipeline({
    layout: "auto",
    compute: { module: gpuDevice.createShaderModule({ code }), entryPoint: "main" },
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
