import { makeMseDiffShader, makeReduceSumShader } from "./mse.shader";
import { ownedBuffer, type GpuBufferRef } from "../../runtime/bufferKernels";
import { BUFFER_USAGE_STORAGE_COPY_DST } from "../../runtime/gpuFlags";

export const runMse = async (
  gpuDevice: GPUDevice | null,
  acquireReusableBuffer: (
    device: GPUDevice,
    size: number,
    usage: number,
  ) => GPUBuffer,
  releaseReusableBuffer: (
    size: number,
    usage: number,
    buffer: GPUBuffer,
  ) => void,
  a: Float32Array,
  b: Float32Array,
  BUFFER_USAGE_STORAGE_COPY_DST: number,
  BUFFER_USAGE_STORAGE_COPY_SRC: number,
  BUFFER_USAGE_MAP_READ_COPY_DST: number,
  MAP_MODE_READ: number,
): Promise<number> => {
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const device = gpuDevice;
  const count = a.length;
  const bytes = count * 4;
  const aBuffer = acquireReusableBuffer(
    device,
    bytes,
    BUFFER_USAGE_STORAGE_COPY_DST,
  );
  const bBuffer = acquireReusableBuffer(
    device,
    bytes,
    BUFFER_USAGE_STORAGE_COPY_DST,
  );
  let currentBuffer = acquireReusableBuffer(
    device,
    bytes,
    BUFFER_USAGE_STORAGE_COPY_SRC,
  );
  let currentBytes = bytes;
  device.queue.writeBuffer(aBuffer, 0, a);
  device.queue.writeBuffer(bBuffer, 0, b);
  const diffPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: makeMseDiffShader(count) }),
      entryPoint: "main",
    },
  });
  const diffBindGroup = device.createBindGroup({
    layout: diffPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuffer } },
      { binding: 1, resource: { buffer: bBuffer } },
      { binding: 2, resource: { buffer: currentBuffer } },
    ],
  });
  const e1 = device.createCommandEncoder();
  const p1 = e1.beginComputePass();
  p1.setPipeline(diffPipeline);
  p1.setBindGroup(0, diffBindGroup);
  p1.dispatchWorkgroups(Math.ceil(count / 64));
  p1.end();
  device.queue.submit([e1.finish()]);
  let currentCount = count;
  while (currentCount > 1) {
    const nextCount = Math.ceil(currentCount / 64);
    const nextBytes = nextCount * 4;
    const nextBuffer = acquireReusableBuffer(
      device,
      nextBytes,
      BUFFER_USAGE_STORAGE_COPY_SRC,
    );
    const reducePipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          code: makeReduceSumShader(currentCount),
        }),
        entryPoint: "main",
      },
    });
    const reduceBindGroup = device.createBindGroup({
      layout: reducePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: currentBuffer } },
        { binding: 1, resource: { buffer: nextBuffer } },
      ],
    });
    const e = device.createCommandEncoder();
    const p = e.beginComputePass();
    p.setPipeline(reducePipeline);
    p.setBindGroup(0, reduceBindGroup);
    p.dispatchWorkgroups(nextCount);
    p.end();
    device.queue.submit([e.finish()]);
    releaseReusableBuffer(
      currentBytes,
      BUFFER_USAGE_STORAGE_COPY_SRC,
      currentBuffer,
    );
    currentBuffer = nextBuffer;
    currentBytes = nextBytes;
    currentCount = nextCount;
  }
  const readBuffer = acquireReusableBuffer(
    device,
    4,
    BUFFER_USAGE_MAP_READ_COPY_DST,
  );
  const er = device.createCommandEncoder();
  er.copyBufferToBuffer(currentBuffer, 0, readBuffer, 0, 4);
  device.queue.submit([er.finish()]);
  await readBuffer.mapAsync(MAP_MODE_READ);
  const sse = new Float32Array(readBuffer.getMappedRange().slice(0))[0];
  readBuffer.unmap();
  releaseReusableBuffer(bytes, BUFFER_USAGE_STORAGE_COPY_DST, aBuffer);
  releaseReusableBuffer(bytes, BUFFER_USAGE_STORAGE_COPY_DST, bBuffer);
  releaseReusableBuffer(
    currentBytes,
    BUFFER_USAGE_STORAGE_COPY_SRC,
    currentBuffer,
  );
  releaseReusableBuffer(4, BUFFER_USAGE_MAP_READ_COPY_DST, readBuffer);
  return sse / count;
};

export const runMseBuffer = async (
  gpuDevice: GPUDevice | null,
  acquireReusableBuffer: (
    size: number,
    usage: number,
  ) => GPUBuffer,
  a: GpuBufferRef,
  b: GpuBufferRef,
  count: number,
  BUFFER_USAGE_STORAGE_COPY_SRC: number,
  BUFFER_USAGE_MAP_READ_COPY_DST: number,
  MAP_MODE_READ: number,
): Promise<number> => {
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const device = gpuDevice;
  const bytes = count * 4;
  let currentBuffer = acquireReusableBuffer(bytes, BUFFER_USAGE_STORAGE_COPY_SRC);
  const diffPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: makeMseDiffShader(count) }),
      entryPoint: "main",
    },
  });
  const diffBindGroup = device.createBindGroup({
    layout: diffPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: a.buffer } },
      { binding: 1, resource: { buffer: b.buffer } },
      { binding: 2, resource: { buffer: currentBuffer } },
    ],
  });
  const e1 = device.createCommandEncoder();
  const p1 = e1.beginComputePass();
  p1.setPipeline(diffPipeline);
  p1.setBindGroup(0, diffBindGroup);
  p1.dispatchWorkgroups(Math.ceil(count / 64));
  p1.end();
  device.queue.submit([e1.finish()]);
  let currentCount = count;
  while (currentCount > 1) {
    const nextCount = Math.ceil(currentCount / 64);
    const nextBytes = nextCount * 4;
    const nextBuffer = acquireReusableBuffer(nextBytes, BUFFER_USAGE_STORAGE_COPY_SRC);
    const reducePipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: makeReduceSumShader(currentCount) }),
        entryPoint: "main",
      },
    });
    const reduceBindGroup = device.createBindGroup({
      layout: reducePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: currentBuffer } },
        { binding: 1, resource: { buffer: nextBuffer } },
      ],
    });
    const e = device.createCommandEncoder();
    const p = e.beginComputePass();
    p.setPipeline(reducePipeline);
    p.setBindGroup(0, reduceBindGroup);
    p.dispatchWorkgroups(nextCount);
    p.end();
    device.queue.submit([e.finish()]);

    currentBuffer = nextBuffer;
    currentCount = nextCount;
  }
  const readBuffer = acquireReusableBuffer(4, BUFFER_USAGE_MAP_READ_COPY_DST);
  const er = device.createCommandEncoder();
  er.copyBufferToBuffer(currentBuffer, 0, readBuffer, 0, 4);
  device.queue.submit([er.finish()]);
  await readBuffer.mapAsync(MAP_MODE_READ);
  const sse = new Float32Array(readBuffer.getMappedRange().slice(0))[0];
  readBuffer.unmap();

  return sse / count;
};

export const runMseBufferToScalarBuffer = async (
  gpuDevice: GPUDevice | null,
  acquireReusableBuffer: (
    size: number,
    usage: number,
  ) => GPUBuffer,
  a: GpuBufferRef,
  b: GpuBufferRef,
  count: number,
  BUFFER_USAGE_STORAGE_COPY_SRC: number,
): Promise<GpuBufferRef> => {
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const device = gpuDevice;
  const bytes = count * 4;
  let currentBuffer = acquireReusableBuffer(bytes, BUFFER_USAGE_STORAGE_COPY_SRC);
  const diffPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: makeMseDiffShader(count) }),
      entryPoint: "main",
    },
  });
  const diffBindGroup = device.createBindGroup({
    layout: diffPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: a.buffer } },
      { binding: 1, resource: { buffer: b.buffer } },
      { binding: 2, resource: { buffer: currentBuffer } },
    ],
  });
  const e1 = device.createCommandEncoder();
  const p1 = e1.beginComputePass();
  p1.setPipeline(diffPipeline);
  p1.setBindGroup(0, diffBindGroup);
  p1.dispatchWorkgroups(Math.ceil(count / 64));
  p1.end();
  device.queue.submit([e1.finish()]);

  let currentCount = count;
  while (currentCount > 1) {
    const nextCount = Math.ceil(currentCount / 64);
    const nextBytes = nextCount * 4;
    const nextBuffer = acquireReusableBuffer(nextBytes, BUFFER_USAGE_STORAGE_COPY_SRC);
    const reducePipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: makeReduceSumShader(currentCount) }),
        entryPoint: "main",
      },
    });
    const reduceBindGroup = device.createBindGroup({
      layout: reducePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: currentBuffer } },
        { binding: 1, resource: { buffer: nextBuffer } },
      ],
    });
    const e = device.createCommandEncoder();
    const p = e.beginComputePass();
    p.setPipeline(reducePipeline);
    p.setBindGroup(0, reduceBindGroup);
    p.dispatchWorkgroups(nextCount);
    p.end();
    device.queue.submit([e.finish()]);

    currentBuffer = nextBuffer;
    currentCount = nextCount;
  }

  const outBuffer = device.createBuffer({
    size: Float32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_STORAGE_COPY_SRC | BUFFER_USAGE_STORAGE_COPY_DST,
  });
  const copyEncoder = device.createCommandEncoder();
  copyEncoder.copyBufferToBuffer(
    currentBuffer,
    0,
    outBuffer,
    0,
    Float32Array.BYTES_PER_ELEMENT,
  );
  device.queue.submit([copyEncoder.finish()]);
  return ownedBuffer(outBuffer);
};
