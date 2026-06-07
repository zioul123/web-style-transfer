import { ownedBuffer, type GpuBufferRef } from "../../runtime/bufferKernels";
import { getOrCreateComputePipeline } from "../../runtime/computePipelineCache";
import { getGpuDevice } from "../../runtime/deviceState";
import { BUFFER_USAGE_STORAGE_COPY_SRC } from "../../runtime/gpuFlags";
import { makeWeightedScalarSumShader } from "./weightedScalarSum.shader";

export const runWeightedScalarSumBuffer = (
  scalarBuffers: readonly GpuBufferRef[],
  weights: readonly number[],
): GpuBufferRef => {
  if (scalarBuffers.length === 0) {
    throw new Error("weighted scalar sum requires at least one scalar buffer.");
  }
  if (scalarBuffers.length !== weights.length) {
    throw new Error(
      "weighted scalar sum requires matching scalar and weight counts.",
    );
  }
  const gpuDevice = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");

  const pipeline = getOrCreateComputePipeline(
    gpuDevice,
    `weighted-scalar-sum:${scalarBuffers.length}:${weights.join(",")}`,
    makeWeightedScalarSumShader(weights),
  );
  const outBuffer = gpuDevice.createBuffer({
    size: Float32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_STORAGE_COPY_SRC,
  });
  const outBindingIndex = scalarBuffers.length;
  const entries: GPUBindGroupEntry[] = scalarBuffers.map(
    (scalarBuffer, index) => ({
      binding: index,
      resource: { buffer: scalarBuffer.buffer },
    }),
  );
  entries.push({
    binding: outBindingIndex,
    resource: { buffer: outBuffer },
  });
  const bindGroup = gpuDevice.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });
  const encoder = gpuDevice.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  gpuDevice.queue.submit([encoder.finish()]);
  return ownedBuffer(outBuffer);
};
