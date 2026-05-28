import {
  type GramKernelVariant,
  runGramBackwardBuffer,
  runGramMatrixBuffer,
} from "../gram/gram.run";
import { runContentLossBackwardBuffer } from "./contentLoss.run";
import { makeReduceSumShader } from "./mse.shader";
import {
  makeStyleLossTermsShader,
  makeStyleLossTermsSymmetricShader,
} from "./styleLoss.shader";
import {
  ownedBuffer,
  readGpuBufferToArray,
  releaseOwnedBuffer,
  uploadToOwnedBuffer,
  type GpuBufferRef,
} from "../../runtime/bufferKernels";
import { getOrCreateComputePipeline } from "../../runtime/computePipelineCache";
import { getGpuDevice } from "../../runtime/deviceState";
import {
  BUFFER_USAGE_STORAGE_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_SRC,
  BUFFER_USAGE_UNIFORM_COPY_DST,
} from "../../runtime/gpuFlags";

export const runStyleLossTermsFromTargetGramBuffer = async (
  acquireReusableBuffer: (size: number, usage: number) => GPUBuffer,
  input: GpuBufferRef,
  shape: readonly [number, number, number, number],
  targetGram: GpuBufferRef,
  gramKernel: GramKernelVariant = "scalar",
): Promise<{ inputGram: GpuBufferRef; styleSse: GpuBufferRef }> => {
  const gpuDevice: GPUDevice | null = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  if (shape[0] !== 1) throw new Error("style-loss terms expects batch size 1.");

  const channels = shape[1];
  const spatial = shape[2] * shape[3];
  const gramCount = channels * channels;
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({
    size: 2 * Uint32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_UNIFORM_COPY_DST,
  });
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([channels, spatial]));
  const pairCount = (channels * (channels + 1)) / 2;
  const termsElementCount =
    gramKernel === "symmetric-parallel-dot" ? pairCount : gramCount;
  const workgroupCount = Math.ceil(termsElementCount / 64);

  const inputGramBuffer = gpuDevice.createBuffer({
    size: gramCount * Float32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_STORAGE_COPY_SRC,
  });
  let currentBuffer = acquireReusableBuffer(
    workgroupCount * Float32Array.BYTES_PER_ELEMENT,
    BUFFER_USAGE_STORAGE_COPY_SRC,
  );

  const termsPipeline = getOrCreateComputePipeline(
    gpuDevice,
    `style-loss-terms:${gramKernel}:${gramCount}`,
    gramKernel === "symmetric-parallel-dot"
      ? makeStyleLossTermsSymmetricShader(channels, pairCount)
      : makeStyleLossTermsShader(gramCount),
  );
  const termsBindGroup = gpuDevice.createBindGroup({
    layout: termsPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.buffer } },
      { binding: 1, resource: { buffer: targetGram.buffer } },
      { binding: 2, resource: { buffer: uniformBuffer } },
      { binding: 3, resource: { buffer: inputGramBuffer } },
      { binding: 4, resource: { buffer: currentBuffer } },
    ],
  });
  const termsEncoder = gpuDevice.createCommandEncoder();
  const termsPass = termsEncoder.beginComputePass();
  termsPass.setPipeline(termsPipeline);
  termsPass.setBindGroup(0, termsBindGroup);
  termsPass.dispatchWorkgroups(workgroupCount);
  termsPass.end();
  gpuDevice.queue.submit([termsEncoder.finish()]);

  let currentCount = workgroupCount;
  while (currentCount > 1) {
    const nextCount = Math.ceil(currentCount / 64);
    const nextBuffer = acquireReusableBuffer(
      nextCount * Float32Array.BYTES_PER_ELEMENT,
      BUFFER_USAGE_STORAGE_COPY_SRC,
    );
    const reducePipeline = getOrCreateComputePipeline(
      gpuDevice,
      `style-loss-reduce:${currentCount}`,
      makeReduceSumShader(currentCount),
    );
    const reduceBindGroup = gpuDevice.createBindGroup({
      layout: reducePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: currentBuffer } },
        { binding: 1, resource: { buffer: nextBuffer } },
      ],
    });
    const reduceEncoder = gpuDevice.createCommandEncoder();
    const reducePass = reduceEncoder.beginComputePass();
    reducePass.setPipeline(reducePipeline);
    reducePass.setBindGroup(0, reduceBindGroup);
    reducePass.dispatchWorkgroups(nextCount);
    reducePass.end();
    gpuDevice.queue.submit([reduceEncoder.finish()]);
    currentBuffer = nextBuffer;
    currentCount = nextCount;
  }

  const styleSseBuffer = gpuDevice.createBuffer({
    size: Float32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_STORAGE_COPY_SRC | BUFFER_USAGE_STORAGE_COPY_DST,
  });
  const copyEncoder = gpuDevice.createCommandEncoder();
  copyEncoder.copyBufferToBuffer(
    currentBuffer,
    0,
    styleSseBuffer,
    0,
    Float32Array.BYTES_PER_ELEMENT,
  );
  gpuDevice.queue.submit([copyEncoder.finish()]);

  return {
    inputGram: ownedBuffer(inputGramBuffer),
    styleSse: ownedBuffer(styleSseBuffer),
  };
};

export const runStyleLossBackwardFromTargetGramBuffer = async (
  input: GpuBufferRef,
  shape: readonly [number, number, number, number],
  targetGram: GpuBufferRef,
  gramKernel: GramKernelVariant = "scalar",
): Promise<GpuBufferRef> => {
  const inputGram = await runGramMatrixBuffer(input, shape, gramKernel);
  const gradInBuffer = await runStyleLossBackwardFromInputGramBuffer(
    input,
    shape,
    inputGram,
    targetGram,
  );
  releaseOwnedBuffer(inputGram);
  return gradInBuffer;
};

export const runStyleLossBackwardFromInputGramBuffer = async (
  input: GpuBufferRef,
  shape: readonly [number, number, number, number],
  inputGram: GpuBufferRef,
  targetGram: GpuBufferRef,
): Promise<GpuBufferRef> => {
  const gpuDevice: GPUDevice | null = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const gramGrad = await runContentLossBackwardBuffer(
    inputGram,
    targetGram,
    shape[1] * shape[1],
  );
  const gradInBuffer = await runGramBackwardBuffer(input, shape, gramGrad);
  releaseOwnedBuffer(gramGrad);
  return gradInBuffer;
};

export const runStyleLossBackwardFusedFromGramDiffBuffer = async (
  input: GpuBufferRef,
  shape: readonly [number, number, number, number],
  inputGram: GpuBufferRef,
  targetGram: GpuBufferRef,
): Promise<GpuBufferRef> => {
  const gpuDevice: GPUDevice | null = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const channels = shape[1];
  const spatial = shape[2] * shape[3];
  const count = channels * spatial;
  const uniformBuffer = gpuDevice.createBuffer({
    size: 4 * Uint32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_UNIFORM_COPY_DST,
  });
  gpuDevice.queue.writeBuffer(
    uniformBuffer,
    0,
    new Uint32Array([channels, spatial, shape[0] * channels * spatial, 0]),
  );

  const pipeline = getOrCreateComputePipeline(
    gpuDevice,
    `style-loss-backward-fused:${channels}:${spatial}`,
    `
struct GramBackwardUniforms {
  channels: u32,
  spatial: u32,
  norm: u32,
  _pad: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> inputGramValues: array<f32>;
@group(0) @binding(2) var<storage, read> targetGramValues: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: GramBackwardUniforms;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${count}u) { return; }
  let a = i / uniforms.spatial;
  let p = i % uniforms.spatial;
  var sum: f32 = 0.0;
  for (var b: u32 = 0u; b < uniforms.channels; b = b + 1u) {
    let diffAB = inputGramValues[a * uniforms.channels + b] - targetGramValues[a * uniforms.channels + b];
    let diffBA = inputGramValues[b * uniforms.channels + a] - targetGramValues[b * uniforms.channels + a];
    let inputBP = inputValues[b * uniforms.spatial + p];
    sum = sum + ((diffAB + diffBA) * inputBP);
  }
  let gramCount = f32(uniforms.channels * uniforms.channels);
  out[i] = (2.0 * sum) / (gramCount * f32(uniforms.norm));
}`,
  );
  const outBuffer = gpuDevice.createBuffer({
    size: count * Float32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_STORAGE_COPY_SRC,
  });
  const bindGroup = gpuDevice.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.buffer } },
      { binding: 1, resource: { buffer: inputGram.buffer } },
      { binding: 2, resource: { buffer: targetGram.buffer } },
      { binding: 3, resource: { buffer: uniformBuffer } },
      { binding: 4, resource: { buffer: outBuffer } },
    ],
  });
  const encoder = gpuDevice.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  gpuDevice.queue.submit([encoder.finish()]);
  return ownedBuffer(outBuffer);
};

export const runStyleLossBackward = async (
  gpuDevice: GPUDevice | null,
  _runUnaryShader: (
    code: string,
    input: Float32Array,
    outCount: number,
    extraEntries?: GPUBindGroupEntry[],
  ) => Promise<Float32Array>,
  input: Float32Array,
  shape: readonly [number, number, number, number],
  target: Float32Array,
  _BUFFER_USAGE_STORAGE_COPY_DST: number,
  _BUFFER_USAGE_UNIFORM_COPY_DST: number,
): Promise<Float32Array> => {
  const inputBuffer = uploadToOwnedBuffer(gpuDevice, input);
  const targetBuffer = uploadToOwnedBuffer(gpuDevice, target);
  const inputGram = await runGramMatrixBuffer(inputBuffer, shape);
  const targetGram = await runGramMatrixBuffer(targetBuffer, shape);
  const gramGrad = await runContentLossBackwardBuffer(
    inputGram,
    targetGram,
    shape[1] * shape[1],
  );
  const gradInBuffer = await runGramBackwardBuffer(inputBuffer, shape, gramGrad);
  const out = await readGpuBufferToArray(gpuDevice, gradInBuffer.buffer, input.length);
  releaseOwnedBuffer(inputBuffer);
  releaseOwnedBuffer(targetBuffer);
  releaseOwnedBuffer(inputGram);
  releaseOwnedBuffer(targetGram);
  releaseOwnedBuffer(gramGrad);
  releaseOwnedBuffer(gradInBuffer);
  return out;
};
