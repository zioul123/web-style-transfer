import { makeGramBackwardShader, makeGramMatrixShader } from "./gram.shader";
import {
  ownedBuffer,
  readGpuBufferToArray,
  releaseOwnedBuffer,
  runUnaryShaderToBuffer,
  uploadToOwnedBuffer,
  type GpuBufferRef,
} from "../../runtime/bufferKernels";
import { getGpuDevice } from "../../runtime/deviceState";
import { BUFFER_USAGE_UNIFORM_COPY_DST } from "../../runtime/gpuFlags";

export const runGramMatrixBuffer = async (
  input: GpuBufferRef,
  shape: readonly [number, number, number, number],
): Promise<GpuBufferRef> => {
  if (shape[0] !== 1) throw new Error("gram-matrix expects batch size 1.");
  const gpuDevice: GPUDevice | null = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const channels = shape[1];
  const spatial = shape[2] * shape[3];
  const outCount = channels * channels;
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 2 * 4, usage: BUFFER_USAGE_UNIFORM_COPY_DST });
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([channels, spatial]));
  return ownedBuffer(runUnaryShaderToBuffer(gpuDevice, makeGramMatrixShader(outCount), input.buffer, outCount, [
    { binding: 1, resource: { buffer: uniformBuffer } },
  ]));
};

export const runGramMatrix = async (
  _gpuDevice: GPUDevice | null,
  _runUnaryShader: (code: string, input: Float32Array, outCount: number, extraEntries?: GPUBindGroupEntry[]) => Promise<Float32Array>,
  input: Float32Array,
  shape: readonly [number, number, number, number],
  _BUFFER_USAGE_UNIFORM_COPY_DST: number,
): Promise<Float32Array> => {
  const inputBuffer = uploadToOwnedBuffer(getGpuDevice(), input);
  const outputBuffer = await runGramMatrixBuffer(inputBuffer, shape);
  const outCount = shape[1] * shape[1];
  const output = await readGpuBufferToArray(getGpuDevice(), outputBuffer.buffer, outCount);
  releaseOwnedBuffer(inputBuffer);
  releaseOwnedBuffer(outputBuffer);
  return output;
};

export const runGramBackwardBuffer = async (
  input: GpuBufferRef,
  shape: readonly [number, number, number, number],
  gradOut: GpuBufferRef,
): Promise<GpuBufferRef> => {
  const channels = shape[1];
  const spatial = shape[2] * shape[3];
  const norm = shape[0] * shape[1] * shape[2] * shape[3];
  const gpuDevice: GPUDevice | null = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 4 * 4, usage: BUFFER_USAGE_UNIFORM_COPY_DST });
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([channels, spatial, norm, 0]));
  return ownedBuffer(runUnaryShaderToBuffer(gpuDevice, makeGramBackwardShader(shape[1] * shape[2] * shape[3]), input.buffer, shape[1] * shape[2] * shape[3], [
    { binding: 1, resource: { buffer: gradOut.buffer } },
    { binding: 2, resource: { buffer: uniformBuffer } },
  ]));
};

export const runGramBackward = async (
  _gpuDevice: GPUDevice | null,
  _runUnaryShader: (code: string, input: Float32Array, outCount: number, extraEntries?: GPUBindGroupEntry[]) => Promise<Float32Array>,
  input: Float32Array,
  shape: readonly [number, number, number, number],
  gradOut: Float32Array,
  _BUFFER_USAGE_STORAGE_COPY_DST: number,
  _BUFFER_USAGE_UNIFORM_COPY_DST: number,
): Promise<Float32Array> => {
  const inputBuffer = uploadToOwnedBuffer(getGpuDevice(), input);
  const gradOutBuffer = uploadToOwnedBuffer(getGpuDevice(), gradOut);
  const outputBuffer = await runGramBackwardBuffer(inputBuffer, shape, gradOutBuffer);
  const output = await readGpuBufferToArray(getGpuDevice(), outputBuffer.buffer, input.length);
  releaseOwnedBuffer(inputBuffer);
  releaseOwnedBuffer(gradOutBuffer);
  releaseOwnedBuffer(outputBuffer);
  return output;
};
