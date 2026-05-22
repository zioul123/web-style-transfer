import { makeReluBackwardShader, makeReluShader } from "./relu.shader";
import {
  borrowedBuffer,
  readGpuBufferToArray,
  releaseOwnedBuffer,
  runUnaryShaderToBuffer,
  uploadToOwnedBuffer,
  type GpuBufferRef,
} from "../../runtime/bufferKernels";
import { getGpuDevice } from "../../runtime/deviceState";

export const runReluForwardBuffer = async (
  input: GpuBufferRef,
  count: number,
): Promise<GpuBufferRef> => {
  const outputBuffer = runUnaryShaderToBuffer(
    getGpuDevice(),
    makeReluShader(count),
    input.buffer,
    count,
  );
  return borrowedBuffer(outputBuffer);
};

export const runReluForward = async (
  _runUnaryShader: (
    code: string,
    input: Float32Array,
    outCount: number,
    extraEntries?: GPUBindGroupEntry[],
  ) => Promise<Float32Array>,
  input: Float32Array,
): Promise<Float32Array> => {
  const inputBuffer = uploadToOwnedBuffer(getGpuDevice(), input);
  const outBuffer = await runReluForwardBuffer(inputBuffer, input.length);
  const output = await readGpuBufferToArray(
    getGpuDevice(),
    outBuffer.buffer,
    input.length,
  );
  releaseOwnedBuffer(inputBuffer);
  releaseOwnedBuffer(outBuffer);
  return output;
};

export const runReluBackwardBuffer = async (
  input: GpuBufferRef,
  gradOut: GpuBufferRef,
  count: number,
): Promise<GpuBufferRef> => {
  return borrowedBuffer(
    runUnaryShaderToBuffer(
      getGpuDevice(),
      makeReluBackwardShader(count),
      input.buffer,
      count,
      [{ binding: 1, resource: { buffer: gradOut.buffer } }],
    ),
  );
};

export const runReluBackward = async (
  gpuDevice: GPUDevice | null,
  _runUnaryShader: (
    code: string,
    input: Float32Array,
    outCount: number,
    extraEntries?: GPUBindGroupEntry[],
  ) => Promise<Float32Array>,
  input: Float32Array,
  gradOut: Float32Array,
): Promise<Float32Array> => {
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const inputBuffer = uploadToOwnedBuffer(gpuDevice, input);
  const gradOutBuffer = uploadToOwnedBuffer(gpuDevice, gradOut);
  const outputBuffer = await runReluBackwardBuffer(
    inputBuffer,
    gradOutBuffer,
    input.length,
  );
  const output = await readGpuBufferToArray(gpuDevice, outputBuffer.buffer, input.length);
  releaseOwnedBuffer(inputBuffer);
  releaseOwnedBuffer(gradOutBuffer);
  releaseOwnedBuffer(outputBuffer);
  return output;
};
