import { makeMseBackwardShader } from "./mse.shader";
import {
  ownedBuffer,
  readGpuBufferToArray,
  releaseOwnedBuffer,
  runUnaryShaderToBuffer,
  uploadToOwnedBuffer,
  type GpuBufferRef,
} from "../../runtime/bufferKernels";
import { getGpuDevice } from "../../runtime/deviceState";
export const runContentLossBackwardBuffer = async (
  input: GpuBufferRef,
  target: GpuBufferRef,
  count: number,
): Promise<GpuBufferRef> => {
  const gpuDevice = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  return ownedBuffer(
    runUnaryShaderToBuffer(gpuDevice, makeMseBackwardShader(count), input.buffer, count, [
      { binding: 1, resource: { buffer: target.buffer } },
    ]),
  );
};

export const runContentLossBackward = async (
  gpuDevice: GPUDevice | null,
  _runUnaryShader: (
    code: string,
    input: Float32Array,
    outCount: number,
    extraEntries?: GPUBindGroupEntry[],
  ) => Promise<Float32Array>,
  input: Float32Array,
  target: Float32Array,
  _BUFFER_USAGE_STORAGE_COPY_DST_VALUE: number,
): Promise<Float32Array> => {
  if (input.length !== target.length) return new Float32Array(input.length);
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const inputBuffer = uploadToOwnedBuffer(gpuDevice, input);
  const targetBuffer = uploadToOwnedBuffer(gpuDevice, target);
  const outputBuffer = await runContentLossBackwardBuffer(inputBuffer, targetBuffer, input.length);
  const output = await readGpuBufferToArray(gpuDevice, outputBuffer.buffer, input.length);
  releaseOwnedBuffer(inputBuffer);
  releaseOwnedBuffer(targetBuffer);
  releaseOwnedBuffer(outputBuffer);
  return output;
};
