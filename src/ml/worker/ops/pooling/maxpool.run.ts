import {
  makeMaxPool2dBackwardShader,
  makeMaxPool2dShader,
} from "./maxpool.shader";
import {
  borrowedBuffer,
  readGpuBufferToArray,
  releaseOwnedBuffer,
  runUnaryShaderToBuffer,
  uploadToOwnedBuffer,
  type GpuBufferRef,
} from "../../runtime/bufferKernels";
import { getGpuDevice } from "../../runtime/deviceState";
import { BUFFER_USAGE_UNIFORM_COPY_DST } from "../../runtime/gpuFlags";

const getMaxPoolDims = (
  shape: readonly [number, number, number, number],
): { channels: number; inHeight: number; inWidth: number; outHeight: number; outWidth: number } => {
  if (shape[0] !== 1) throw new Error("maxpool2d currently expects batch size 1.");
  const channels: number = shape[1];
  const inHeight: number = shape[2];
  const inWidth: number = shape[3];
  const outHeight: number = Math.floor(inHeight / 2);
  const outWidth: number = Math.floor(inWidth / 2);
  return { channels, inHeight, inWidth, outHeight, outWidth };
};

export const runMaxPool2dForwardBuffer = async (
  input: GpuBufferRef,
  shape: readonly [number, number, number, number],
): Promise<GpuBufferRef> => {
  const { channels, inHeight, inWidth, outHeight, outWidth } = getMaxPoolDims(shape);
  const gpuDevice = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const outCount: number = channels * outHeight * outWidth;
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({
    size: 5 * 4,
    usage: BUFFER_USAGE_UNIFORM_COPY_DST,
  });
  gpuDevice.queue.writeBuffer(
    uniformBuffer,
    0,
    new Uint32Array([channels, inHeight, inWidth, outHeight, outWidth]),
  );
  return borrowedBuffer(
    runUnaryShaderToBuffer(
      gpuDevice,
      makeMaxPool2dShader(outCount),
      input.buffer,
      outCount,
      [{ binding: 1, resource: { buffer: uniformBuffer } }],
    ),
  );
};

export const runMaxPool2dForward = async (
  _gpuDevice: GPUDevice | null,
  _runUnaryShader: (
    code: string,
    input: Float32Array,
    outCount: number,
    extraEntries?: GPUBindGroupEntry[],
  ) => Promise<Float32Array>,
  input: Float32Array,
  shape: readonly [number, number, number, number],
  _BUFFER_USAGE_UNIFORM_COPY_DST: number,
): Promise<Float32Array> => {
  const inputBuffer = uploadToOwnedBuffer(getGpuDevice(), input);
  const outputBuffer = await runMaxPool2dForwardBuffer(inputBuffer, shape);
  const { channels, outHeight, outWidth } = getMaxPoolDims(shape);
  const outCount = channels * outHeight * outWidth;
  const output = await readGpuBufferToArray(getGpuDevice(), outputBuffer.buffer, outCount);
  releaseOwnedBuffer(inputBuffer);
  releaseOwnedBuffer(outputBuffer);
  return output;
};

export const runMaxPool2dBackwardBuffer = async (
  input: GpuBufferRef,
  shape: readonly [number, number, number, number],
  gradOut: GpuBufferRef,
): Promise<GpuBufferRef> => {
  const { channels, inHeight, inWidth, outHeight, outWidth } = getMaxPoolDims(shape);
  const gpuDevice = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({
    size: 8 * 4,
    usage: BUFFER_USAGE_UNIFORM_COPY_DST,
  });
  gpuDevice.queue.writeBuffer(
    uniformBuffer,
    0,
    new Uint32Array([channels, inHeight, inWidth, outHeight, outWidth, 0, 0, 0]),
  );
  return borrowedBuffer(
    runUnaryShaderToBuffer(
      gpuDevice,
      makeMaxPool2dBackwardShader(shape[1] * shape[2] * shape[3]),
      input.buffer,
      shape[1] * shape[2] * shape[3],
      [
        { binding: 1, resource: { buffer: gradOut.buffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    ),
  );
};

export const runMaxPool2dBackward = async (
  gpuDevice: GPUDevice | null,
  _runUnaryShader: (
    code: string,
    input: Float32Array,
    outCount: number,
    extraEntries?: GPUBindGroupEntry[],
  ) => Promise<Float32Array>,
  input: Float32Array,
  shape: readonly [number, number, number, number],
  gradOut: Float32Array,
  _BUFFER_USAGE_STORAGE_COPY_DST: number,
  _BUFFER_USAGE_UNIFORM_COPY_DST: number,
): Promise<Float32Array> => {
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const inputBuffer = uploadToOwnedBuffer(gpuDevice, input);
  const gradOutBuffer = uploadToOwnedBuffer(gpuDevice, gradOut);
  const outputBuffer = await runMaxPool2dBackwardBuffer(
    inputBuffer,
    shape,
    gradOutBuffer,
  );
  const output = await readGpuBufferToArray(gpuDevice, outputBuffer.buffer, input.length);
  releaseOwnedBuffer(inputBuffer);
  releaseOwnedBuffer(gradOutBuffer);
  releaseOwnedBuffer(outputBuffer);
  return output;
};
