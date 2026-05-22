import {
  makeConv2dBackwardInputShader,
  makeConv2dReluShader,
  makeConv2dShader,
} from "./conv2d.shader";
import {
  borrowedBuffer,
  readGpuBufferToArray,
  releaseOwnedBuffer,
  runUnaryShaderToBuffer,
  uploadToOwnedBuffer,
  type GpuBufferRef,
} from "../../runtime/bufferKernels";
import { getGpuDevice } from "../../runtime/deviceState";
import {
  BUFFER_USAGE_STORAGE_COPY_DST,
  BUFFER_USAGE_UNIFORM_COPY_DST,
} from "../../runtime/gpuFlags";

const validateConv2dParams = (
  inputShape: readonly [number, number, number, number],
  weightShape: readonly [number, number, number, number],
): { inChannels: number; outChannels: number; height: number; width: number } => {
  const [batch, inChannels, height, width] = inputShape;
  const [outChannels, weightInChannels, kernelHeight, kernelWidth] = weightShape;
  if (batch !== 1 || kernelHeight !== 3 || kernelWidth !== 3 || inChannels !== weightInChannels) {
    throw new Error("conv2d only supports VGG-compatible params.");
  }
  return { inChannels, outChannels, height, width };
};

export const runConv2dForwardBuffer = async (
  input: GpuBufferRef,
  inputShape: readonly [number, number, number, number],
  weight: Float32Array,
  weightShape: readonly [number, number, number, number],
  bias: ArrayLike<number>,
): Promise<GpuBufferRef> => {
  const { inChannels, outChannels, height, width } = validateConv2dParams(inputShape, weightShape);
  if (bias.length !== outChannels) throw new Error("conv2d-forward only supports VGG-compatible params.");
  const gpuDevice: GPUDevice | null = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const outCount: number = outChannels * height * width;
  const weightBuffer: GPUBuffer = gpuDevice.createBuffer({ size: weight.byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST });
  const biasBuffer: GPUBuffer = gpuDevice.createBuffer({ size: outChannels * 4, usage: BUFFER_USAGE_STORAGE_COPY_DST });
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 4 * 4, usage: BUFFER_USAGE_UNIFORM_COPY_DST });
  gpuDevice.queue.writeBuffer(weightBuffer, 0, weight);
  gpuDevice.queue.writeBuffer(biasBuffer, 0, new Float32Array(bias));
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([inChannels, outChannels, height, width]));
  return borrowedBuffer(runUnaryShaderToBuffer(gpuDevice, makeConv2dShader(outCount), input.buffer, outCount, [
    { binding: 1, resource: { buffer: weightBuffer } },
    { binding: 2, resource: { buffer: biasBuffer } },
    { binding: 3, resource: { buffer: uniformBuffer } },
  ]));
};

export const runConv2dForward = async (
  _gpuDevice: GPUDevice | null,
  _runUnaryShader: (code: string, input: Float32Array, outCount: number, extraEntries?: GPUBindGroupEntry[]) => Promise<Float32Array>,
  input: Float32Array,
  inputShape: readonly [number, number, number, number],
  weight: Float32Array,
  weightShape: readonly [number, number, number, number],
  bias: ArrayLike<number>,
  _BUFFER_USAGE_STORAGE_COPY_DST: number,
  _BUFFER_USAGE_UNIFORM_COPY_DST: number,
): Promise<Float32Array> => {
  const inputBuffer = uploadToOwnedBuffer(getGpuDevice(), input);
  const outCount = weightShape[0] * inputShape[2] * inputShape[3];
  const outputBuffer = await runConv2dForwardBuffer(inputBuffer, inputShape, weight, weightShape, bias);
  const output = await readGpuBufferToArray(getGpuDevice(), outputBuffer.buffer, outCount);
  releaseOwnedBuffer(inputBuffer);
  releaseOwnedBuffer(outputBuffer);
  return output;
};

export const runConv2dReluForward = async (
  gpuDevice: GPUDevice | null,
  runUnaryShader: (code: string, input: Float32Array, outCount: number, extraEntries?: GPUBindGroupEntry[]) => Promise<Float32Array>,
  input: Float32Array,
  inputShape: readonly [number, number, number, number],
  weight: Float32Array,
  weightShape: readonly [number, number, number, number],
  bias: ArrayLike<number>,
  BUFFER_USAGE_STORAGE_COPY_DST_VALUE: number,
  BUFFER_USAGE_UNIFORM_COPY_DST_VALUE: number,
): Promise<Float32Array> => {
  const outChannels = weightShape[0];
  const height = inputShape[2];
  const width = inputShape[3];
  const outputCount = outChannels * height * width;
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const weightBuffer: GPUBuffer = gpuDevice.createBuffer({ size: weight.byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST_VALUE });
  const biasBuffer: GPUBuffer = gpuDevice.createBuffer({ size: new Float32Array(bias).byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST_VALUE });
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 4 * 4, usage: BUFFER_USAGE_UNIFORM_COPY_DST_VALUE });
  gpuDevice.queue.writeBuffer(weightBuffer, 0, weight);
  gpuDevice.queue.writeBuffer(biasBuffer, 0, new Float32Array(bias));
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([inputShape[1], outChannels, height, width]));
  return runUnaryShader(makeConv2dReluShader(outputCount), input, outputCount, [
    { binding: 1, resource: { buffer: weightBuffer } },
    { binding: 2, resource: { buffer: biasBuffer } },
    { binding: 3, resource: { buffer: uniformBuffer } },
  ]);
};

export const runConv2dBackwardInputBuffer = async (
  gradOut: GpuBufferRef,
  inputShape: readonly [number, number, number, number],
  weight: Float32Array,
  weightShape: readonly [number, number, number, number],
): Promise<GpuBufferRef> => {
  const { inChannels, outChannels, height, width } = validateConv2dParams(inputShape, weightShape);
  const gpuDevice: GPUDevice | null = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const outCount = inChannels * height * width;
  const weightBuffer: GPUBuffer = gpuDevice.createBuffer({ size: weight.byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST });
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 4 * 4, usage: BUFFER_USAGE_UNIFORM_COPY_DST });
  gpuDevice.queue.writeBuffer(weightBuffer, 0, weight);
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([inChannels, outChannels, height, width]));
  return borrowedBuffer(runUnaryShaderToBuffer(gpuDevice, makeConv2dBackwardInputShader(outCount), gradOut.buffer, outCount, [
    { binding: 1, resource: { buffer: weightBuffer } },
    { binding: 2, resource: { buffer: uniformBuffer } },
  ]));
};

export const runConv2dBackwardInput = async (
  _gpuDevice: GPUDevice | null,
  _runUnaryShader: (code: string, input: Float32Array, outCount: number, extraEntries?: GPUBindGroupEntry[]) => Promise<Float32Array>,
  inputShape: readonly [number, number, number, number],
  gradOut: Float32Array,
  weight: Float32Array,
  weightShape: readonly [number, number, number, number],
  _BUFFER_USAGE_STORAGE_COPY_DST: number,
  _BUFFER_USAGE_UNIFORM_COPY_DST: number,
): Promise<Float32Array> => {
  const gradOutBuffer = uploadToOwnedBuffer(getGpuDevice(), gradOut);
  const outputBuffer = await runConv2dBackwardInputBuffer(gradOutBuffer, inputShape, weight, weightShape);
  const outCount = inputShape[1] * inputShape[2] * inputShape[3];
  const output = await readGpuBufferToArray(getGpuDevice(), outputBuffer.buffer, outCount);
  releaseOwnedBuffer(gradOutBuffer);
  releaseOwnedBuffer(outputBuffer);
  return output;
};
