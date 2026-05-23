import {
  runGramBackwardBuffer,
  runGramMatrixBuffer,
} from "../gram/gram.run";
import { runContentLossBackwardBuffer } from "./contentLoss.run";
import {
  readGpuBufferToArray,
  releaseOwnedBuffer,
  uploadToOwnedBuffer,
  type GpuBufferRef,
} from "../../runtime/bufferKernels";
import { getGpuDevice } from "../../runtime/deviceState";

export const runStyleLossBackwardFromTargetGramBuffer = async (
  input: GpuBufferRef,
  shape: readonly [number, number, number, number],
  targetGram: GpuBufferRef,
): Promise<GpuBufferRef> => {
  const gpuDevice: GPUDevice | null = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const inputGram = await runGramMatrixBuffer(input, shape);
  const gramGrad = await runContentLossBackwardBuffer(
    inputGram,
    targetGram,
    shape[1] * shape[1],
  );
  const gradInBuffer = await runGramBackwardBuffer(input, shape, gramGrad);
  releaseOwnedBuffer(inputGram);
  releaseOwnedBuffer(gramGrad);
  return gradInBuffer;
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

export const runStyleLossBackwardFromTargetGram = async (
  gpuDevice: GPUDevice | null,
  _runUnaryShader: (
    code: string,
    input: Float32Array,
    outCount: number,
    extraEntries?: GPUBindGroupEntry[],
  ) => Promise<Float32Array>,
  input: Float32Array,
  shape: readonly [number, number, number, number],
  targetGram: Float32Array,
  _BUFFER_USAGE_STORAGE_COPY_DST: number,
  _BUFFER_USAGE_UNIFORM_COPY_DST: number,
): Promise<Float32Array> => {
  const inputBuffer = uploadToOwnedBuffer(gpuDevice, input);
  const targetGramBuffer = uploadToOwnedBuffer(gpuDevice, targetGram);
  const inputGram = await runGramMatrixBuffer(inputBuffer, shape);
  const gramGrad = await runContentLossBackwardBuffer(
    inputGram,
    targetGramBuffer,
    shape[1] * shape[1],
  );
  const gradInBuffer = await runGramBackwardBuffer(inputBuffer, shape, gramGrad);
  const out = await readGpuBufferToArray(gpuDevice, gradInBuffer.buffer, input.length);
  releaseOwnedBuffer(inputBuffer);
  releaseOwnedBuffer(targetGramBuffer);
  releaseOwnedBuffer(inputGram);
  releaseOwnedBuffer(gramGrad);
  releaseOwnedBuffer(gradInBuffer);
  return out;
};
