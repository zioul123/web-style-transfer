import type { FeatureMatrixShape } from "../../../../types";
import {
  featureMatrixElementCount,
  validateFeatureMatrixLength,
  validatePerChannelValues,
} from "../../../pointcloud/metadata";
import {
  makePointFeatureNormalizeBackwardShader,
  makePointFeatureNormalizeForwardShader,
  makePointwiseExpBackwardShader,
  makePointwiseExpForwardShader,
} from "./featureMatrix.shader";
import {
  ownedBuffer,
  readGpuBufferToArray,
  releaseOwnedBuffer,
  runUnaryShaderToBuffer,
  uploadFloat32StorageBuffer,
  uploadToOwnedBuffer,
  uploadUint32UniformBuffer,
  type GpuBufferRef,
} from "../../runtime/bufferKernels";
import { getGpuDevice } from "../../runtime/deviceState";

const createFeatureMatrixUniformBuffer = (
  gpuDevice: GPUDevice,
  shape: FeatureMatrixShape,
): GPUBuffer =>
  uploadUint32UniformBuffer(gpuDevice, [shape.pointCount, shape.channelCount]);

export const runPointwiseExpForwardBuffer = async (
  input: GpuBufferRef,
  count: number,
): Promise<GpuBufferRef> => {
  const gpuDevice = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  return ownedBuffer(
    runUnaryShaderToBuffer(
      gpuDevice,
      makePointwiseExpForwardShader(count),
      input.buffer,
      count,
    ),
  );
};

export const runPointwiseExpBackwardBuffer = async (
  input: GpuBufferRef,
  gradOut: GpuBufferRef,
  count: number,
): Promise<GpuBufferRef> => {
  const gpuDevice = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  return ownedBuffer(
    runUnaryShaderToBuffer(
      gpuDevice,
      makePointwiseExpBackwardShader(count),
      input.buffer,
      count,
      [{ binding: 1, resource: { buffer: gradOut.buffer } }],
    ),
  );
};

export const runPointFeatureNormalizeForwardBuffer = async (
  input: GpuBufferRef,
  shape: FeatureMatrixShape,
  mean: readonly number[],
  std: readonly number[],
): Promise<GpuBufferRef> => {
  validatePerChannelValues(shape, mean, "Mean");
  validatePerChannelValues(shape, std, "Std");
  const gpuDevice = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const count = featureMatrixElementCount(shape);
  const meanBuffer = uploadFloat32StorageBuffer(gpuDevice, mean);
  const stdBuffer = uploadFloat32StorageBuffer(gpuDevice, std);
  const uniformBuffer = createFeatureMatrixUniformBuffer(gpuDevice, shape);
  try {
    return ownedBuffer(
      runUnaryShaderToBuffer(
        gpuDevice,
        makePointFeatureNormalizeForwardShader(count),
        input.buffer,
        count,
        [
          { binding: 1, resource: { buffer: meanBuffer } },
          { binding: 2, resource: { buffer: stdBuffer } },
          { binding: 3, resource: { buffer: uniformBuffer } },
        ],
      ),
    );
  } finally {
    meanBuffer.destroy();
    stdBuffer.destroy();
    uniformBuffer.destroy();
  }
};

export const runPointFeatureNormalizeBackwardBuffer = async (
  gradOut: GpuBufferRef,
  shape: FeatureMatrixShape,
  std: readonly number[],
): Promise<GpuBufferRef> => {
  validatePerChannelValues(shape, std, "Std");
  const gpuDevice = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const count = featureMatrixElementCount(shape);
  const stdBuffer = uploadFloat32StorageBuffer(gpuDevice, std);
  const uniformBuffer = createFeatureMatrixUniformBuffer(gpuDevice, shape);
  try {
    return ownedBuffer(
      runUnaryShaderToBuffer(
        gpuDevice,
        makePointFeatureNormalizeBackwardShader(count),
        gradOut.buffer,
        count,
        [
          { binding: 1, resource: { buffer: stdBuffer } },
          { binding: 2, resource: { buffer: uniformBuffer } },
        ],
      ),
    );
  } finally {
    stdBuffer.destroy();
    uniformBuffer.destroy();
  }
};

export const runPointwiseExpForwardReadback = async (
  input: Float32Array,
): Promise<Float32Array> => {
  const inputBuffer = uploadToOwnedBuffer(getGpuDevice(), input);
  let outputBuffer: GpuBufferRef | null = null;
  try {
    outputBuffer = await runPointwiseExpForwardBuffer(
      inputBuffer,
      input.length,
    );
    return await readGpuBufferToArray(
      getGpuDevice(),
      outputBuffer.buffer,
      input.length,
    );
  } finally {
    releaseOwnedBuffer(inputBuffer);
    if (outputBuffer !== null) releaseOwnedBuffer(outputBuffer);
  }
};

export const runPointwiseExpBackwardReadback = async (
  input: Float32Array,
  gradOut: Float32Array,
): Promise<Float32Array> => {
  if (input.length !== gradOut.length) {
    throw new Error("exp-backward input and gradOut lengths must match.");
  }
  const inputBuffer = uploadToOwnedBuffer(getGpuDevice(), input);
  const gradOutBuffer = uploadToOwnedBuffer(getGpuDevice(), gradOut);
  let outputBuffer: GpuBufferRef | null = null;
  try {
    outputBuffer = await runPointwiseExpBackwardBuffer(
      inputBuffer,
      gradOutBuffer,
      input.length,
    );
    return await readGpuBufferToArray(
      getGpuDevice(),
      outputBuffer.buffer,
      input.length,
    );
  } finally {
    releaseOwnedBuffer(inputBuffer);
    releaseOwnedBuffer(gradOutBuffer);
    if (outputBuffer !== null) releaseOwnedBuffer(outputBuffer);
  }
};

export const runPointFeatureNormalizeForwardReadback = async (
  input: Float32Array,
  shape: FeatureMatrixShape,
  mean: readonly number[],
  std: readonly number[],
): Promise<Float32Array> => {
  validateFeatureMatrixLength(shape, input, "point-feature-normalize-forward");
  const inputBuffer = uploadToOwnedBuffer(getGpuDevice(), input);
  let outputBuffer: GpuBufferRef | null = null;
  try {
    outputBuffer = await runPointFeatureNormalizeForwardBuffer(
      inputBuffer,
      shape,
      mean,
      std,
    );
    return await readGpuBufferToArray(
      getGpuDevice(),
      outputBuffer.buffer,
      input.length,
    );
  } finally {
    releaseOwnedBuffer(inputBuffer);
    if (outputBuffer !== null) releaseOwnedBuffer(outputBuffer);
  }
};

export const runPointFeatureNormalizeBackwardReadback = async (
  gradOut: Float32Array,
  shape: FeatureMatrixShape,
  std: readonly number[],
): Promise<Float32Array> => {
  validateFeatureMatrixLength(
    shape,
    gradOut,
    "point-feature-normalize-backward",
  );
  const gradOutBuffer = uploadToOwnedBuffer(getGpuDevice(), gradOut);
  let outputBuffer: GpuBufferRef | null = null;
  try {
    outputBuffer = await runPointFeatureNormalizeBackwardBuffer(
      gradOutBuffer,
      shape,
      std,
    );
    return await readGpuBufferToArray(
      getGpuDevice(),
      outputBuffer.buffer,
      gradOut.length,
    );
  } finally {
    releaseOwnedBuffer(gradOutBuffer);
    if (outputBuffer !== null) releaseOwnedBuffer(outputBuffer);
  }
};
