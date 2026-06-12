import type { FeatureMatrixShape } from "../../../../types";
import {
  validateFeatureMatrixLength,
  validateSurfacePoolGradOutLength,
  validateSurfacePoolParams,
} from "../../../pointcloud/metadata";
import {
  makeSurfacePoolBackwardShader,
  makeSurfacePoolForwardShader,
} from "./surfacePool.shader";
import {
  ownedBuffer,
  readGpuBufferToArray,
  releaseOwnedBuffer,
  runUnaryShaderToBuffer,
  uploadInt32StorageBuffer,
  uploadToOwnedBuffer,
  uploadUint32UniformBuffer,
  type GpuBufferRef,
} from "../../runtime/bufferKernels";
import { getGpuDevice } from "../../runtime/deviceState";

export type SurfacePoolMapBuffers = {
  readonly mappingBuffer: GPUBuffer;
  readonly inputPointCount: number;
  readonly outputPointCount: number;
  readonly validInputCount: number;
  readonly byteLength: number;
};

const createUniformBuffer = (
  gpuDevice: GPUDevice,
  inputShape: FeatureMatrixShape,
  outputPointCount: number,
  validInputCount: number,
): GPUBuffer =>
  uploadUint32UniformBuffer(gpuDevice, [
    inputShape.pointCount,
    outputPointCount,
    inputShape.channelCount,
    validInputCount,
  ]);

export const createSurfacePoolMapBuffers = (
  gpuDevice: GPUDevice,
  inputPointCount: number,
  outputPointCount: number,
  mapping: Int32Array,
): SurfacePoolMapBuffers => {
  const { validInputCount } = validateSurfacePoolParams(
    { pointCount: inputPointCount, channelCount: 1 },
    { inputPointCount, outputPointCount, mapping },
  );
  return {
    mappingBuffer: uploadInt32StorageBuffer(gpuDevice, mapping),
    inputPointCount,
    outputPointCount,
    validInputCount,
    byteLength: mapping.byteLength,
  };
};

export const destroySurfacePoolMapBuffers = (
  buffers: SurfacePoolMapBuffers,
): void => {
  buffers.mappingBuffer.destroy();
};

const resolveSurfacePoolMap = (
  inputShape: FeatureMatrixShape,
  outputPointCount: number,
  mapping: Int32Array,
  buffers: SurfacePoolMapBuffers | undefined,
): {
  mappingBuffer: GPUBuffer;
  validInputCount: number;
  releaseMappingBuffer: () => void;
} => {
  if (buffers !== undefined) {
    if (
      buffers.inputPointCount !== inputShape.pointCount ||
      buffers.outputPointCount !== outputPointCount
    ) {
      throw new Error("surface-pool cached mapping shape mismatch.");
    }
    return {
      mappingBuffer: buffers.mappingBuffer,
      validInputCount: buffers.validInputCount,
      releaseMappingBuffer: () => undefined,
    };
  }

  const { validInputCount } = validateSurfacePoolParams(inputShape, {
    outputPointCount,
    mapping,
  });
  const gpuDevice = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const mappingBuffer = uploadInt32StorageBuffer(gpuDevice, mapping);
  return {
    mappingBuffer,
    validInputCount,
    releaseMappingBuffer: () => mappingBuffer.destroy(),
  };
};

export const runSurfacePoolForwardBuffer = async (
  input: GpuBufferRef,
  inputShape: FeatureMatrixShape,
  outputPointCount: number,
  mapping: Int32Array,
  buffers?: SurfacePoolMapBuffers,
): Promise<GpuBufferRef> => {
  const gpuDevice = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const outputCount = outputPointCount * inputShape.channelCount;
  const { mappingBuffer, validInputCount, releaseMappingBuffer } =
    resolveSurfacePoolMap(inputShape, outputPointCount, mapping, buffers);
  const uniformBuffer = createUniformBuffer(
    gpuDevice,
    inputShape,
    outputPointCount,
    validInputCount,
  );
  try {
    return ownedBuffer(
      runUnaryShaderToBuffer(
        gpuDevice,
        makeSurfacePoolForwardShader(outputCount),
        input.buffer,
        outputCount,
        [
          { binding: 1, resource: { buffer: mappingBuffer } },
          { binding: 2, resource: { buffer: uniformBuffer } },
        ],
      ),
    );
  } finally {
    releaseMappingBuffer();
    uniformBuffer.destroy();
  }
};

export const runSurfacePoolBackwardBuffer = async (
  input: GpuBufferRef,
  gradOut: GpuBufferRef,
  inputShape: FeatureMatrixShape,
  outputPointCount: number,
  mapping: Int32Array,
  buffers?: SurfacePoolMapBuffers,
): Promise<GpuBufferRef> => {
  const gpuDevice = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const outputCount = inputShape.pointCount * inputShape.channelCount;
  const { mappingBuffer, validInputCount, releaseMappingBuffer } =
    resolveSurfacePoolMap(inputShape, outputPointCount, mapping, buffers);
  const uniformBuffer = createUniformBuffer(
    gpuDevice,
    inputShape,
    outputPointCount,
    validInputCount,
  );
  try {
    return ownedBuffer(
      runUnaryShaderToBuffer(
        gpuDevice,
        makeSurfacePoolBackwardShader(outputCount),
        input.buffer,
        outputCount,
        [
          { binding: 1, resource: { buffer: gradOut.buffer } },
          { binding: 2, resource: { buffer: mappingBuffer } },
          { binding: 3, resource: { buffer: uniformBuffer } },
        ],
      ),
    );
  } finally {
    releaseMappingBuffer();
    uniformBuffer.destroy();
  }
};

export const runSurfacePoolForwardReadback = async (
  input: Float32Array,
  inputShape: FeatureMatrixShape,
  outputPointCount: number,
  mapping: Int32Array,
): Promise<Float32Array> => {
  validateFeatureMatrixLength(inputShape, input, "surface-pool-forward input");
  const inputBuffer = uploadToOwnedBuffer(getGpuDevice(), input);
  let outputBuffer: GpuBufferRef | null = null;
  try {
    outputBuffer = await runSurfacePoolForwardBuffer(
      inputBuffer,
      inputShape,
      outputPointCount,
      mapping,
    );
    return await readGpuBufferToArray(
      getGpuDevice(),
      outputBuffer.buffer,
      outputPointCount * inputShape.channelCount,
    );
  } finally {
    releaseOwnedBuffer(inputBuffer);
    if (outputBuffer !== null) releaseOwnedBuffer(outputBuffer);
  }
};

export const runSurfacePoolBackwardReadback = async (
  input: Float32Array,
  gradOut: Float32Array,
  inputShape: FeatureMatrixShape,
  outputPointCount: number,
  mapping: Int32Array,
): Promise<Float32Array> => {
  validateFeatureMatrixLength(inputShape, input, "surface-pool-backward input");
  validateSurfacePoolGradOutLength(
    gradOut.length,
    inputShape,
    outputPointCount,
  );
  const inputBuffer = uploadToOwnedBuffer(getGpuDevice(), input);
  const gradOutBuffer = uploadToOwnedBuffer(getGpuDevice(), gradOut);
  let outputBuffer: GpuBufferRef | null = null;
  try {
    outputBuffer = await runSurfacePoolBackwardBuffer(
      inputBuffer,
      gradOutBuffer,
      inputShape,
      outputPointCount,
      mapping,
    );
    return await readGpuBufferToArray(
      getGpuDevice(),
      outputBuffer.buffer,
      inputShape.pointCount * inputShape.channelCount,
    );
  } finally {
    releaseOwnedBuffer(inputBuffer);
    releaseOwnedBuffer(gradOutBuffer);
    if (outputBuffer !== null) releaseOwnedBuffer(outputBuffer);
  }
};
