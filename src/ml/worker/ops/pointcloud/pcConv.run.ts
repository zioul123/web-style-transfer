import type { FeatureMatrixShape } from "../../../../types";
import {
  POINT_CLOUD_CONV_DEFAULT_INDEX_MAP,
  validateFeatureMatrixLength,
  validatePointCloudConvBiasLength,
  validatePointCloudConvGradOutLength,
  validatePointCloudConvParams,
  validatePointCloudConvWeightLength,
} from "../../../pointcloud/metadata";
import {
  makePointCloudConvBackwardProjectShader,
  makePointCloudConvBackwardFeaturesShader,
  makePointCloudConvForwardShader,
  makePointCloudConvInterpolateShader,
  makePointCloudConvProjectShader,
  makePointCloudConvReverseGatherShader,
  makePointCloudConvZeroRangeShader,
} from "./pcConv.shader";
import {
  ownedBuffer,
  readGpuBufferToArray,
  releaseOwnedBuffer,
  runUnaryShaderToBuffer,
  uploadFloat32StorageBuffer,
  uploadToOwnedBuffer,
  uploadUint32StorageBuffer,
  uploadUint32UniformBuffer,
  type GpuBufferRef,
} from "../../runtime/bufferKernels";

const DEFAULT_PC_CONV_MAX_INTERMEDIATE_BYTES = 32 * 1024 * 1024;
import { getOrCreateComputePipeline } from "../../runtime/computePipelineCache";
import { getGpuDevice } from "../../runtime/deviceState";
import {
  BUFFER_USAGE_STORAGE_COPY_SRC,
  BUFFER_USAGE_UNIFORM_COPY_DST,
} from "../../runtime/gpuFlags";

export type PointCloudConvForwardKernelVariant =
  | "scalar"
  | "interpolate-project";

export type PointCloudConvBackwardFeaturesKernelVariant =
  | "scalar"
  | "reverse-adjacency";

const pcConvWorkgroupSize = 64;

export type PointCloudConvKnnBuffers = {
  readonly inputPointCount: number;
  readonly sampleCount: number;
  readonly neighborCount: number;
  readonly entryCount: number;
  readonly knnIndexBuffer: GPUBuffer;
  readonly knnWeightBuffer: GPUBuffer;
  readonly reverseOffsetBuffer: GPUBuffer;
  readonly reverseSampleKernelBuffer: GPUBuffer;
  readonly reverseWeightBuffer: GPUBuffer;
  readonly forwardKnnBytes: number;
  readonly reverseAdjacencyBytes: number;
};

export type PointCloudConvStaticBuffers = {
  readonly weightBuffer: GPUBuffer;
  readonly biasBuffer: GPUBuffer;
  readonly kernelIndexMapBuffer: GPUBuffer;
  readonly byteLength: number;
};

const validateConvParams = (
  inputShape: FeatureMatrixShape,
  sampleCount: number,
  neighborCount: number,
  knnIndices: Uint32Array,
  knnWeights: Float32Array,
  weightLength: number,
  weightShape: readonly [number, number, number, number],
  biasOrGradOutLength: number,
  isBackward: boolean,
  kernelIndexMap: readonly number[],
): { outputChannels: number } => {
  const { outputChannels } = validatePointCloudConvParams(
    inputShape,
    { sampleCount, neighborCount, indices: knnIndices, weights: knnWeights },
    weightShape,
    kernelIndexMap,
  );
  validatePointCloudConvWeightLength(weightLength, weightShape);
  if (isBackward) {
    validatePointCloudConvGradOutLength(
      biasOrGradOutLength,
      sampleCount,
      outputChannels,
    );
  } else {
    validatePointCloudConvBiasLength(biasOrGradOutLength, outputChannels);
  }
  return { outputChannels };
};

const assertSupportedForwardKernelVariant = (
  kernelVariant: PointCloudConvForwardKernelVariant,
): void => {
  if (kernelVariant !== "scalar" && kernelVariant !== "interpolate-project") {
    throw new Error(`Unsupported pcConvForwardKernel: ${kernelVariant}.`);
  }
};

const assertSupportedBackwardFeaturesKernelVariant = (
  kernelVariant: PointCloudConvBackwardFeaturesKernelVariant,
): void => {
  if (kernelVariant !== "scalar" && kernelVariant !== "reverse-adjacency") {
    throw new Error(
      `Unsupported pcConvBackwardFeaturesKernel: ${kernelVariant}.`,
    );
  }
};

const normalizeMaxIntermediateBytes = (value: number | undefined): number => {
  if (value === undefined) return DEFAULT_PC_CONV_MAX_INTERMEDIATE_BYTES;
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error("pcConvMaxIntermediateBytes must be a positive number.");
  }
  return Math.floor(value);
};

const intermediateBatchSamples = (
  gpuDevice: GPUDevice,
  sampleCount: number,
  inputChannels: number,
  dispatchElementsPerSample: number,
  maxIntermediateBytes: number | undefined,
): { batchSamples: number; intermediateBytes: number } => {
  const maxBytes = normalizeMaxIntermediateBytes(maxIntermediateBytes);
  const bytesPerSample = 9 * inputChannels * Float32Array.BYTES_PER_ELEMENT;
  const memoryBatchSamples = Math.floor(maxBytes / bytesPerSample);
  const dispatchBatchSamples = Math.floor(
    maxDispatchElementCount(gpuDevice) / Math.max(1, dispatchElementsPerSample),
  );
  const batchSamples = Math.min(memoryBatchSamples, dispatchBatchSamples);
  if (batchSamples < 1) {
    throw new Error(
      memoryBatchSamples < 1
        ? "pcConvMaxIntermediateBytes is too small for one point-cloud convolution sample."
        : "Point-cloud convolution sample is too large for the WebGPU dispatch limit.",
    );
  }
  const clamped = Math.min(sampleCount, batchSamples);
  return {
    batchSamples: clamped,
    intermediateBytes: clamped * bytesPerSample,
  };
};

const createUniformBuffer = (
  gpuDevice: GPUDevice,
  inputShape: FeatureMatrixShape,
  sampleCount: number,
  neighborCount: number,
  outputChannels: number,
): GPUBuffer =>
  uploadUint32UniformBuffer(gpuDevice, [
    inputShape.pointCount,
    inputShape.channelCount,
    sampleCount,
    neighborCount,
    outputChannels,
  ]);

const createChunkUniformBuffer = (gpuDevice: GPUDevice): GPUBuffer =>
  gpuDevice.createBuffer({
    size: 8 * Uint32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_UNIFORM_COPY_DST,
  });

const createRangeUniformBuffer = (gpuDevice: GPUDevice): GPUBuffer =>
  gpuDevice.createBuffer({
    size: 4 * Uint32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_UNIFORM_COPY_DST,
  });

const maxDispatchWorkgroupCount = (gpuDevice: GPUDevice): number => {
  const limit = gpuDevice.limits.maxComputeWorkgroupsPerDimension;
  if (Number.isFinite(limit) && limit > 0) return Math.floor(limit);
  return 65_535;
};

const maxDispatchElementCount = (gpuDevice: GPUDevice): number =>
  maxDispatchWorkgroupCount(gpuDevice) * pcConvWorkgroupSize;

const writeChunkUniforms = (
  gpuDevice: GPUDevice,
  uniformBuffer: GPUBuffer,
  inputShape: FeatureMatrixShape,
  sampleCount: number,
  neighborCount: number,
  outputChannels: number,
  batchStart: number,
  batchSampleCount: number,
  outputStart = 0,
): void => {
  gpuDevice.queue.writeBuffer(
    uniformBuffer,
    0,
    new Uint32Array([
      inputShape.pointCount,
      inputShape.channelCount,
      sampleCount,
      neighborCount,
      outputChannels,
      batchStart,
      batchSampleCount,
      outputStart,
    ]),
  );
};

const writeRangeUniforms = (
  gpuDevice: GPUDevice,
  uniformBuffer: GPUBuffer,
  start: number,
  count: number,
): void => {
  gpuDevice.queue.writeBuffer(
    uniformBuffer,
    0,
    new Uint32Array([start, count, 0, 0]),
  );
};

export const createPointCloudConvKnnBuffers = (
  gpuDevice: GPUDevice,
  inputPointCount: number,
  sampleCount: number,
  neighborCount: number,
  knnIndices: Uint32Array,
  knnWeights: Float32Array,
): PointCloudConvKnnBuffers => {
  const entryCount = sampleCount * 9 * neighborCount;
  if (knnIndices.length !== entryCount || knnWeights.length !== entryCount) {
    throw new Error("pc-conv KNN metadata shape mismatch.");
  }

  const counts = new Uint32Array(inputPointCount);
  for (let entryIndex = 0; entryIndex < entryCount; entryIndex += 1) {
    counts[knnIndices[entryIndex]] += 1;
  }

  const reverseOffsets = new Uint32Array(inputPointCount + 1);
  for (let point = 0; point < inputPointCount; point += 1) {
    reverseOffsets[point + 1] = reverseOffsets[point] + counts[point];
  }
  const cursors = new Uint32Array(reverseOffsets.slice(0, inputPointCount));
  const reverseSampleKernel = new Uint32Array(entryCount);
  const reverseWeights = new Float32Array(entryCount);
  for (let entryIndex = 0; entryIndex < entryCount; entryIndex += 1) {
    const point = knnIndices[entryIndex];
    const reverseIndex = cursors[point];
    cursors[point] += 1;
    reverseSampleKernel[reverseIndex] = Math.floor(entryIndex / neighborCount);
    reverseWeights[reverseIndex] = knnWeights[entryIndex];
  }

  let knnIndexBuffer: GPUBuffer | undefined;
  let knnWeightBuffer: GPUBuffer | undefined;
  let reverseOffsetBuffer: GPUBuffer | undefined;
  let reverseSampleKernelBuffer: GPUBuffer | undefined;
  let reverseWeightBuffer: GPUBuffer | undefined;
  try {
    knnIndexBuffer = uploadUint32StorageBuffer(gpuDevice, knnIndices);
    knnWeightBuffer = uploadFloat32StorageBuffer(gpuDevice, knnWeights);
    reverseOffsetBuffer = uploadUint32StorageBuffer(gpuDevice, reverseOffsets);
    reverseSampleKernelBuffer = uploadUint32StorageBuffer(
      gpuDevice,
      reverseSampleKernel,
    );
    reverseWeightBuffer = uploadFloat32StorageBuffer(gpuDevice, reverseWeights);
    return {
      inputPointCount,
      sampleCount,
      neighborCount,
      entryCount,
      knnIndexBuffer,
      knnWeightBuffer,
      reverseOffsetBuffer,
      reverseSampleKernelBuffer,
      reverseWeightBuffer,
      forwardKnnBytes: knnIndices.byteLength + knnWeights.byteLength,
      reverseAdjacencyBytes:
        reverseOffsets.byteLength +
        reverseSampleKernel.byteLength +
        reverseWeights.byteLength,
    };
  } catch (error) {
    knnIndexBuffer?.destroy();
    knnWeightBuffer?.destroy();
    reverseOffsetBuffer?.destroy();
    reverseSampleKernelBuffer?.destroy();
    reverseWeightBuffer?.destroy();
    throw error;
  }
};

export const destroyPointCloudConvKnnBuffers = (
  buffers: PointCloudConvKnnBuffers,
): void => {
  buffers.knnIndexBuffer.destroy();
  buffers.knnWeightBuffer.destroy();
  buffers.reverseOffsetBuffer.destroy();
  buffers.reverseSampleKernelBuffer.destroy();
  buffers.reverseWeightBuffer.destroy();
};

export const createPointCloudConvStaticBuffers = (
  gpuDevice: GPUDevice,
  weight: Float32Array,
  bias: Float32Array | readonly number[],
  kernelIndexMap: readonly number[] = POINT_CLOUD_CONV_DEFAULT_INDEX_MAP,
): PointCloudConvStaticBuffers => {
  const biasValues =
    bias instanceof Float32Array ? bias : new Float32Array(bias);
  const kernelMapValues = new Uint32Array(kernelIndexMap);
  let weightBuffer: GPUBuffer | undefined;
  let biasBuffer: GPUBuffer | undefined;
  let kernelIndexMapBuffer: GPUBuffer | undefined;
  try {
    weightBuffer = uploadFloat32StorageBuffer(gpuDevice, weight);
    biasBuffer = uploadFloat32StorageBuffer(gpuDevice, biasValues);
    kernelIndexMapBuffer = uploadUint32StorageBuffer(
      gpuDevice,
      kernelMapValues,
    );
    return {
      weightBuffer,
      biasBuffer,
      kernelIndexMapBuffer,
      byteLength:
        weight.byteLength + biasValues.byteLength + kernelMapValues.byteLength,
    };
  } catch (error) {
    weightBuffer?.destroy();
    biasBuffer?.destroy();
    kernelIndexMapBuffer?.destroy();
    throw error;
  }
};

export const destroyPointCloudConvStaticBuffers = (
  buffers: PointCloudConvStaticBuffers,
): void => {
  buffers.weightBuffer.destroy();
  buffers.biasBuffer.destroy();
  buffers.kernelIndexMapBuffer.destroy();
};

const dispatchPipeline = (
  gpuDevice: GPUDevice,
  encoder: GPUCommandEncoder,
  cacheKey: string,
  shaderCode: string,
  entries: GPUBindGroupEntry[],
  dispatchElementCount: number,
): void => {
  const pipeline = getOrCreateComputePipeline(gpuDevice, cacheKey, shaderCode);
  const bindGroup = gpuDevice.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });
  const workgroupCount = Math.ceil(dispatchElementCount / pcConvWorkgroupSize);
  if (workgroupCount > maxDispatchWorkgroupCount(gpuDevice)) {
    throw new Error(
      `Point-cloud convolution dispatch exceeds WebGPU workgroup limit: ${workgroupCount} > ${maxDispatchWorkgroupCount(
        gpuDevice,
      )}.`,
    );
  }
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroupCount);
  pass.end();
};

const runOptimizedForwardBuffer = (
  gpuDevice: GPUDevice,
  input: GpuBufferRef,
  inputShape: FeatureMatrixShape,
  sampleCount: number,
  neighborCount: number,
  outputChannels: number,
  knnBuffers: PointCloudConvKnnBuffers,
  staticBuffers: PointCloudConvStaticBuffers,
  maxIntermediateBytes: number | undefined,
): GpuBufferRef => {
  const { batchSamples, intermediateBytes } = intermediateBatchSamples(
    gpuDevice,
    sampleCount,
    inputShape.channelCount,
    Math.max(9 * inputShape.channelCount, outputChannels),
    maxIntermediateBytes,
  );
  const outputBuffer = gpuDevice.createBuffer({
    size: sampleCount * outputChannels * Float32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_STORAGE_COPY_SRC,
  });
  const intermediateBuffer = gpuDevice.createBuffer({
    size: intermediateBytes,
    usage: BUFFER_USAGE_STORAGE_COPY_SRC,
  });
  const uniformBuffer = createChunkUniformBuffer(gpuDevice);
  const interpolateShader = makePointCloudConvInterpolateShader();
  const projectShader = makePointCloudConvProjectShader();
  let success = false;

  try {
    for (
      let batchStart = 0;
      batchStart < sampleCount;
      batchStart += batchSamples
    ) {
      const batchSampleCount = Math.min(batchSamples, sampleCount - batchStart);
      writeChunkUniforms(
        gpuDevice,
        uniformBuffer,
        inputShape,
        sampleCount,
        neighborCount,
        outputChannels,
        batchStart,
        batchSampleCount,
      );
      const encoder = gpuDevice.createCommandEncoder();
      dispatchPipeline(
        gpuDevice,
        encoder,
        "pc-conv:interpolate",
        interpolateShader,
        [
          { binding: 0, resource: { buffer: input.buffer } },
          { binding: 1, resource: { buffer: knnBuffers.knnIndexBuffer } },
          { binding: 2, resource: { buffer: knnBuffers.knnWeightBuffer } },
          { binding: 3, resource: { buffer: uniformBuffer } },
          { binding: 4, resource: { buffer: intermediateBuffer } },
        ],
        batchSampleCount * 9 * inputShape.channelCount,
      );
      dispatchPipeline(
        gpuDevice,
        encoder,
        "pc-conv:project",
        projectShader,
        [
          { binding: 0, resource: { buffer: intermediateBuffer } },
          { binding: 1, resource: { buffer: staticBuffers.weightBuffer } },
          { binding: 2, resource: { buffer: staticBuffers.biasBuffer } },
          {
            binding: 3,
            resource: { buffer: staticBuffers.kernelIndexMapBuffer },
          },
          { binding: 4, resource: { buffer: uniformBuffer } },
          { binding: 5, resource: { buffer: outputBuffer } },
        ],
        batchSampleCount * outputChannels,
      );
      gpuDevice.queue.submit([encoder.finish()]);
    }
    success = true;
    return ownedBuffer(outputBuffer);
  } finally {
    if (!success) outputBuffer.destroy();
    intermediateBuffer.destroy();
    uniformBuffer.destroy();
  }
};

const zeroOutputBuffer = (
  gpuDevice: GPUDevice,
  outputBuffer: GPUBuffer,
  count: number,
): void => {
  const uniformBuffer = createRangeUniformBuffer(gpuDevice);
  const shader = makePointCloudConvZeroRangeShader();
  try {
    const maxElements = maxDispatchElementCount(gpuDevice);
    for (let start = 0; start < count; start += maxElements) {
      const rangeCount = Math.min(maxElements, count - start);
      writeRangeUniforms(gpuDevice, uniformBuffer, start, rangeCount);
      const encoder = gpuDevice.createCommandEncoder();
      dispatchPipeline(
        gpuDevice,
        encoder,
        "pc-conv:zero-range",
        shader,
        [
          { binding: 0, resource: { buffer: outputBuffer } },
          { binding: 1, resource: { buffer: uniformBuffer } },
        ],
        rangeCount,
      );
      gpuDevice.queue.submit([encoder.finish()]);
    }
  } finally {
    uniformBuffer.destroy();
  }
};

const runOptimizedBackwardFeaturesBuffer = (
  gpuDevice: GPUDevice,
  gradOut: GpuBufferRef,
  inputShape: FeatureMatrixShape,
  sampleCount: number,
  neighborCount: number,
  outputChannels: number,
  knnBuffers: PointCloudConvKnnBuffers,
  staticBuffers: PointCloudConvStaticBuffers,
  maxIntermediateBytes: number | undefined,
): GpuBufferRef => {
  const { batchSamples, intermediateBytes } = intermediateBatchSamples(
    gpuDevice,
    sampleCount,
    inputShape.channelCount,
    9 * inputShape.channelCount,
    maxIntermediateBytes,
  );
  const outputCount = inputShape.pointCount * inputShape.channelCount;
  const outputBuffer = gpuDevice.createBuffer({
    size: outputCount * Float32Array.BYTES_PER_ELEMENT,
    usage: BUFFER_USAGE_STORAGE_COPY_SRC,
  });
  const intermediateBuffer = gpuDevice.createBuffer({
    size: intermediateBytes,
    usage: BUFFER_USAGE_STORAGE_COPY_SRC,
  });
  const uniformBuffer = createChunkUniformBuffer(gpuDevice);
  const projectShader = makePointCloudConvBackwardProjectShader();
  const gatherShader = makePointCloudConvReverseGatherShader();
  let success = false;

  try {
    zeroOutputBuffer(gpuDevice, outputBuffer, outputCount);
    for (
      let batchStart = 0;
      batchStart < sampleCount;
      batchStart += batchSamples
    ) {
      const batchSampleCount = Math.min(batchSamples, sampleCount - batchStart);
      writeChunkUniforms(
        gpuDevice,
        uniformBuffer,
        inputShape,
        sampleCount,
        neighborCount,
        outputChannels,
        batchStart,
        batchSampleCount,
      );
      const encoder = gpuDevice.createCommandEncoder();
      dispatchPipeline(
        gpuDevice,
        encoder,
        "pc-conv:backward-project",
        projectShader,
        [
          { binding: 0, resource: { buffer: gradOut.buffer } },
          { binding: 1, resource: { buffer: staticBuffers.weightBuffer } },
          {
            binding: 2,
            resource: { buffer: staticBuffers.kernelIndexMapBuffer },
          },
          { binding: 3, resource: { buffer: uniformBuffer } },
          { binding: 4, resource: { buffer: intermediateBuffer } },
        ],
        batchSampleCount * 9 * inputShape.channelCount,
      );
      gpuDevice.queue.submit([encoder.finish()]);
      const maxElements = maxDispatchElementCount(gpuDevice);
      for (
        let outputStart = 0;
        outputStart < outputCount;
        outputStart += maxElements
      ) {
        const outputRangeCount = Math.min(
          maxElements,
          outputCount - outputStart,
        );
        writeChunkUniforms(
          gpuDevice,
          uniformBuffer,
          inputShape,
          sampleCount,
          neighborCount,
          outputChannels,
          batchStart,
          batchSampleCount,
          outputStart,
        );
        const gatherEncoder = gpuDevice.createCommandEncoder();
        dispatchPipeline(
          gpuDevice,
          gatherEncoder,
          "pc-conv:reverse-gather",
          gatherShader,
          [
            { binding: 0, resource: { buffer: intermediateBuffer } },
            {
              binding: 1,
              resource: { buffer: knnBuffers.reverseOffsetBuffer },
            },
            {
              binding: 2,
              resource: { buffer: knnBuffers.reverseSampleKernelBuffer },
            },
            {
              binding: 3,
              resource: { buffer: knnBuffers.reverseWeightBuffer },
            },
            { binding: 4, resource: { buffer: uniformBuffer } },
            { binding: 5, resource: { buffer: outputBuffer } },
          ],
          outputRangeCount,
        );
        gpuDevice.queue.submit([gatherEncoder.finish()]);
      }
    }
    success = true;
    return ownedBuffer(outputBuffer);
  } finally {
    if (!success) outputBuffer.destroy();
    intermediateBuffer.destroy();
    uniformBuffer.destroy();
  }
};

const assertKnnBuffersMatch = (
  inputShape: FeatureMatrixShape,
  sampleCount: number,
  neighborCount: number,
  buffers: PointCloudConvKnnBuffers | undefined,
): void => {
  if (buffers === undefined) return;
  if (
    buffers.inputPointCount !== inputShape.pointCount ||
    buffers.sampleCount !== sampleCount ||
    buffers.neighborCount !== neighborCount
  ) {
    throw new Error("Persistent pc-conv KNN buffers do not match the layer.");
  }
};

export const runPointCloudConvForwardBuffer = async (
  input: GpuBufferRef,
  inputShape: FeatureMatrixShape,
  knnIndices: Uint32Array,
  knnWeights: Float32Array,
  sampleCount: number,
  neighborCount: number,
  weight: Float32Array,
  weightShape: readonly [number, number, number, number],
  bias: Float32Array | readonly number[],
  kernelIndexMap: readonly number[] = POINT_CLOUD_CONV_DEFAULT_INDEX_MAP,
  staticBuffers?: PointCloudConvStaticBuffers,
  knnBuffers?: PointCloudConvKnnBuffers,
  kernelVariant: PointCloudConvForwardKernelVariant = "scalar",
  maxIntermediateBytes?: number,
): Promise<GpuBufferRef> => {
  assertSupportedForwardKernelVariant(kernelVariant);
  assertKnnBuffersMatch(inputShape, sampleCount, neighborCount, knnBuffers);
  const { outputChannels } = validateConvParams(
    inputShape,
    sampleCount,
    neighborCount,
    knnIndices,
    knnWeights,
    weight.length,
    weightShape,
    bias.length,
    false,
    kernelIndexMap,
  );
  const gpuDevice = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const outputCount = sampleCount * outputChannels;
  const localStaticBuffers =
    staticBuffers ??
    createPointCloudConvStaticBuffers(gpuDevice, weight, bias, kernelIndexMap);
  let localKnnBuffers: PointCloudConvKnnBuffers | undefined;
  const activeKnnBuffers =
    knnBuffers ??
    (kernelVariant === "interpolate-project"
      ? (localKnnBuffers = createPointCloudConvKnnBuffers(
          gpuDevice,
          inputShape.pointCount,
          sampleCount,
          neighborCount,
          knnIndices,
          knnWeights,
        ))
      : undefined);
  if (kernelVariant === "interpolate-project") {
    if (activeKnnBuffers === undefined) {
      throw new Error("Missing pc-conv KNN buffers for optimized forward.");
    }
    try {
      return runOptimizedForwardBuffer(
        gpuDevice,
        input,
        inputShape,
        sampleCount,
        neighborCount,
        outputChannels,
        activeKnnBuffers,
        localStaticBuffers,
        maxIntermediateBytes,
      );
    } finally {
      if (staticBuffers === undefined) {
        destroyPointCloudConvStaticBuffers(localStaticBuffers);
      }
      if (localKnnBuffers !== undefined) {
        destroyPointCloudConvKnnBuffers(localKnnBuffers);
      }
    }
  }

  const knnIndexBuffer =
    knnBuffers?.knnIndexBuffer ??
    uploadUint32StorageBuffer(gpuDevice, knnIndices);
  const knnWeightBuffer =
    knnBuffers?.knnWeightBuffer ??
    uploadFloat32StorageBuffer(gpuDevice, knnWeights);
  const uniformBuffer = createUniformBuffer(
    gpuDevice,
    inputShape,
    sampleCount,
    neighborCount,
    outputChannels,
  );
  try {
    return ownedBuffer(
      runUnaryShaderToBuffer(
        gpuDevice,
        makePointCloudConvForwardShader(outputCount),
        input.buffer,
        outputCount,
        [
          { binding: 1, resource: { buffer: knnIndexBuffer } },
          { binding: 2, resource: { buffer: knnWeightBuffer } },
          { binding: 3, resource: { buffer: localStaticBuffers.weightBuffer } },
          { binding: 4, resource: { buffer: localStaticBuffers.biasBuffer } },
          {
            binding: 5,
            resource: { buffer: localStaticBuffers.kernelIndexMapBuffer },
          },
          { binding: 6, resource: { buffer: uniformBuffer } },
        ],
      ),
    );
  } finally {
    if (knnBuffers === undefined) {
      knnIndexBuffer.destroy();
      knnWeightBuffer.destroy();
    }
    if (staticBuffers === undefined) {
      destroyPointCloudConvStaticBuffers(localStaticBuffers);
    }
    uniformBuffer.destroy();
  }
};

export const runPointCloudConvBackwardFeaturesBuffer = async (
  gradOut: GpuBufferRef,
  inputShape: FeatureMatrixShape,
  knnIndices: Uint32Array,
  knnWeights: Float32Array,
  sampleCount: number,
  neighborCount: number,
  weight: Float32Array,
  weightShape: readonly [number, number, number, number],
  kernelIndexMap: readonly number[] = POINT_CLOUD_CONV_DEFAULT_INDEX_MAP,
  staticBuffers?: PointCloudConvStaticBuffers,
  knnBuffers?: PointCloudConvKnnBuffers,
  kernelVariant: PointCloudConvBackwardFeaturesKernelVariant = "scalar",
  maxIntermediateBytes?: number,
): Promise<GpuBufferRef> => {
  assertSupportedBackwardFeaturesKernelVariant(kernelVariant);
  assertKnnBuffersMatch(inputShape, sampleCount, neighborCount, knnBuffers);
  const { outputChannels } = validateConvParams(
    inputShape,
    sampleCount,
    neighborCount,
    knnIndices,
    knnWeights,
    weight.length,
    weightShape,
    sampleCount * weightShape[0],
    true,
    kernelIndexMap,
  );
  const gpuDevice = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const outputCount = inputShape.pointCount * inputShape.channelCount;
  const localStaticBuffers =
    staticBuffers ??
    createPointCloudConvStaticBuffers(
      gpuDevice,
      weight,
      new Float32Array(outputChannels),
      kernelIndexMap,
    );
  let localKnnBuffers: PointCloudConvKnnBuffers | undefined;
  const activeKnnBuffers =
    knnBuffers ??
    (kernelVariant === "reverse-adjacency"
      ? (localKnnBuffers = createPointCloudConvKnnBuffers(
          gpuDevice,
          inputShape.pointCount,
          sampleCount,
          neighborCount,
          knnIndices,
          knnWeights,
        ))
      : undefined);
  if (kernelVariant === "reverse-adjacency") {
    if (activeKnnBuffers === undefined) {
      throw new Error(
        "Missing pc-conv KNN buffers for reverse-adjacency backward.",
      );
    }
    try {
      return runOptimizedBackwardFeaturesBuffer(
        gpuDevice,
        gradOut,
        inputShape,
        sampleCount,
        neighborCount,
        outputChannels,
        activeKnnBuffers,
        localStaticBuffers,
        maxIntermediateBytes,
      );
    } finally {
      if (staticBuffers === undefined) {
        destroyPointCloudConvStaticBuffers(localStaticBuffers);
      }
      if (localKnnBuffers !== undefined) {
        destroyPointCloudConvKnnBuffers(localKnnBuffers);
      }
    }
  }

  const knnIndexBuffer =
    knnBuffers?.knnIndexBuffer ??
    uploadUint32StorageBuffer(gpuDevice, knnIndices);
  const knnWeightBuffer =
    knnBuffers?.knnWeightBuffer ??
    uploadFloat32StorageBuffer(gpuDevice, knnWeights);
  const uniformBuffer = createUniformBuffer(
    gpuDevice,
    inputShape,
    sampleCount,
    neighborCount,
    outputChannels,
  );
  try {
    return ownedBuffer(
      runUnaryShaderToBuffer(
        gpuDevice,
        makePointCloudConvBackwardFeaturesShader(outputCount),
        gradOut.buffer,
        outputCount,
        [
          { binding: 1, resource: { buffer: knnIndexBuffer } },
          { binding: 2, resource: { buffer: knnWeightBuffer } },
          { binding: 3, resource: { buffer: localStaticBuffers.weightBuffer } },
          {
            binding: 4,
            resource: { buffer: localStaticBuffers.kernelIndexMapBuffer },
          },
          { binding: 5, resource: { buffer: uniformBuffer } },
        ],
      ),
    );
  } finally {
    if (knnBuffers === undefined) {
      knnIndexBuffer.destroy();
      knnWeightBuffer.destroy();
    }
    if (staticBuffers === undefined) {
      destroyPointCloudConvStaticBuffers(localStaticBuffers);
    }
    uniformBuffer.destroy();
  }
};

export const runPointCloudConvForwardReadback = async (
  input: Float32Array,
  inputShape: FeatureMatrixShape,
  knnIndices: Uint32Array,
  knnWeights: Float32Array,
  sampleCount: number,
  neighborCount: number,
  weight: Float32Array,
  weightShape: readonly [number, number, number, number],
  bias: Float32Array | readonly number[],
  kernelIndexMap: readonly number[] = POINT_CLOUD_CONV_DEFAULT_INDEX_MAP,
  kernelVariant: PointCloudConvForwardKernelVariant = "scalar",
  maxIntermediateBytes?: number,
): Promise<Float32Array> => {
  validateFeatureMatrixLength(inputShape, input, "pc-conv-forward input");
  const inputBuffer = uploadToOwnedBuffer(getGpuDevice(), input);
  let outputBuffer: GpuBufferRef | null = null;
  try {
    outputBuffer = await runPointCloudConvForwardBuffer(
      inputBuffer,
      inputShape,
      knnIndices,
      knnWeights,
      sampleCount,
      neighborCount,
      weight,
      weightShape,
      bias,
      kernelIndexMap,
      undefined,
      undefined,
      kernelVariant,
      maxIntermediateBytes,
    );
    return await readGpuBufferToArray(
      getGpuDevice(),
      outputBuffer.buffer,
      sampleCount * weightShape[0],
    );
  } finally {
    releaseOwnedBuffer(inputBuffer);
    if (outputBuffer !== null) releaseOwnedBuffer(outputBuffer);
  }
};

export const runPointCloudConvBackwardFeaturesReadback = async (
  gradOut: Float32Array,
  inputShape: FeatureMatrixShape,
  knnIndices: Uint32Array,
  knnWeights: Float32Array,
  sampleCount: number,
  neighborCount: number,
  weight: Float32Array,
  weightShape: readonly [number, number, number, number],
  kernelIndexMap: readonly number[] = POINT_CLOUD_CONV_DEFAULT_INDEX_MAP,
  kernelVariant: PointCloudConvBackwardFeaturesKernelVariant = "scalar",
  maxIntermediateBytes?: number,
): Promise<Float32Array> => {
  validatePointCloudConvGradOutLength(
    gradOut.length,
    sampleCount,
    weightShape[0],
  );
  const gradOutBuffer = uploadToOwnedBuffer(getGpuDevice(), gradOut);
  let outputBuffer: GpuBufferRef | null = null;
  try {
    outputBuffer = await runPointCloudConvBackwardFeaturesBuffer(
      gradOutBuffer,
      inputShape,
      knnIndices,
      knnWeights,
      sampleCount,
      neighborCount,
      weight,
      weightShape,
      kernelIndexMap,
      undefined,
      undefined,
      kernelVariant,
      maxIntermediateBytes,
    );
    return await readGpuBufferToArray(
      getGpuDevice(),
      outputBuffer.buffer,
      inputShape.pointCount * inputShape.channelCount,
    );
  } finally {
    releaseOwnedBuffer(gradOutBuffer);
    if (outputBuffer !== null) releaseOwnedBuffer(outputBuffer);
  }
};
