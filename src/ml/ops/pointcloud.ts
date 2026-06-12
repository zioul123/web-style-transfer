import type {
  FeatureMatrixData,
  FeatureMatrixShape,
  PointCloudKnnData,
  SurfacePoolData,
  TensorData,
} from "../../types/worker-protocol/core";
import {
  POINT_CLOUD_CONV_DEFAULT_INDEX_MAP,
  assertSameFeatureMatrixShape,
  getSurfacePoolHasOutput,
  validatePerChannelValues,
  validatePointCloudConvBiasLength,
  validatePointCloudConvGradOutShape,
  validatePointCloudConvParams as validatePointCloudConvMetadata,
  validateSurfacePoolGradOutShape,
  validateSurfacePoolParams as validateSurfacePoolMetadata,
} from "../pointcloud/metadata";

const isPointCloudConvBias = (
  value: readonly number[] | FeatureMatrixData,
): value is readonly number[] => Array.isArray(value);

const validatePointCloudConvData = (
  inputShape: FeatureMatrixShape,
  knn: PointCloudKnnData,
  weight: TensorData,
  biasOrGradOut: readonly number[] | FeatureMatrixData,
  kernelIndexMap: readonly number[],
): { inputChannels: number; outputChannels: number } => {
  const channels = validatePointCloudConvMetadata(
    inputShape,
    knn,
    weight.shape,
    kernelIndexMap,
  );
  if (isPointCloudConvBias(biasOrGradOut)) {
    validatePointCloudConvBiasLength(
      biasOrGradOut.length,
      channels.outputChannels,
    );
  } else {
    validatePointCloudConvGradOutShape(
      biasOrGradOut,
      knn.sampleCount,
      channels.outputChannels,
    );
  }
  return channels;
};

const weightAt = (
  weight: TensorData,
  outputChannel: number,
  inputChannel: number,
  mappedKernelIndex: number,
): number => {
  const inputChannels = weight.shape[1];
  return (
    weight.values[
      (outputChannel * inputChannels + inputChannel) * 9 + mappedKernelIndex
    ] ?? 0
  );
};

export const cpuPointwiseExpForward = (
  input: FeatureMatrixData,
): FeatureMatrixData => {
  const out = new Float32Array(input.values.length);
  for (let i = 0; i < input.values.length; i += 1) {
    out[i] = Math.exp(input.values[i]);
  }
  return {
    pointCount: input.pointCount,
    channelCount: input.channelCount,
    values: out,
  };
};

export const cpuPointCloudConvForward = (
  input: FeatureMatrixData,
  knn: PointCloudKnnData,
  weight: TensorData,
  bias: readonly number[],
  kernelIndexMap: readonly number[] = POINT_CLOUD_CONV_DEFAULT_INDEX_MAP,
): FeatureMatrixData => {
  const { inputChannels, outputChannels } = validatePointCloudConvData(
    input,
    knn,
    weight,
    bias,
    kernelIndexMap,
  );
  const output = new Float32Array(knn.sampleCount * outputChannels);
  for (let sample = 0; sample < knn.sampleCount; sample += 1) {
    for (
      let outputChannel = 0;
      outputChannel < outputChannels;
      outputChannel += 1
    ) {
      let sum = bias[outputChannel];
      for (let kernel = 0; kernel < 9; kernel += 1) {
        const mappedKernel = kernelIndexMap[kernel];
        const knnBase = (sample * 9 + kernel) * knn.neighborCount;
        for (
          let inputChannel = 0;
          inputChannel < inputChannels;
          inputChannel += 1
        ) {
          let weightedFeature = 0;
          for (let neighbor = 0; neighbor < knn.neighborCount; neighbor += 1) {
            const knnIndex = knnBase + neighbor;
            const pointIndex = knn.indices[knnIndex];
            weightedFeature +=
              input.values[pointIndex * inputChannels + inputChannel] *
              knn.weights[knnIndex];
          }
          sum +=
            weightedFeature *
            weightAt(weight, outputChannel, inputChannel, mappedKernel);
        }
      }
      output[sample * outputChannels + outputChannel] = sum;
    }
  }
  return {
    pointCount: knn.sampleCount,
    channelCount: outputChannels,
    values: output,
  };
};

export const cpuPointCloudConvBackwardFeatures = (
  inputShape: FeatureMatrixShape,
  gradOut: FeatureMatrixData,
  knn: PointCloudKnnData,
  weight: TensorData,
  kernelIndexMap: readonly number[] = POINT_CLOUD_CONV_DEFAULT_INDEX_MAP,
): FeatureMatrixData => {
  const { inputChannels, outputChannels } = validatePointCloudConvData(
    inputShape,
    knn,
    weight,
    gradOut,
    kernelIndexMap,
  );
  const gradFeatures = new Float32Array(
    inputShape.pointCount * inputShape.channelCount,
  );
  for (let sample = 0; sample < knn.sampleCount; sample += 1) {
    for (let kernel = 0; kernel < 9; kernel += 1) {
      const mappedKernel = kernelIndexMap[kernel];
      const knnBase = (sample * 9 + kernel) * knn.neighborCount;
      for (let neighbor = 0; neighbor < knn.neighborCount; neighbor += 1) {
        const knnIndex = knnBase + neighbor;
        const pointIndex = knn.indices[knnIndex];
        const knnWeight = knn.weights[knnIndex];
        for (
          let inputChannel = 0;
          inputChannel < inputChannels;
          inputChannel += 1
        ) {
          let sum = 0;
          for (
            let outputChannel = 0;
            outputChannel < outputChannels;
            outputChannel += 1
          ) {
            sum +=
              gradOut.values[sample * outputChannels + outputChannel] *
              weightAt(weight, outputChannel, inputChannel, mappedKernel);
          }
          gradFeatures[pointIndex * inputChannels + inputChannel] +=
            sum * knnWeight;
        }
      }
    }
  }
  return {
    pointCount: inputShape.pointCount,
    channelCount: inputShape.channelCount,
    values: gradFeatures,
  };
};

export const cpuSurfacePoolForward = (
  input: FeatureMatrixData,
  pool: SurfacePoolData,
): FeatureMatrixData => {
  const { validInputCount } = validateSurfacePoolMetadata(input, pool);
  const output = new Float32Array(pool.outputPointCount * input.channelCount);
  output.fill(Number.NEGATIVE_INFINITY);
  const mean = new Float32Array(input.channelCount);
  for (let point = 0; point < input.pointCount; point += 1) {
    const outputPoint = pool.mapping[point];
    if (outputPoint < 0) continue;
    const inputBase = point * input.channelCount;
    const outputBase = outputPoint * input.channelCount;
    for (let channel = 0; channel < input.channelCount; channel += 1) {
      const value = input.values[inputBase + channel];
      mean[channel] += value;
      output[outputBase + channel] = Math.max(
        output[outputBase + channel],
        value,
      );
    }
  }
  for (let channel = 0; channel < input.channelCount; channel += 1) {
    mean[channel] /= validInputCount;
  }
  const hasOutput = getSurfacePoolHasOutput(
    pool.outputPointCount,
    pool.mapping,
  );
  for (
    let outputPoint = 0;
    outputPoint < pool.outputPointCount;
    outputPoint += 1
  ) {
    if (hasOutput[outputPoint]) continue;
    const outputBase = outputPoint * input.channelCount;
    for (let channel = 0; channel < input.channelCount; channel += 1) {
      output[outputBase + channel] = mean[channel];
    }
  }
  return {
    pointCount: pool.outputPointCount,
    channelCount: input.channelCount,
    values: output,
  };
};

export const cpuSurfacePoolBackward = (
  input: FeatureMatrixData,
  gradOut: FeatureMatrixData,
  pool: SurfacePoolData,
): FeatureMatrixData => {
  const { validInputCount } = validateSurfacePoolMetadata(input, pool);
  validateSurfacePoolGradOutShape(gradOut, input, pool.outputPointCount);
  const hasOutput = getSurfacePoolHasOutput(
    pool.outputPointCount,
    pool.mapping,
  );
  const orphanGrad = new Float32Array(input.channelCount);
  for (
    let outputPoint = 0;
    outputPoint < pool.outputPointCount;
    outputPoint += 1
  ) {
    if (hasOutput[outputPoint]) continue;
    const gradBase = outputPoint * input.channelCount;
    for (let channel = 0; channel < input.channelCount; channel += 1) {
      orphanGrad[channel] += gradOut.values[gradBase + channel];
    }
  }

  const gradIn = new Float32Array(input.values.length);
  for (let point = 0; point < input.pointCount; point += 1) {
    const outputPoint = pool.mapping[point];
    if (outputPoint < 0) continue;
    const inputBase = point * input.channelCount;
    const gradOutBase = outputPoint * input.channelCount;
    for (let channel = 0; channel < input.channelCount; channel += 1) {
      let maxValue = Number.NEGATIVE_INFINITY;
      let tieCount = 0;
      for (let otherPoint = 0; otherPoint < input.pointCount; otherPoint += 1) {
        if (pool.mapping[otherPoint] !== outputPoint) continue;
        const value = input.values[otherPoint * input.channelCount + channel];
        if (value > maxValue) {
          maxValue = value;
          tieCount = 1;
        } else if (value === maxValue) {
          tieCount += 1;
        }
      }
      let grad = orphanGrad[channel] / validInputCount;
      if (input.values[inputBase + channel] === maxValue) {
        grad += gradOut.values[gradOutBase + channel] / tieCount;
      }
      gradIn[inputBase + channel] = grad;
    }
  }
  return {
    pointCount: input.pointCount,
    channelCount: input.channelCount,
    values: gradIn,
  };
};

export const cpuPointwiseExpBackward = (
  input: FeatureMatrixData,
  gradOut: FeatureMatrixData,
): FeatureMatrixData => {
  assertSameFeatureMatrixShape(input, gradOut);
  const gradIn = new Float32Array(input.values.length);
  for (let i = 0; i < input.values.length; i += 1) {
    gradIn[i] = gradOut.values[i] * Math.exp(input.values[i]);
  }
  return {
    pointCount: input.pointCount,
    channelCount: input.channelCount,
    values: gradIn,
  };
};

export const cpuPointFeatureNormalizeForward = (
  input: FeatureMatrixData,
  mean: readonly number[],
  std: readonly number[],
): FeatureMatrixData => {
  validatePerChannelValues(input, mean, "Mean");
  validatePerChannelValues(input, std, "Std");
  const out = new Float32Array(input.values.length);
  for (let point = 0; point < input.pointCount; point += 1) {
    const base = point * input.channelCount;
    for (let channel = 0; channel < input.channelCount; channel += 1) {
      const index = base + channel;
      out[index] = (input.values[index] - mean[channel]) / std[channel];
    }
  }
  return {
    pointCount: input.pointCount,
    channelCount: input.channelCount,
    values: out,
  };
};

export const cpuPointFeatureNormalizeBackward = (
  gradOut: FeatureMatrixData,
  std: readonly number[],
): FeatureMatrixData => {
  validatePerChannelValues(gradOut, std, "Std");
  const gradIn = new Float32Array(gradOut.values.length);
  for (let point = 0; point < gradOut.pointCount; point += 1) {
    const base = point * gradOut.channelCount;
    for (let channel = 0; channel < gradOut.channelCount; channel += 1) {
      const index = base + channel;
      gradIn[index] = gradOut.values[index] / std[channel];
    }
  }
  return {
    pointCount: gradOut.pointCount,
    channelCount: gradOut.channelCount,
    values: gradIn,
  };
};
