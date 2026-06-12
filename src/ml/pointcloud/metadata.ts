import type {
  FeatureMatrixShape,
  TensorShape,
} from "../../types/worker-protocol/core";

export const POINT_CLOUD_CONV_KERNEL_POINT_COUNT = 9;

export const POINT_CLOUD_CONV_DEFAULT_INDEX_MAP: readonly number[] = [
  4, 5, 2, 1, 0, 3, 6, 7, 8,
];

type NumericValues =
  | readonly number[]
  | Float32Array
  | Int32Array
  | Uint32Array;

export type PointCloudKnnShape = {
  sampleCount: number;
  kernelPointCount?: number;
  neighborCount: number;
  indices: NumericValues;
  weights: NumericValues;
};

export type SurfacePoolShape = {
  inputPointCount?: number;
  outputPointCount: number;
  mapping: NumericValues;
};

export const featureMatrixElementCount = (shape: FeatureMatrixShape): number =>
  shape.pointCount * shape.channelCount;

export const assertSameFeatureMatrixShape = (
  a: FeatureMatrixShape,
  b: FeatureMatrixShape,
): void => {
  if (a.pointCount !== b.pointCount || a.channelCount !== b.channelCount) {
    throw new Error("Feature matrix shapes must match.");
  }
};

export const validateFeatureMatrixLength = (
  shape: FeatureMatrixShape,
  values: { readonly length: number },
  label: string,
): void => {
  const expected = featureMatrixElementCount(shape);
  if (values.length !== expected) {
    throw new Error(
      `${label} expected ${expected} values, got ${values.length}.`,
    );
  }
};

export const validatePerChannelValues = (
  shape: FeatureMatrixShape,
  values: readonly number[],
  label: string,
): void => {
  if (values.length !== shape.channelCount) {
    throw new Error(`${label} length must match feature channel count.`);
  }
};

export const validatePointCloudConvKernelIndexMap = (
  kernelIndexMap: readonly number[],
): void => {
  if (kernelIndexMap.length !== POINT_CLOUD_CONV_KERNEL_POINT_COUNT) {
    throw new Error("Point-cloud conv kernel index map must have 9 entries.");
  }
  const seen = new Set<number>();
  for (const value of kernelIndexMap) {
    if (
      !Number.isInteger(value) ||
      value < 0 ||
      value >= POINT_CLOUD_CONV_KERNEL_POINT_COUNT
    ) {
      throw new Error(
        "Point-cloud conv kernel index map entries must be 0..8.",
      );
    }
    seen.add(value);
  }
  if (seen.size !== POINT_CLOUD_CONV_KERNEL_POINT_COUNT) {
    throw new Error("Point-cloud conv kernel index map must be a permutation.");
  }
};

export const pointCloudConvOutputElementCount = (
  sampleCount: number,
  outputChannels: number,
): number => sampleCount * outputChannels;

export const pointCloudConvWeightElementCount = (
  weightShape: TensorShape,
): number =>
  weightShape[0] * weightShape[1] * POINT_CLOUD_CONV_KERNEL_POINT_COUNT;

export const validatePointCloudConvWeightLength = (
  weightLength: number,
  weightShape: TensorShape,
): void => {
  const expected = pointCloudConvWeightElementCount(weightShape);
  if (weightLength !== expected) {
    throw new Error(`pc-conv expected ${expected} weight values.`);
  }
};

export const validatePointCloudConvParams = (
  inputShape: FeatureMatrixShape,
  knn: PointCloudKnnShape,
  weightShape: TensorShape,
  kernelIndexMap: readonly number[] = POINT_CLOUD_CONV_DEFAULT_INDEX_MAP,
): { inputChannels: number; outputChannels: number } => {
  const [outputChannels, inputChannels, kernelHeight, kernelWidth] =
    weightShape;
  const kernelPointCount =
    knn.kernelPointCount ?? POINT_CLOUD_CONV_KERNEL_POINT_COUNT;
  if (
    inputShape.channelCount !== inputChannels ||
    kernelHeight !== 3 ||
    kernelWidth !== 3 ||
    kernelPointCount !== POINT_CLOUD_CONV_KERNEL_POINT_COUNT
  ) {
    throw new Error("pc-conv only supports VGG-compatible 3x3 weights.");
  }
  if (knn.indices.length !== knn.weights.length) {
    throw new Error("pc-conv KNN indices and weights length mismatch.");
  }
  const expectedKnnLength =
    knn.sampleCount * POINT_CLOUD_CONV_KERNEL_POINT_COUNT * knn.neighborCount;
  if (knn.indices.length !== expectedKnnLength) {
    throw new Error("pc-conv KNN metadata shape mismatch.");
  }
  for (const index of knn.indices) {
    if (
      !Number.isInteger(index) ||
      index < 0 ||
      index >= inputShape.pointCount
    ) {
      throw new Error("pc-conv KNN index is out of input feature bounds.");
    }
  }
  validatePointCloudConvKernelIndexMap(kernelIndexMap);
  return { inputChannels, outputChannels };
};

export const validatePointCloudConvBiasLength = (
  biasLength: number,
  outputChannels: number,
): void => {
  if (biasLength !== outputChannels) {
    throw new Error("pc-conv bias length must match output filters.");
  }
};

export const validatePointCloudConvGradOutShape = (
  gradOutShape: FeatureMatrixShape,
  sampleCount: number,
  outputChannels: number,
): void => {
  if (
    gradOutShape.pointCount !== sampleCount ||
    gradOutShape.channelCount !== outputChannels
  ) {
    throw new Error("pc-conv backward gradOut shape mismatch.");
  }
};

export const validatePointCloudConvGradOutLength = (
  gradOutLength: number,
  sampleCount: number,
  outputChannels: number,
): void => {
  const expected = pointCloudConvOutputElementCount(
    sampleCount,
    outputChannels,
  );
  if (gradOutLength !== expected) {
    throw new Error("pc-conv backward gradOut shape mismatch.");
  }
};

export const validateSurfacePoolParams = (
  inputShape: FeatureMatrixShape,
  pool: SurfacePoolShape,
): { validInputCount: number } => {
  if (
    pool.inputPointCount !== undefined &&
    pool.inputPointCount !== inputShape.pointCount
  ) {
    throw new Error("surface-pool input and mapping shape mismatch.");
  }
  if (pool.mapping.length !== inputShape.pointCount) {
    throw new Error("surface-pool input and mapping shape mismatch.");
  }
  let validInputCount = 0;
  for (const outputPoint of pool.mapping) {
    if (!Number.isInteger(outputPoint)) {
      throw new Error("Surface pool mapping entries must be integers.");
    }
    if (outputPoint >= pool.outputPointCount) {
      throw new Error("surface-pool mapping points past output bounds.");
    }
    if (outputPoint >= 0) validInputCount += 1;
  }
  if (validInputCount === 0) {
    throw new Error("surface-pool requires at least one valid mapping.");
  }
  return { validInputCount };
};

export const getSurfacePoolHasOutput = (
  outputPointCount: number,
  mapping: NumericValues,
): boolean[] => {
  const hasOutput = Array.from(
    { length: outputPointCount },
    (): boolean => false,
  );
  for (const outputPoint of mapping) {
    if (outputPoint >= 0) hasOutput[outputPoint] = true;
  }
  return hasOutput;
};

export const validateSurfacePoolGradOutShape = (
  gradOutShape: FeatureMatrixShape,
  inputShape: FeatureMatrixShape,
  outputPointCount: number,
): void => {
  if (
    gradOutShape.pointCount !== outputPointCount ||
    gradOutShape.channelCount !== inputShape.channelCount
  ) {
    throw new Error("surface-pool backward gradOut shape mismatch.");
  }
};

export const validateSurfacePoolGradOutLength = (
  gradOutLength: number,
  inputShape: FeatureMatrixShape,
  outputPointCount: number,
): void => {
  const expected = outputPointCount * inputShape.channelCount;
  if (gradOutLength !== expected) {
    throw new Error("surface-pool backward gradOut shape mismatch.");
  }
};
