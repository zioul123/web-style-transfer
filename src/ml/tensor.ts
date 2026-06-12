import type {
  FeatureMatrixData,
  FeatureMatrixShape,
  PointCloudKnnData,
  SurfacePoolData,
  TensorData,
  TensorShape,
} from "../types/worker-protocol/core";
import {
  POINT_CLOUD_CONV_KERNEL_POINT_COUNT,
  featureMatrixElementCount as featureMatrixShapeElementCount,
} from "./pointcloud/metadata";
export { assertSameFeatureMatrixShape } from "./pointcloud/metadata";

const elementCount = (shape: TensorShape): number =>
  shape[0] * shape[1] * shape[2] * shape[3];

type NumericValues = ArrayLike<number>;

export const featureMatrixElementCount = (
  pointCount: number,
  channelCount: number,
): number => featureMatrixShapeElementCount({ pointCount, channelCount });

export const assertSameLength = (a: TensorData, b: TensorData): void => {
  if (elementCount(a.shape) !== elementCount(b.shape)) {
    throw new Error("Tensor sizes must match.");
  }
};

export const createTensor = (
  shape: TensorShape,
  values: NumericValues,
): TensorData => {
  const expected = elementCount(shape);
  if (values.length !== expected) {
    throw new Error(`Expected ${expected} values, got ${values.length}.`);
  }
  return {
    shape,
    values: values instanceof Float32Array ? values : new Float32Array(values),
  };
};

export const createFeatureMatrix = (
  pointCount: number,
  channelCount: number,
  values: NumericValues,
): FeatureMatrixData => {
  const expected = featureMatrixElementCount(pointCount, channelCount);
  if (values.length !== expected) {
    throw new Error(
      `Expected ${expected} feature values, got ${values.length}.`,
    );
  }
  return {
    pointCount,
    channelCount,
    values: values instanceof Float32Array ? values : new Float32Array(values),
  };
};

export const createPointCloudKnn = (
  sampleCount: number,
  kernelPointCount: number,
  neighborCount: number,
  indices: NumericValues,
  weights: NumericValues,
): PointCloudKnnData => {
  const expected = sampleCount * kernelPointCount * neighborCount;
  if (kernelPointCount !== POINT_CLOUD_CONV_KERNEL_POINT_COUNT) {
    throw new Error("Point-cloud conv KNN metadata must use 9 kernel points.");
  }
  if (indices.length !== expected || weights.length !== expected) {
    throw new Error(`Expected ${expected} KNN values.`);
  }
  for (let i = 0; i < indices.length; i += 1) {
    const index = indices[i];
    if (!Number.isInteger(index) || index < 0) {
      throw new Error("KNN indices must be non-negative integers.");
    }
  }
  const indexValues =
    indices instanceof Uint32Array
      ? indices
      : indices instanceof Int32Array
        ? new Uint32Array(indices.buffer, indices.byteOffset, indices.length)
        : new Uint32Array(indices);
  return {
    sampleCount,
    kernelPointCount,
    neighborCount,
    indices: indexValues,
    weights:
      weights instanceof Float32Array ? weights : new Float32Array(weights),
  };
};

export const createFeatureMatrixShape = (
  pointCount: number,
  channelCount: number,
): FeatureMatrixShape => ({ pointCount, channelCount });

export const createSurfacePoolMap = (
  inputPointCount: number,
  outputPointCount: number,
  mapping: NumericValues,
): SurfacePoolData => {
  if (mapping.length !== inputPointCount) {
    throw new Error("Surface pool mapping length must match input points.");
  }
  for (let i = 0; i < mapping.length; i += 1) {
    const value = mapping[i];
    if (!Number.isInteger(value)) {
      throw new Error("Surface pool mapping entries must be integers.");
    }
    if (value >= outputPointCount) {
      throw new Error("Surface pool mapping points past output bounds.");
    }
  }
  const mappingValues =
    mapping instanceof Int32Array ? mapping : new Int32Array(mapping);
  return { inputPointCount, outputPointCount, mapping: mappingValues };
};
