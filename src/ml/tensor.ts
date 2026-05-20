import type { TensorData, TensorShape } from "../types";

const elementCount = (shape: TensorShape): number =>
  shape[0] * shape[1] * shape[2] * shape[3];

export const assertSameLength = (a: TensorData, b: TensorData): void => {
  if (elementCount(a.shape) !== elementCount(b.shape)) {
    throw new Error("Tensor sizes must match.");
  }
};

export const createTensor = (
  shape: TensorShape,
  values: number[],
): TensorData => {
  const expected = elementCount(shape);
  if (values.length !== expected) {
    throw new Error(`Expected ${expected} values, got ${values.length}.`);
  }
  return { shape, values: new Float32Array(values) };
};
