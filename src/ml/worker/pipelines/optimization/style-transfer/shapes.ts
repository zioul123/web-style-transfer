import type { TensorShape4D } from "./types";

export const elementCount = (shape: TensorShape4D): number =>
  shape[0] * shape[1] * shape[2] * shape[3];

export const gramElementCount = (shape: TensorShape4D): number =>
  shape[1] * shape[1];

export const convOutputShape = (
  inputShape: TensorShape4D,
  outChannels: number,
): TensorShape4D => [inputShape[0], outChannels, inputShape[2], inputShape[3]];

export const pooledShape = (inputShape: TensorShape4D): TensorShape4D => [
  inputShape[0],
  inputShape[1],
  Math.floor(inputShape[2] / 2),
  Math.floor(inputShape[3] / 2),
];
