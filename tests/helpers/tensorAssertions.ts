import { expect } from "@playwright/test";

export type TensorDiffStats = {
  maxDiff: number;
  meanDiff: number;
};

export const vectorMaxDiff = (
  actual: readonly number[],
  expected: readonly number[],
): number => {
  expect(actual.length).toBe(expected.length);
  let maxDiff = 0;
  for (let index = 0; index < expected.length; index += 1) {
    maxDiff = Math.max(maxDiff, Math.abs(actual[index] - expected[index]));
  }
  return maxDiff;
};

export const vectorMeanDiff = (
  actual: readonly number[],
  expected: readonly number[],
): number => {
  expect(actual.length).toBe(expected.length);
  if (expected.length === 0) return 0;

  let totalDiff = 0;
  for (let index = 0; index < expected.length; index += 1) {
    totalDiff += Math.abs(actual[index] - expected[index]);
  }
  return totalDiff / expected.length;
};

export const vectorDiffStats = (
  actual: readonly number[],
  expected: readonly number[],
): TensorDiffStats => ({
  maxDiff: vectorMaxDiff(actual, expected),
  meanDiff: vectorMeanDiff(actual, expected),
});

export const expectTensorCloseTo = (
  actual: readonly number[],
  expected: readonly number[],
  precision = 5,
): void => {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index], precision);
  }
};

export const expectScalarCloseTo = (
  actual: number,
  expected: number,
  precision = 5,
): void => {
  expect(actual).toBeCloseTo(expected, precision);
};

export const expectScalarWithinTolerance = (
  actual: number,
  expected: number,
  tolerance: number,
): void => {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance);
};
