import type { InputOptimizerConfig, InputOptimizerVectorOps } from "./types";

const clampUnit = (value: number): number => Math.max(0, Math.min(1, value));

export const createCpuVectorOps = (
  count: number,
): InputOptimizerVectorOps<Float32Array> => ({
  clone: async (input: Float32Array): Promise<Float32Array> =>
    new Float32Array(input),

  zeros: async (): Promise<Float32Array> => new Float32Array(count),

  sub: async (a: Float32Array, b: Float32Array): Promise<Float32Array> => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = a[i] - b[i];
    return out;
  },

  scale: async (input: Float32Array, scalar: number): Promise<Float32Array> => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) out[i] = input[i] * scalar;
    return out;
  },

  addScaled: async (
    input: Float32Array,
    direction: Float32Array,
    scalar: number,
  ): Promise<Float32Array> => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1)
      out[i] = input[i] + direction[i] * scalar;
    return out;
  },

  addScaledByDotAndScalarBuffer: async (
    input: Float32Array,
    direction: Float32Array,
    dotLeft: Float32Array,
    dotRight: Float32Array,
    dotScale: number,
    scalarBuffer: Float32Array,
    scalarScale: number,
  ): Promise<Float32Array> => {
    let dot = 0;
    for (let i = 0; i < count; i += 1) dot += dotLeft[i] * dotRight[i];
    const scalar = dotScale * dot + scalarScale * scalarBuffer[0];
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1)
      out[i] = input[i] + direction[i] * scalar;
    return out;
  },

  dotToScalarBuffer: async (
    a: Float32Array,
    b: Float32Array,
  ): Promise<Float32Array> => {
    let dot = 0;
    for (let i = 0; i < count; i += 1) dot += a[i] * b[i];
    return new Float32Array([dot]);
  },

  addScaledByScalarBuffer: async (
    input: Float32Array,
    direction: Float32Array,
    scalarBuffer: Float32Array,
    scalarScale: number,
  ): Promise<Float32Array> => {
    const scalar = scalarBuffer[0] * scalarScale;
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1)
      out[i] = input[i] + direction[i] * scalar;
    return out;
  },

  dot: async (a: Float32Array, b: Float32Array): Promise<number> => {
    let total = 0;
    for (let i = 0; i < count; i += 1) total += a[i] * b[i];
    return total;
  },

  dotPairWithRight: async (
    leftA: Float32Array,
    leftB: Float32Array,
    right: Float32Array,
  ): Promise<readonly [number, number]> => {
    let totalA = 0;
    let totalB = 0;
    for (let i = 0; i < count; i += 1) {
      const rightValue = right[i];
      totalA += leftA[i] * rightValue;
      totalB += leftB[i] * rightValue;
    }
    return [totalA, totalB] as const;
  },

  absSum: async (input: Float32Array): Promise<number> => {
    let total = 0;
    for (let i = 0; i < count; i += 1) total += Math.abs(input[i]);
    return total;
  },

  updateClamp: async (
    input: Float32Array,
    direction: Float32Array,
    learningRate: number,
  ): Promise<Float32Array> => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1)
      out[i] = clampUnit(input[i] - learningRate * direction[i]);
    return out;
  },

  adamUpdateClamp: async (
    input: Float32Array,
    grad: Float32Array,
    m: Float32Array,
    v: Float32Array,
    step: number,
    config: InputOptimizerConfig,
  ): Promise<Float32Array> => {
    const out = new Float32Array(count);
    const t = step + 1;
    const b1Correction = 1 - Math.pow(config.adamBeta1, t);
    const b2Correction = 1 - Math.pow(config.adamBeta2, t);
    for (let i = 0; i < count; i += 1) {
      const g = grad[i];
      m[i] = config.adamBeta1 * m[i] + (1 - config.adamBeta1) * g;
      v[i] = config.adamBeta2 * v[i] + (1 - config.adamBeta2) * g * g;
      const mHat = m[i] / b1Correction;
      const vHat = v[i] / b2Correction;
      out[i] = clampUnit(
        input[i] -
          config.learningRate * (mHat / (Math.sqrt(vHat) + config.adamEpsilon)),
      );
    }
    return out;
  },

  dispose: (): void => undefined,
});
