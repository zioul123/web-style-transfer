export type TensorShape = readonly [number, number, number, number]

export type TensorData = {
  shape: TensorShape
  values: Float32Array
}

const elementCount = (shape: TensorShape): number => shape[0] * shape[1] * shape[2] * shape[3]

const assertSameLength = (a: TensorData, b: TensorData): void => {
  if (elementCount(a.shape) !== elementCount(b.shape)) {
    throw new Error('Tensor sizes must match.')
  }
}

export const createTensor = (shape: TensorShape, values: number[]): TensorData => {
  const expected = elementCount(shape)
  if (values.length !== expected) {
    throw new Error(`Expected ${expected} values, got ${values.length}.`)
  }
  return { shape, values: new Float32Array(values) }
}

export const cpuAdd = (a: TensorData, b: TensorData): TensorData => {
  assertSameLength(a, b)
  const out: Float32Array = new Float32Array(a.values.length)
  for (let i = 0; i < out.length; i += 1) out[i] = a.values[i] + b.values[i]
  return { shape: a.shape, values: out }
}

export const cpuSub = (a: TensorData, b: TensorData): TensorData => {
  assertSameLength(a, b)
  const out: Float32Array = new Float32Array(a.values.length)
  for (let i = 0; i < out.length; i += 1) out[i] = a.values[i] - b.values[i]
  return { shape: a.shape, values: out }
}

export const cpuMul = (a: TensorData, b: TensorData): TensorData => {
  assertSameLength(a, b)
  const out: Float32Array = new Float32Array(a.values.length)
  for (let i = 0; i < out.length; i += 1) out[i] = a.values[i] * b.values[i]
  return { shape: a.shape, values: out }
}

export const cpuDiv = (a: TensorData, b: TensorData): TensorData => {
  assertSameLength(a, b)
  const out: Float32Array = new Float32Array(a.values.length)
  for (let i = 0; i < out.length; i += 1) out[i] = a.values[i] / b.values[i]
  return { shape: a.shape, values: out }
}

export const cpuClamp = (a: TensorData, min: number, max: number): TensorData => {
  const out: Float32Array = new Float32Array(a.values.length)
  for (let i = 0; i < out.length; i += 1) {
    out[i] = Math.min(max, Math.max(min, a.values[i]))
  }
  return { shape: a.shape, values: out }
}

export const cpuMse = (a: TensorData, b: TensorData): number => {
  assertSameLength(a, b)
  let sum = 0
  for (let i = 0; i < a.values.length; i += 1) {
    const d = a.values[i] - b.values[i]
    sum += d * d
  }
  return sum / a.values.length
}
