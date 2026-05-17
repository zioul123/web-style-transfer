import type { TensorData } from '../../types'
import { assertSameLength } from '../tensor'

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
