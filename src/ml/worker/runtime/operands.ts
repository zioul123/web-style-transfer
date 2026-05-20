import type { WorkerRequest, WorkerTensorOperand } from '../../../types'
import { createTensor } from '../../index'

export const isBinaryTensorOpPayload = (
  payload: WorkerRequest,
): payload is Extract<WorkerRequest, { type: 'tensor-op'; op: 'add' | 'sub' | 'mul' | 'div' | 'mse'; a: WorkerTensorOperand; b: WorkerTensorOperand }> =>
  payload.type === 'tensor-op'
  && (payload.op === 'add' || payload.op === 'sub' || payload.op === 'mul' || payload.op === 'div' || payload.op === 'mse')

export const getTensorFromOperand = (operand: WorkerTensorOperand): { ok: true; values: Float32Array } | { ok: false } => {
  if (operand.kind !== 'tensor') return { ok: false }
  const tensor = createTensor(operand.tensor.shape, operand.tensor.values)
  return { ok: true, values: tensor.values }
}

export const getValuesFromOperand = (operand: WorkerTensorOperand): Float32Array => {
  if (operand.kind === 'scalar') return new Float32Array([operand.scalar])
  return createTensor(operand.tensor.shape, operand.tensor.values).values
}
