import { makeGramBackwardShader, makeGramMatrixShader } from './gram.shader'

export const runGramMatrix = async (gpuDevice: GPUDevice | null, runUnaryShader: (code: string, input: Float32Array, outCount: number, extraEntries?: GPUBindGroupEntry[]) => Promise<Float32Array>, input: Float32Array, shape: readonly [number, number, number, number], BUFFER_USAGE_UNIFORM_COPY_DST: number): Promise<Float32Array> => {
  if (shape[0] !== 1) throw new Error('gram-matrix expects batch size 1.')
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const channels = shape[1]
  const spatial = shape[2] * shape[3]
  const outCount = channels * channels
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 2 * 4, usage: BUFFER_USAGE_UNIFORM_COPY_DST })
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([channels, spatial]))
  return runUnaryShader(makeGramMatrixShader(outCount), input, outCount, [{ binding: 1, resource: { buffer: uniformBuffer } }])
}

export const runGramBackward = async (gpuDevice: GPUDevice | null, runUnaryShader: (code: string, input: Float32Array, outCount: number, extraEntries?: GPUBindGroupEntry[]) => Promise<Float32Array>, input: Float32Array, shape: readonly [number, number, number, number], gradOut: Float32Array, BUFFER_USAGE_STORAGE_COPY_DST: number, BUFFER_USAGE_UNIFORM_COPY_DST: number): Promise<Float32Array> => {
  const channels = shape[1]
  const spatial = shape[2] * shape[3]
  const norm = shape[0] * shape[1] * shape[2] * shape[3]
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const gradOutBuffer: GPUBuffer = gpuDevice.createBuffer({ size: gradOut.byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST })
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 4 * 4, usage: BUFFER_USAGE_UNIFORM_COPY_DST })
  gpuDevice.queue.writeBuffer(gradOutBuffer, 0, gradOut)
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([channels, spatial, norm, 0]))
  return runUnaryShader(makeGramBackwardShader(input.length), input, input.length, [{ binding: 1, resource: { buffer: gradOutBuffer } }, { binding: 2, resource: { buffer: uniformBuffer } }])
}
