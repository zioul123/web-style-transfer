import { makeReluBackwardShader, makeReluShader } from './relu.shader'

export const runReluForward = async (runUnaryShader: (code: string, input: Float32Array, outCount: number, extraEntries?: GPUBindGroupEntry[]) => Promise<Float32Array>, input: Float32Array): Promise<Float32Array> => runUnaryShader(makeReluShader(input.length), input, input.length)

export const runReluBackward = async (
  gpuDevice: GPUDevice | null,
  runUnaryShader: (code: string, input: Float32Array, outCount: number, extraEntries?: GPUBindGroupEntry[]) => Promise<Float32Array>,
  input: Float32Array,
  gradOut: Float32Array,
): Promise<Float32Array> => {
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const gradOutBuffer: GPUBuffer = gpuDevice.createBuffer({ size: gradOut.byteLength, usage: 0x0080 | 0x0008 })
  gpuDevice.queue.writeBuffer(gradOutBuffer, 0, gradOut)
  return runUnaryShader(makeReluBackwardShader(input.length), input, input.length, [{ binding: 1, resource: { buffer: gradOutBuffer } }])
}
