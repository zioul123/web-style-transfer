import { makeConv2dReluShader, makeConv2dShader } from './conv2d.shader'

export const runConv2dForward = async (gpuDevice: GPUDevice | null, runUnaryShader: (code: string, input: Float32Array, outCount: number, extraEntries?: GPUBindGroupEntry[]) => Promise<Float32Array>, input: Float32Array, inputShape: readonly [number, number, number, number], weight: Float32Array, weightShape: readonly [number, number, number, number], bias: ArrayLike<number>, BUFFER_USAGE_STORAGE_COPY_DST: number, BUFFER_USAGE_UNIFORM_COPY_DST: number): Promise<Float32Array> => {
  const [batch, inChannels, height, width] = inputShape; const [outChannels, weightInChannels, kernelHeight, kernelWidth] = weightShape
  if (batch !== 1 || kernelHeight !== 3 || kernelWidth !== 3 || inChannels !== weightInChannels || bias.length !== outChannels) throw new Error('conv2d-forward only supports VGG-compatible params.')
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const outCount: number = outChannels * height * width
  const weightBuffer: GPUBuffer = gpuDevice.createBuffer({ size: weight.byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST }); const biasBuffer: GPUBuffer = gpuDevice.createBuffer({ size: outChannels * 4, usage: BUFFER_USAGE_STORAGE_COPY_DST }); const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 4 * 4, usage: BUFFER_USAGE_UNIFORM_COPY_DST })
  gpuDevice.queue.writeBuffer(weightBuffer, 0, weight); gpuDevice.queue.writeBuffer(biasBuffer, 0, new Float32Array(bias)); gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([inChannels, outChannels, height, width]))
  return runUnaryShader(makeConv2dShader(outCount), input, outCount, [{ binding: 1, resource: { buffer: weightBuffer } }, { binding: 2, resource: { buffer: biasBuffer } }, { binding: 3, resource: { buffer: uniformBuffer } }])
}

export const runConv2dReluForward = async (gpuDevice: GPUDevice | null, runUnaryShader: (code: string, input: Float32Array, outCount: number, extraEntries?: GPUBindGroupEntry[]) => Promise<Float32Array>, input: Float32Array, inputShape: readonly [number, number, number, number], weight: Float32Array, weightShape: readonly [number, number, number, number], bias: ArrayLike<number>, BUFFER_USAGE_STORAGE_COPY_DST: number, BUFFER_USAGE_UNIFORM_COPY_DST: number): Promise<Float32Array> => {
  const outChannels = weightShape[0]; const height = inputShape[2]; const width = inputShape[3]; const outputCount = outChannels * height * width
  if (gpuDevice === null) throw new Error('WebGPU is not initialized.')
  const weightBuffer: GPUBuffer = gpuDevice.createBuffer({ size: weight.byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST }); const biasBuffer: GPUBuffer = gpuDevice.createBuffer({ size: new Float32Array(bias).byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST }); const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 4 * 4, usage: BUFFER_USAGE_UNIFORM_COPY_DST })
  gpuDevice.queue.writeBuffer(weightBuffer, 0, weight); gpuDevice.queue.writeBuffer(biasBuffer, 0, new Float32Array(bias)); gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([inputShape[1], outChannels, height, width]))
  return runUnaryShader(makeConv2dReluShader(outputCount), input, outputCount, [{ binding: 1, resource: { buffer: weightBuffer } }, { binding: 2, resource: { buffer: biasBuffer } }, { binding: 3, resource: { buffer: uniformBuffer } }])
}
