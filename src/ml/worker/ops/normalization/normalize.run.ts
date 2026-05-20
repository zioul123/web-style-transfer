import {
  makeNormalizeBackwardShader,
  makeNormalizeShader,
} from "./normalize.shader";

export const runNormalizeForward = async (
  gpuDevice: GPUDevice | null,
  runUnaryShader: (
    code: string,
    input: Float32Array,
    outCount: number,
    extraEntries?: GPUBindGroupEntry[],
  ) => Promise<Float32Array>,
  input: Float32Array,
  shape: readonly [number, number, number, number],
  mean: readonly number[],
  std: readonly number[],
  BUFFER_USAGE_STORAGE_COPY_DST: number,
  BUFFER_USAGE_UNIFORM_COPY_DST: number,
): Promise<Float32Array> => {
  const channels: number = shape[1];
  if (mean.length !== channels || std.length !== channels)
    throw new Error("Mean/std lengths must match input channels.");
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const meanBuffer: GPUBuffer = gpuDevice.createBuffer({
    size: channels * 4,
    usage: BUFFER_USAGE_STORAGE_COPY_DST,
  });
  const stdBuffer: GPUBuffer = gpuDevice.createBuffer({
    size: channels * 4,
    usage: BUFFER_USAGE_STORAGE_COPY_DST,
  });
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({
    size: 4 * 4,
    usage: BUFFER_USAGE_UNIFORM_COPY_DST,
  });
  gpuDevice.queue.writeBuffer(meanBuffer, 0, new Float32Array(mean));
  gpuDevice.queue.writeBuffer(stdBuffer, 0, new Float32Array(std));
  gpuDevice.queue.writeBuffer(
    uniformBuffer,
    0,
    new Uint32Array([channels, shape[2], shape[3]]),
  );
  return runUnaryShader(
    makeNormalizeShader(input.length),
    input,
    input.length,
    [
      { binding: 1, resource: { buffer: meanBuffer } },
      { binding: 2, resource: { buffer: stdBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer } },
    ],
  );
};

export const runNormalizeBackward = async (
  gpuDevice: GPUDevice | null,
  runUnaryShader: (
    code: string,
    input: Float32Array,
    outCount: number,
    extraEntries?: GPUBindGroupEntry[],
  ) => Promise<Float32Array>,
  gradOut: Float32Array,
  shape: readonly [number, number, number, number],
  std: readonly number[],
  BUFFER_USAGE_STORAGE_COPY_DST: number,
  BUFFER_USAGE_UNIFORM_COPY_DST: number,
): Promise<Float32Array> => {
  const channels = shape[1];
  if (std.length !== channels)
    throw new Error("Std length must match input channels.");
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const stdBuffer: GPUBuffer = gpuDevice.createBuffer({
    size: channels * 4,
    usage: BUFFER_USAGE_STORAGE_COPY_DST,
  });
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({
    size: 4 * 4,
    usage: BUFFER_USAGE_UNIFORM_COPY_DST,
  });
  gpuDevice.queue.writeBuffer(stdBuffer, 0, new Float32Array(std));
  gpuDevice.queue.writeBuffer(
    uniformBuffer,
    0,
    new Uint32Array([channels, shape[2], shape[3], 0]),
  );
  return runUnaryShader(
    makeNormalizeBackwardShader(gradOut.length),
    gradOut,
    gradOut.length,
    [
      { binding: 1, resource: { buffer: stdBuffer } },
      { binding: 2, resource: { buffer: uniformBuffer } },
    ],
  );
};
