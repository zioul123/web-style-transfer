import { makeMseBackwardShader } from "./mse.shader";

export const runContentLossBackward = async (
  gpuDevice: GPUDevice | null,
  runUnaryShader: (
    code: string,
    input: Float32Array,
    outCount: number,
    extraEntries?: GPUBindGroupEntry[],
  ) => Promise<Float32Array>,
  input: Float32Array,
  target: Float32Array,
  BUFFER_USAGE_STORAGE_COPY_DST: number,
): Promise<Float32Array> => {
  if (input.length !== target.length) return new Float32Array(input.length);
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const targetBuffer: GPUBuffer = gpuDevice.createBuffer({
    size: target.byteLength,
    usage: BUFFER_USAGE_STORAGE_COPY_DST,
  });
  gpuDevice.queue.writeBuffer(targetBuffer, 0, target);
  return runUnaryShader(
    makeMseBackwardShader(input.length),
    input,
    input.length,
    [{ binding: 1, resource: { buffer: targetBuffer } }],
  );
};
