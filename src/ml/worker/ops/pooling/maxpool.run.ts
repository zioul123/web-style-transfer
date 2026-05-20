import {
  makeMaxPool2dBackwardShader,
  makeMaxPool2dShader,
} from "./maxpool.shader";

export const runMaxPool2dForward = async (
  gpuDevice: GPUDevice | null,
  runUnaryShader: (
    code: string,
    input: Float32Array,
    outCount: number,
    extraEntries?: GPUBindGroupEntry[],
  ) => Promise<Float32Array>,
  input: Float32Array,
  shape: readonly [number, number, number, number],
  BUFFER_USAGE_UNIFORM_COPY_DST: number,
): Promise<Float32Array> => {
  if (shape[0] !== 1)
    throw new Error("maxpool2d-forward currently expects batch size 1.");
  const channels: number = shape[1];
  const inHeight: number = shape[2];
  const inWidth: number = shape[3];
  const outHeight: number = Math.floor(inHeight / 2);
  const outWidth: number = Math.floor(inWidth / 2);
  const outCount: number = channels * outHeight * outWidth;
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({
    size: 5 * 4,
    usage: BUFFER_USAGE_UNIFORM_COPY_DST,
  });
  gpuDevice.queue.writeBuffer(
    uniformBuffer,
    0,
    new Uint32Array([channels, inHeight, inWidth, outHeight, outWidth]),
  );
  return runUnaryShader(makeMaxPool2dShader(outCount), input, outCount, [
    { binding: 1, resource: { buffer: uniformBuffer } },
  ]);
};

export const runMaxPool2dBackward = async (
  gpuDevice: GPUDevice | null,
  runUnaryShader: (
    code: string,
    input: Float32Array,
    outCount: number,
    extraEntries?: GPUBindGroupEntry[],
  ) => Promise<Float32Array>,
  input: Float32Array,
  shape: readonly [number, number, number, number],
  gradOut: Float32Array,
  BUFFER_USAGE_STORAGE_COPY_DST: number,
  BUFFER_USAGE_UNIFORM_COPY_DST: number,
): Promise<Float32Array> => {
  if (shape[0] !== 1)
    throw new Error("maxpool2d-backward currently expects batch size 1.");
  const channels = shape[1];
  const inHeight = shape[2];
  const inWidth = shape[3];
  const outHeight = Math.floor(inHeight / 2);
  const outWidth = Math.floor(inWidth / 2);
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const gradOutBuffer: GPUBuffer = gpuDevice.createBuffer({
    size: gradOut.byteLength,
    usage: BUFFER_USAGE_STORAGE_COPY_DST,
  });
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({
    size: 8 * 4,
    usage: BUFFER_USAGE_UNIFORM_COPY_DST,
  });
  gpuDevice.queue.writeBuffer(gradOutBuffer, 0, gradOut);
  gpuDevice.queue.writeBuffer(
    uniformBuffer,
    0,
    new Uint32Array([
      channels,
      inHeight,
      inWidth,
      outHeight,
      outWidth,
      0,
      0,
      0,
    ]),
  );
  return runUnaryShader(
    makeMaxPool2dBackwardShader(input.length),
    input,
    input.length,
    [
      { binding: 1, resource: { buffer: gradOutBuffer } },
      { binding: 2, resource: { buffer: uniformBuffer } },
    ],
  );
};
