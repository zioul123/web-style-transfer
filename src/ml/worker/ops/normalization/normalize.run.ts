import {
  makeNormalizeBackwardVec4Shader,
  makeNormalizeBackwardShader,
  makeNormalizeVec4Shader,
  makeNormalizeShader,
} from "./normalize.shader";
import {
  ownedBuffer,
  readGpuBufferToArray,
  releaseOwnedBuffer,
  runUnaryShaderToBuffer,
  runUnaryShaderToBufferWithDispatch,
  uploadToOwnedBuffer,
  type GpuBufferRef,
} from "../../runtime/bufferKernels";
import { getGpuDevice } from "../../runtime/deviceState";
import {
  BUFFER_USAGE_STORAGE_COPY_DST,
  BUFFER_USAGE_UNIFORM_COPY_DST,
} from "../../runtime/gpuFlags";

export const runNormalizeForwardBuffer = async (
  input: GpuBufferRef,
  shape: readonly [number, number, number, number],
  mean: readonly number[],
  std: readonly number[],
  useVec4: boolean = false,
): Promise<GpuBufferRef> => {
  const channels: number = shape[1];
  if (mean.length !== channels || std.length !== channels)
    throw new Error("Mean/std lengths must match input channels.");
  const gpuDevice: GPUDevice | null = getGpuDevice();
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
  const count = shape[1] * shape[2] * shape[3];
  try {
    return ownedBuffer(
      useVec4 && count % 4 === 0
        ? runUnaryShaderToBufferWithDispatch(
            gpuDevice,
            makeNormalizeVec4Shader(count / 4),
            input.buffer,
            count,
            count / 4,
            [
              { binding: 1, resource: { buffer: meanBuffer } },
              { binding: 2, resource: { buffer: stdBuffer } },
              { binding: 3, resource: { buffer: uniformBuffer } },
            ],
          )
        : runUnaryShaderToBuffer(
            gpuDevice,
            makeNormalizeShader(count),
            input.buffer,
            count,
            [
              { binding: 1, resource: { buffer: meanBuffer } },
              { binding: 2, resource: { buffer: stdBuffer } },
              { binding: 3, resource: { buffer: uniformBuffer } },
            ],
          ),
    );
  } finally {
    meanBuffer.destroy();
    stdBuffer.destroy();
    uniformBuffer.destroy();
  }
};

export const runNormalizeForward = async (
  _gpuDevice: GPUDevice | null,
  _runUnaryShader: (
    code: string,
    input: Float32Array,
    outCount: number,
    extraEntries?: GPUBindGroupEntry[],
  ) => Promise<Float32Array>,
  input: Float32Array,
  shape: readonly [number, number, number, number],
  mean: readonly number[],
  std: readonly number[],
): Promise<Float32Array> => {
  const inBuffer = uploadToOwnedBuffer(getGpuDevice(), input);
  let outBuffer: GpuBufferRef | null = null;
  try {
    outBuffer = await runNormalizeForwardBuffer(inBuffer, shape, mean, std);
    return await readGpuBufferToArray(
      getGpuDevice(),
      outBuffer.buffer,
      input.length,
    );
  } finally {
    releaseOwnedBuffer(inBuffer);
    if (outBuffer !== null) releaseOwnedBuffer(outBuffer);
  }
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
  try {
    return await runUnaryShader(
      makeNormalizeBackwardShader(gradOut.length),
      gradOut,
      gradOut.length,
      [
        { binding: 1, resource: { buffer: stdBuffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    );
  } finally {
    stdBuffer.destroy();
    uniformBuffer.destroy();
  }
};

export const runNormalizeBackwardBuffer = async (
  gradOut: GpuBufferRef,
  shape: readonly [number, number, number, number],
  std: readonly number[],
  useVec4: boolean = false,
): Promise<GpuBufferRef> => {
  const channels: number = shape[1];
  if (std.length !== channels)
    throw new Error("Std length must match input channels.");
  const gpuDevice: GPUDevice | null = getGpuDevice();
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
  const count = shape[1] * shape[2] * shape[3];
  try {
    return ownedBuffer(
      useVec4 && count % 4 === 0
        ? runUnaryShaderToBufferWithDispatch(
            gpuDevice,
            makeNormalizeBackwardVec4Shader(count / 4),
            gradOut.buffer,
            count,
            count / 4,
            [
              { binding: 1, resource: { buffer: stdBuffer } },
              { binding: 2, resource: { buffer: uniformBuffer } },
            ],
          )
        : runUnaryShaderToBuffer(
            gpuDevice,
            makeNormalizeBackwardShader(count),
            gradOut.buffer,
            count,
            [
              { binding: 1, resource: { buffer: stdBuffer } },
              { binding: 2, resource: { buffer: uniformBuffer } },
            ],
          ),
    );
  } finally {
    stdBuffer.destroy();
    uniformBuffer.destroy();
  }
};
