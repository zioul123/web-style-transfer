import {
  makeConv2dBackwardInputShader,
  makeConv2dBackwardInputTransposedWeightShader,
  makeConv2dReluSpatialVec4Shader,
  makeConv2dReluShader,
  makeConv2dSpatialVec4Shader,
  makeConv2dShader,
} from "./conv2d.shader";
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

export type Conv2dForwardStaticBuffers = {
  weightBuffer: GPUBuffer;
  biasBuffer: GPUBuffer;
};

export type Conv2dBackwardStaticBuffers = {
  weightBuffer: GPUBuffer;
  transposedWeightBuffer?: GPUBuffer;
};

export type ConvForwardKernelVariant = "scalar" | "spatial-vec4" | "tiled-spatial";
export type ConvBackwardInputKernelVariant =
  | "scalar"
  | "spatial-vec4"
  | "transposed-weight-spatial-vec4";

const assertSupportedForwardKernelVariant = (
  kernelVariant: ConvForwardKernelVariant,
): void => {
  if (kernelVariant === "tiled-spatial") {
    throw new Error("convForwardKernel=tiled-spatial is not implemented yet.");
  }
};

const assertSupportedBackwardInputKernelVariant = (
  kernelVariant: ConvBackwardInputKernelVariant,
): void => {
  if (kernelVariant === "spatial-vec4") {
    throw new Error("convBackwardInputKernel=spatial-vec4 is not implemented yet.");
  }
};

const validateConv2dParams = (
  inputShape: readonly [number, number, number, number],
  weightShape: readonly [number, number, number, number],
): { inChannels: number; outChannels: number; height: number; width: number } => {
  const [batch, inChannels, height, width] = inputShape;
  const [outChannels, weightInChannels, kernelHeight, kernelWidth] = weightShape;
  if (batch !== 1 || kernelHeight !== 3 || kernelWidth !== 3 || inChannels !== weightInChannels) {
    throw new Error("conv2d only supports VGG-compatible params.");
  }
  return { inChannels, outChannels, height, width };
};

const destroyForwardLocalBuffers = (
  staticBuffers: Conv2dForwardStaticBuffers | undefined,
  weightBuffer: GPUBuffer,
  biasBuffer: GPUBuffer,
  uniformBuffer: GPUBuffer,
): void => {
  uniformBuffer.destroy();
  if (staticBuffers === undefined) {
    weightBuffer.destroy();
    biasBuffer.destroy();
  }
};

export const runConv2dForwardBuffer = async (
  input: GpuBufferRef,
  inputShape: readonly [number, number, number, number],
  weight: Float32Array,
  weightShape: readonly [number, number, number, number],
  bias: ArrayLike<number>,
  staticBuffers?: Conv2dForwardStaticBuffers,
  kernelVariant: ConvForwardKernelVariant = "scalar",
): Promise<GpuBufferRef> => {
  assertSupportedForwardKernelVariant(kernelVariant);
  const { inChannels, outChannels, height, width } = validateConv2dParams(inputShape, weightShape);
  if (bias.length !== outChannels) throw new Error("conv2d-forward only supports VGG-compatible params.");
  const gpuDevice: GPUDevice | null = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const outCount: number = outChannels * height * width;
  const weightBuffer: GPUBuffer = staticBuffers?.weightBuffer ?? gpuDevice.createBuffer({ size: weight.byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST });
  const biasBuffer: GPUBuffer = staticBuffers?.biasBuffer ?? gpuDevice.createBuffer({ size: outChannels * Float32Array.BYTES_PER_ELEMENT, usage: BUFFER_USAGE_STORAGE_COPY_DST });
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 4 * Uint32Array.BYTES_PER_ELEMENT, usage: BUFFER_USAGE_UNIFORM_COPY_DST });
  if (staticBuffers === undefined) {
    gpuDevice.queue.writeBuffer(weightBuffer, 0, weight);
    gpuDevice.queue.writeBuffer(biasBuffer, 0, new Float32Array(bias));
  }
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([inChannels, outChannels, height, width]));
  const useSpatialVec4: boolean = kernelVariant === "spatial-vec4" && outCount >= 4;
  const outputBuffer: GPUBuffer = (() => {
    try {
      if (useSpatialVec4) {
        return runUnaryShaderToBufferWithDispatch(
          gpuDevice,
          makeConv2dSpatialVec4Shader(outCount),
          input.buffer,
          outCount,
          Math.ceil(outCount / 4),
          [
            { binding: 1, resource: { buffer: weightBuffer } },
            { binding: 2, resource: { buffer: biasBuffer } },
            { binding: 3, resource: { buffer: uniformBuffer } },
          ],
        );
      }
      return runUnaryShaderToBuffer(gpuDevice, makeConv2dShader(outCount), input.buffer, outCount, [
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: biasBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer } },
      ]);
    } finally {
      destroyForwardLocalBuffers(staticBuffers, weightBuffer, biasBuffer, uniformBuffer);
    }
  })();
  return ownedBuffer(outputBuffer);
};

export const runConv2dForwardReadback = async (
  _gpuDevice: GPUDevice | null,
  _runUnaryShader: (code: string, input: Float32Array, outCount: number, extraEntries?: GPUBindGroupEntry[]) => Promise<Float32Array>,
  input: Float32Array,
  inputShape: readonly [number, number, number, number],
  weight: Float32Array,
  weightShape: readonly [number, number, number, number],
  bias: ArrayLike<number>,
): Promise<Float32Array> => {
  const inputBuffer = uploadToOwnedBuffer(getGpuDevice(), input);
  let outputBuffer: GpuBufferRef | null = null;
  try {
    const outCount: number = weightShape[0] * inputShape[2] * inputShape[3];
    outputBuffer = await runConv2dForwardBuffer(inputBuffer, inputShape, weight, weightShape, bias);
    return await readGpuBufferToArray(getGpuDevice(), outputBuffer.buffer, outCount);
  } finally {
    releaseOwnedBuffer(inputBuffer);
    if (outputBuffer !== null) releaseOwnedBuffer(outputBuffer);
  }
};

export const runConv2dReluForwardReadback = async (
  gpuDevice: GPUDevice | null,
  runUnaryShader: (code: string, input: Float32Array, outCount: number, extraEntries?: GPUBindGroupEntry[]) => Promise<Float32Array>,
  input: Float32Array,
  inputShape: readonly [number, number, number, number],
  weight: Float32Array,
  weightShape: readonly [number, number, number, number],
  bias: ArrayLike<number>,
  BUFFER_USAGE_STORAGE_COPY_DST_VALUE: number,
  BUFFER_USAGE_UNIFORM_COPY_DST_VALUE: number,
): Promise<Float32Array> => {
  const outChannels: number = weightShape[0];
  const height: number = inputShape[2];
  const width: number = inputShape[3];
  const outputCount: number = outChannels * height * width;
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const weightBuffer: GPUBuffer = gpuDevice.createBuffer({ size: weight.byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST_VALUE });
  const biasBuffer: GPUBuffer = gpuDevice.createBuffer({ size: new Float32Array(bias).byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST_VALUE });
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 4 * Uint32Array.BYTES_PER_ELEMENT, usage: BUFFER_USAGE_UNIFORM_COPY_DST_VALUE });
  gpuDevice.queue.writeBuffer(weightBuffer, 0, weight);
  gpuDevice.queue.writeBuffer(biasBuffer, 0, new Float32Array(bias));
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([inputShape[1], outChannels, height, width]));
  try {
    return await runUnaryShader(makeConv2dReluShader(outputCount), input, outputCount, [
      { binding: 1, resource: { buffer: weightBuffer } },
      { binding: 2, resource: { buffer: biasBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer } },
    ]);
  } finally {
    weightBuffer.destroy();
    biasBuffer.destroy();
    uniformBuffer.destroy();
  }
};

export const runConv2dReluForwardBuffer = async (
  input: GpuBufferRef,
  inputShape: readonly [number, number, number, number],
  weight: Float32Array,
  weightShape: readonly [number, number, number, number],
  bias: ArrayLike<number>,
  staticBuffers?: Conv2dForwardStaticBuffers,
  kernelVariant: ConvForwardKernelVariant = "scalar",
): Promise<GpuBufferRef> => {
  assertSupportedForwardKernelVariant(kernelVariant);
  const { inChannels, outChannels, height, width } = validateConv2dParams(inputShape, weightShape);
  if (bias.length !== outChannels) throw new Error("conv2d-relu-forward only supports VGG-compatible params.");
  const gpuDevice: GPUDevice | null = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const outCount: number = outChannels * height * width;
  const weightBuffer: GPUBuffer = staticBuffers?.weightBuffer ?? gpuDevice.createBuffer({ size: weight.byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST });
  const biasBuffer: GPUBuffer = staticBuffers?.biasBuffer ?? gpuDevice.createBuffer({ size: outChannels * Float32Array.BYTES_PER_ELEMENT, usage: BUFFER_USAGE_STORAGE_COPY_DST });
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 4 * Uint32Array.BYTES_PER_ELEMENT, usage: BUFFER_USAGE_UNIFORM_COPY_DST });
  if (staticBuffers === undefined) {
    gpuDevice.queue.writeBuffer(weightBuffer, 0, weight);
    gpuDevice.queue.writeBuffer(biasBuffer, 0, new Float32Array(bias));
  }
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([inChannels, outChannels, height, width]));
  const useSpatialVec4: boolean = kernelVariant === "spatial-vec4" && outCount >= 4;
  const outputBuffer: GPUBuffer = (() => {
    try {
      if (useSpatialVec4) {
        return runUnaryShaderToBufferWithDispatch(
          gpuDevice,
          makeConv2dReluSpatialVec4Shader(outCount),
          input.buffer,
          outCount,
          Math.ceil(outCount / 4),
          [
            { binding: 1, resource: { buffer: weightBuffer } },
            { binding: 2, resource: { buffer: biasBuffer } },
            { binding: 3, resource: { buffer: uniformBuffer } },
          ],
        );
      }
      return runUnaryShaderToBuffer(gpuDevice, makeConv2dReluShader(outCount), input.buffer, outCount, [
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: biasBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer } },
      ]);
    } finally {
      destroyForwardLocalBuffers(staticBuffers, weightBuffer, biasBuffer, uniformBuffer);
    }
  })();
  return ownedBuffer(outputBuffer);
};

export const runConv2dBackwardInputBuffer = async (
  gradOut: GpuBufferRef,
  inputShape: readonly [number, number, number, number],
  weight: Float32Array,
  weightShape: readonly [number, number, number, number],
  staticBuffers?: Conv2dBackwardStaticBuffers,
  kernelVariant: ConvBackwardInputKernelVariant = "scalar",
): Promise<GpuBufferRef> => {
  assertSupportedBackwardInputKernelVariant(kernelVariant);
  const { inChannels, outChannels, height, width } = validateConv2dParams(inputShape, weightShape);
  const gpuDevice: GPUDevice | null = getGpuDevice();
  if (gpuDevice === null) throw new Error("WebGPU is not initialized.");
  const outCount: number = inChannels * height * width;
  const weightBuffer: GPUBuffer = staticBuffers?.weightBuffer ?? gpuDevice.createBuffer({ size: weight.byteLength, usage: BUFFER_USAGE_STORAGE_COPY_DST });
  let transposedWeightBuffer: GPUBuffer | undefined = staticBuffers?.transposedWeightBuffer;
  let ownsTransposedWeightBuffer = false;
  const uniformBuffer: GPUBuffer = gpuDevice.createBuffer({ size: 4 * Uint32Array.BYTES_PER_ELEMENT, usage: BUFFER_USAGE_UNIFORM_COPY_DST });
  if (staticBuffers === undefined) {
    gpuDevice.queue.writeBuffer(weightBuffer, 0, weight);
  }
  if (kernelVariant === "transposed-weight-spatial-vec4" && transposedWeightBuffer === undefined) {
    const transposed = new Float32Array(weight.length);
    for (let oc = 0; oc < outChannels; oc += 1) {
      for (let ic = 0; ic < inChannels; ic += 1) {
        for (let kh = 0; kh < 3; kh += 1) {
          for (let kw = 0; kw < 3; kw += 1) {
            const sourceIndex = ((oc * inChannels + ic) * 9) + kh * 3 + kw;
            const targetIndex = (((ic * 3 + kh) * 3 + kw) * outChannels) + oc;
            transposed[targetIndex] = weight[sourceIndex];
          }
        }
      }
    }
    transposedWeightBuffer = gpuDevice.createBuffer({
      size: transposed.byteLength,
      usage: BUFFER_USAGE_STORAGE_COPY_DST,
    });
    ownsTransposedWeightBuffer = true;
    gpuDevice.queue.writeBuffer(transposedWeightBuffer, 0, transposed);
  }
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([inChannels, outChannels, height, width]));
  const shouldUseTransposed: boolean = kernelVariant === "transposed-weight-spatial-vec4";
  const selectedWeightBuffer: GPUBuffer =
    shouldUseTransposed && transposedWeightBuffer !== undefined
      ? transposedWeightBuffer
      : weightBuffer;
  const shaderCode: string = shouldUseTransposed
    ? makeConv2dBackwardInputTransposedWeightShader(outCount)
    : makeConv2dBackwardInputShader(outCount);
  const outputBuffer: GPUBuffer = (() => {
    try {
      return runUnaryShaderToBuffer(gpuDevice, shaderCode, gradOut.buffer, outCount, [
        { binding: 1, resource: { buffer: selectedWeightBuffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ]);
    } finally {
      uniformBuffer.destroy();
      if (staticBuffers === undefined) weightBuffer.destroy();
      if (ownsTransposedWeightBuffer) transposedWeightBuffer?.destroy();
    }
  })();
  return ownedBuffer(outputBuffer);
};

export const runConv2dBackwardInputReadback = async (
  _gpuDevice: GPUDevice | null,
  _runUnaryShader: (code: string, input: Float32Array, outCount: number, extraEntries?: GPUBindGroupEntry[]) => Promise<Float32Array>,
  inputShape: readonly [number, number, number, number],
  gradOut: Float32Array,
  weight: Float32Array,
  weightShape: readonly [number, number, number, number],
): Promise<Float32Array> => {
  const gradOutBuffer = uploadToOwnedBuffer(getGpuDevice(), gradOut);
  let outputBuffer: GpuBufferRef | null = null;
  try {
    outputBuffer = await runConv2dBackwardInputBuffer(gradOutBuffer, inputShape, weight, weightShape);
    const outCount: number = inputShape[1] * inputShape[2] * inputShape[3];
    return await readGpuBufferToArray(getGpuDevice(), outputBuffer.buffer, outCount);
  } finally {
    releaseOwnedBuffer(gradOutBuffer);
    if (outputBuffer !== null) releaseOwnedBuffer(outputBuffer);
  }
};
