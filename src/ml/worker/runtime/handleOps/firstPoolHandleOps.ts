/// <reference lib="webworker" />

import { makeConv2dBackwardInputShader, makeConv2dShader } from "../../ops/convolution/conv2d.shader";
import { makeGramBackwardShader, makeGramMatrixShader } from "../../ops/gram/gram.shader";
import { makeMseBackwardShader, makeMseDiffShader, makeReduceSumShader } from "../../ops/loss/mse.shader";
import { makeNormalizeBackwardShader, makeNormalizeShader } from "../../ops/normalization/normalize.shader";
import { makeMaxPool2dBackwardShader, makeMaxPool2dShader } from "../../ops/pooling/maxpool.shader";
import { makeReluBackwardShader, makeReluShader } from "../../ops/relu/relu.shader";
import { acquireReusableBuffer, releaseReusableBuffer } from "../bufferPool";
import { getGpuDevice } from "../deviceState";
import { BUFFER_USAGE_STORAGE_COPY_DST, BUFFER_USAGE_STORAGE_COPY_SRC } from "../gpuFlags";
import type { RuntimeTensorHandle, RuntimeTensorShape } from "../computeContext";
import { chainTensorBinaryOp, chainTensorScalarOp, readScalarBoundary, releaseTensorHandle, runUnaryOnHandle } from "../computeContext";

const requireDevice = (): GPUDevice => {
  const device = getGpuDevice();
  if (device === null) throw new Error("WebGPU is not initialized.");
  return device;
};

const shapeCount = (shape: RuntimeTensorShape): number => shape[0] * shape[1] * shape[2] * shape[3];
const toShape = (c: number, h: number, w: number): RuntimeTensorShape => [1, c, h, w];
const asHandle = (shape: RuntimeTensorShape, buffer: GPUBuffer): RuntimeTensorHandle => ({ kind: "gpu-tensor-handle", shape, elementCount: shapeCount(shape), byteLength: shapeCount(shape)*4, usage: BUFFER_USAGE_STORAGE_COPY_SRC, buffer, owner: "runtime" });

const runWithShape = (input: RuntimeTensorHandle, outShape: RuntimeTensorShape, code: string, extraEntries: GPUBindGroupEntry[] = []): RuntimeTensorHandle => {
  const device = requireDevice();
  const outBuffer = acquireReusableBuffer(device, shapeCount(outShape) * 4, BUFFER_USAGE_STORAGE_COPY_SRC);
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: input.buffer } }, ...extraEntries, { binding: extraEntries.length + 1, resource: { buffer: outBuffer } }] });
  const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(Math.ceil(shapeCount(outShape) / 64)); pass.end(); device.queue.submit([encoder.finish()]);
  return asHandle(outShape, outBuffer);
};

export const runNormalizeForwardHandle = async (input: RuntimeTensorHandle, mean: readonly number[], std: readonly number[]): Promise<RuntimeTensorHandle> => {
  const device = requireDevice(); const c = input.shape[1];
  const meanBuffer = acquireReusableBuffer(device, c * 4, BUFFER_USAGE_STORAGE_COPY_DST); const stdBuffer = acquireReusableBuffer(device, c * 4, BUFFER_USAGE_STORAGE_COPY_DST); const uniformBuffer = acquireReusableBuffer(device, 16, BUFFER_USAGE_STORAGE_COPY_DST);
  device.queue.writeBuffer(meanBuffer, 0, new Float32Array(mean)); device.queue.writeBuffer(stdBuffer, 0, new Float32Array(std)); device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([c, input.shape[2], input.shape[3], 0]));
  const out = runUnaryOnHandle(input, makeNormalizeShader(input.elementCount), [{ binding: 1, resource: { buffer: meanBuffer } }, { binding: 2, resource: { buffer: stdBuffer } }, { binding: 3, resource: { buffer: uniformBuffer } }]);
  releaseReusableBuffer(c * 4, BUFFER_USAGE_STORAGE_COPY_DST, meanBuffer); releaseReusableBuffer(c * 4, BUFFER_USAGE_STORAGE_COPY_DST, stdBuffer); releaseReusableBuffer(16, BUFFER_USAGE_STORAGE_COPY_DST, uniformBuffer); return out;
};
export const runNormalizeBackwardHandle = async (gradOut: RuntimeTensorHandle, std: readonly number[]): Promise<RuntimeTensorHandle> => {
  const device = requireDevice(); const c = gradOut.shape[1]; const stdBuffer = acquireReusableBuffer(device, c * 4, BUFFER_USAGE_STORAGE_COPY_DST); const uniformBuffer = acquireReusableBuffer(device, 16, BUFFER_USAGE_STORAGE_COPY_DST);
  device.queue.writeBuffer(stdBuffer, 0, new Float32Array(std)); device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([c, gradOut.shape[2], gradOut.shape[3], 0]));
  const out = runUnaryOnHandle(gradOut, makeNormalizeBackwardShader(gradOut.elementCount), [{ binding: 1, resource: { buffer: stdBuffer } }, { binding: 2, resource: { buffer: uniformBuffer } }]);
  releaseReusableBuffer(c * 4, BUFFER_USAGE_STORAGE_COPY_DST, stdBuffer); releaseReusableBuffer(16, BUFFER_USAGE_STORAGE_COPY_DST, uniformBuffer); return out;
};
export const runConv2dForwardHandle = async (input: RuntimeTensorHandle, weight: Float32Array, weightShape: readonly [number, number, number, number], bias: readonly number[]): Promise<RuntimeTensorHandle> => {
  const device = requireDevice(); const outShape = toShape(weightShape[0], input.shape[2], input.shape[3]);
  const weightBuffer = acquireReusableBuffer(device, weight.byteLength, BUFFER_USAGE_STORAGE_COPY_DST); const biasBuffer = acquireReusableBuffer(device, weightShape[0]*4, BUFFER_USAGE_STORAGE_COPY_DST); const uniformBuffer = acquireReusableBuffer(device, 16, BUFFER_USAGE_STORAGE_COPY_DST);
  device.queue.writeBuffer(weightBuffer, 0, weight); device.queue.writeBuffer(biasBuffer, 0, new Float32Array(bias)); device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([input.shape[1], weightShape[0], input.shape[2], input.shape[3]]));
  const out = runWithShape(input, outShape, makeConv2dShader(shapeCount(outShape)), [{ binding: 1, resource: { buffer: weightBuffer } }, { binding: 2, resource: { buffer: biasBuffer } }, { binding: 3, resource: { buffer: uniformBuffer } }]);
  releaseReusableBuffer(weight.byteLength, BUFFER_USAGE_STORAGE_COPY_DST, weightBuffer); releaseReusableBuffer(weightShape[0]*4, BUFFER_USAGE_STORAGE_COPY_DST, biasBuffer); releaseReusableBuffer(16, BUFFER_USAGE_STORAGE_COPY_DST, uniformBuffer); return out;
};
export const runConv2dBackwardInputHandle = async (inputShape: RuntimeTensorShape, gradOut: RuntimeTensorHandle, weight: Float32Array, weightShape: readonly [number, number, number, number]): Promise<RuntimeTensorHandle> => {
  const device = requireDevice(); const weightBuffer = acquireReusableBuffer(device, weight.byteLength, BUFFER_USAGE_STORAGE_COPY_DST); const uniformBuffer = acquireReusableBuffer(device, 16, BUFFER_USAGE_STORAGE_COPY_DST);
  device.queue.writeBuffer(weightBuffer, 0, weight); device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([inputShape[1], weightShape[0], inputShape[2], inputShape[3]]));
  const out = runWithShape(gradOut, inputShape, makeConv2dBackwardInputShader(shapeCount(inputShape)), [{ binding: 1, resource: { buffer: weightBuffer } }, { binding: 2, resource: { buffer: uniformBuffer } }]);
  releaseReusableBuffer(weight.byteLength, BUFFER_USAGE_STORAGE_COPY_DST, weightBuffer); releaseReusableBuffer(16, BUFFER_USAGE_STORAGE_COPY_DST, uniformBuffer); return out;
};
export const runReluForwardHandle = async (input: RuntimeTensorHandle): Promise<RuntimeTensorHandle> => runUnaryOnHandle(input, makeReluShader(input.elementCount));
export const runReluBackwardHandle = async (input: RuntimeTensorHandle, gradOut: RuntimeTensorHandle): Promise<RuntimeTensorHandle> => runUnaryOnHandle(input, makeReluBackwardShader(input.elementCount), [{ binding: 1, resource: { buffer: gradOut.buffer } }]);
export const runMaxPool2dForwardHandle = async (input: RuntimeTensorHandle): Promise<RuntimeTensorHandle> => {
  const c = input.shape[1]; const outH = Math.floor(input.shape[2] / 2); const outW = Math.floor(input.shape[3] / 2); const outShape = toShape(c, outH, outW);
  const device = requireDevice(); const uniformBuffer = acquireReusableBuffer(device, 20, BUFFER_USAGE_STORAGE_COPY_DST); device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([c, input.shape[2], input.shape[3], outH, outW]));
  const out = runWithShape(input, outShape, makeMaxPool2dShader(shapeCount(outShape)), [{ binding: 1, resource: { buffer: uniformBuffer } }]);
  releaseReusableBuffer(20, BUFFER_USAGE_STORAGE_COPY_DST, uniformBuffer); return out;
};
export const runMaxPool2dBackwardHandle = async (input: RuntimeTensorHandle, gradOut: RuntimeTensorHandle): Promise<RuntimeTensorHandle> => {
  const c = input.shape[1]; const outH = Math.floor(input.shape[2] / 2); const outW = Math.floor(input.shape[3] / 2); const device = requireDevice();
  const uniformBuffer = acquireReusableBuffer(device, 32, BUFFER_USAGE_STORAGE_COPY_DST); device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([c, input.shape[2], input.shape[3], outH, outW, 0, 0, 0]));
  const out = runUnaryOnHandle(input, makeMaxPool2dBackwardShader(input.elementCount), [{ binding: 1, resource: { buffer: gradOut.buffer } }, { binding: 2, resource: { buffer: uniformBuffer } }]);
  releaseReusableBuffer(32, BUFFER_USAGE_STORAGE_COPY_DST, uniformBuffer); return out;
};
export const runGramMatrixHandle = async (input: RuntimeTensorHandle): Promise<RuntimeTensorHandle> => {
  const c = input.shape[1]; const spatial = input.shape[2] * input.shape[3]; const outShape: RuntimeTensorShape = [1, 1, c, c]; const device = requireDevice(); const uniformBuffer = acquireReusableBuffer(device, 8, BUFFER_USAGE_STORAGE_COPY_DST);
  device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([c, spatial]));
  const out = runWithShape(input, outShape, makeGramMatrixShader(c*c), [{ binding: 1, resource: { buffer: uniformBuffer } }]);
  releaseReusableBuffer(8, BUFFER_USAGE_STORAGE_COPY_DST, uniformBuffer); return out;
};
export const runContentLossBackwardHandle = async (input: RuntimeTensorHandle, target: RuntimeTensorHandle): Promise<RuntimeTensorHandle> => runUnaryOnHandle(input, makeMseBackwardShader(input.elementCount), [{ binding: 1, resource: { buffer: target.buffer } }]);
export const runStyleLossBackwardHandle = async (input: RuntimeTensorHandle, target: RuntimeTensorHandle): Promise<RuntimeTensorHandle> => {
  const inputGram = await runGramMatrixHandle(input); const targetGram = await runGramMatrixHandle(target); const gramGrad = await runContentLossBackwardHandle(inputGram, targetGram);
  const c = input.shape[1]; const spatial = input.shape[2] * input.shape[3]; const norm = input.shape[0] * input.shape[1] * input.shape[2] * input.shape[3]; const device = requireDevice(); const uniformBuffer = acquireReusableBuffer(device, 16, BUFFER_USAGE_STORAGE_COPY_DST);
  device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([c, spatial, norm, 0]));
  const out = runUnaryOnHandle(input, makeGramBackwardShader(input.elementCount), [{ binding: 1, resource: { buffer: gramGrad.buffer } }, { binding: 2, resource: { buffer: uniformBuffer } }]);
  releaseReusableBuffer(16, BUFFER_USAGE_STORAGE_COPY_DST, uniformBuffer); releaseTensorHandle(inputGram); releaseTensorHandle(targetGram); releaseTensorHandle(gramGrad); return out;
};
export const runMseHandle = async (a: RuntimeTensorHandle, b: RuntimeTensorHandle): Promise<RuntimeTensorHandle> => {
  const diff = runUnaryOnHandle(a, makeMseDiffShader(a.elementCount), [{ binding: 1, resource: { buffer: b.buffer } }]);
  let current = diff; let currentCount = current.elementCount;
  while (currentCount > 1) {
    const nextCount = Math.ceil(currentCount / 64); const nextShape: RuntimeTensorShape = [1, 1, 1, nextCount];
    const next = runWithShape(current, nextShape, makeReduceSumShader(currentCount));
    releaseTensorHandle(current); current = next; currentCount = nextCount;
  }
  const invCount = 1 / a.elementCount;
  const scaled = await chainTensorScalarOp(current, "mul", invCount);
  releaseTensorHandle(current);
  return scaled;
};
export const readMseScalar = async (a: RuntimeTensorHandle, b: RuntimeTensorHandle): Promise<number> => {
  const mseHandle = await runMseHandle(a, b); const scalar = await readScalarBoundary(mseHandle); releaseTensorHandle(mseHandle); return scalar;
};
export const scaleHandle = async (input: RuntimeTensorHandle, scalar: number): Promise<RuntimeTensorHandle> => chainTensorScalarOp(input, "mul", scalar);
export const addHandles = async (a: RuntimeTensorHandle, b: RuntimeTensorHandle): Promise<RuntimeTensorHandle> => chainTensorBinaryOp(a, b, "add");
