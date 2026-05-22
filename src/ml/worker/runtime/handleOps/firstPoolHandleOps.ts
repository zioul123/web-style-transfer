/// <reference lib="webworker" />

import {
  runConv2dBackwardInput,
  runConv2dForward,
} from "../../ops/convolution/conv2d.run";
import { runGramMatrix } from "../../ops/gram/gram.run";
import { runContentLossBackward } from "../../ops/loss/contentLoss.run";
import { runStyleLossBackward } from "../../ops/loss/styleLoss.run";
import {
  runNormalizeBackward,
  runNormalizeForward,
} from "../../ops/normalization/normalize.run";
import {
  runMaxPool2dBackward,
  runMaxPool2dForward,
} from "../../ops/pooling/maxpool.run";
import { runReluBackward, runReluForward } from "../../ops/relu/relu.run";
import {
  acquireTensorHandleFromCpu,
  readTensorHandleToCpu,
  runUnary,
  type RuntimeTensorHandle,
  type RuntimeTensorShape,
} from "../computeContext";
import { getGpuDevice } from "../deviceState";
import {
  BUFFER_USAGE_STORAGE_COPY_DST,
  BUFFER_USAGE_UNIFORM_COPY_DST,
} from "../gpuFlags";

const fromHandle = async (handle: RuntimeTensorHandle): Promise<Float32Array> =>
  await readTensorHandleToCpu(handle);
const toHandle = (
  shape: RuntimeTensorShape,
  values: Float32Array,
): RuntimeTensorHandle => acquireTensorHandleFromCpu(shape, values);

export const normalizeForwardHandle = async (
  input: RuntimeTensorHandle,
  shape: RuntimeTensorShape,
  mean: readonly [number, number, number],
  std: readonly [number, number, number],
): Promise<RuntimeTensorHandle> =>
  toHandle(
    shape,
    await runNormalizeForward(
      getGpuDevice(),
      runUnary,
      await fromHandle(input),
      shape,
      mean,
      std,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    ),
  );

export const conv2dForwardHandle = async (
  input: RuntimeTensorHandle,
  inputShape: RuntimeTensorShape,
  weightValues: Float32Array,
  weightShape: readonly [number, number, number, number],
  bias: number[],
  outputShape: RuntimeTensorShape,
): Promise<RuntimeTensorHandle> =>
  toHandle(
    outputShape,
    await runConv2dForward(
      getGpuDevice(),
      runUnary,
      await fromHandle(input),
      inputShape,
      weightValues,
      weightShape,
      bias,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    ),
  );

export const reluForwardHandle = async (
  input: RuntimeTensorHandle,
  shape: RuntimeTensorShape,
): Promise<RuntimeTensorHandle> =>
  toHandle(shape, await runReluForward(runUnary, await fromHandle(input)));
export const maxPoolForwardHandle = async (
  input: RuntimeTensorHandle,
  inputShape: RuntimeTensorShape,
  outputShape: RuntimeTensorShape,
): Promise<RuntimeTensorHandle> =>
  toHandle(
    outputShape,
    await runMaxPool2dForward(
      getGpuDevice(),
      runUnary,
      await fromHandle(input),
      inputShape,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    ),
  );
export const gramHandle = async (
  input: RuntimeTensorHandle,
  inputShape: RuntimeTensorShape,
  gramShape: RuntimeTensorShape,
): Promise<RuntimeTensorHandle> =>
  toHandle(
    gramShape,
    await runGramMatrix(
      getGpuDevice(),
      runUnary,
      await fromHandle(input),
      inputShape,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    ),
  );
export const mseScalarFromHandles = async (
  a: RuntimeTensorHandle,
  b: RuntimeTensorHandle,
): Promise<number> => {
  const device = getGpuDevice();
  if (device === null) throw new Error("WebGPU is not initialized.");
  await device.queue.onSubmittedWorkDone();
  const aValues = await fromHandle(a);
  await device.queue.onSubmittedWorkDone();
  const bValues = await fromHandle(b);
  await device.queue.onSubmittedWorkDone();
  if (aValues.length !== bValues.length) {
    throw new Error("mseScalarFromHandles expects equal-length tensors.");
  }
  let sum = 0;
  for (let i = 0; i < aValues.length; i += 1) {
    const d = aValues[i] - bValues[i];
    sum += d * d;
  }
  return sum / aValues.length;
};

export const styleLossBackwardHandle = async (
  input: RuntimeTensorHandle,
  inputShape: RuntimeTensorShape,
  targetValues: Float32Array,
): Promise<RuntimeTensorHandle> =>
  toHandle(
    inputShape,
    await runStyleLossBackward(
      getGpuDevice(),
      runUnary,
      await fromHandle(input),
      inputShape,
      targetValues,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    ),
  );

export const styleLossBackwardHandleFromTarget = async (
  input: RuntimeTensorHandle,
  inputShape: RuntimeTensorShape,
  target: RuntimeTensorHandle,
): Promise<RuntimeTensorHandle> =>
  toHandle(
    inputShape,
    await runStyleLossBackward(
      getGpuDevice(),
      runUnary,
      await fromHandle(input),
      inputShape,
      await fromHandle(target),
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    ),
  );
export const contentLossBackwardHandle = async (
  input: RuntimeTensorHandle,
  target: RuntimeTensorHandle,
  shape: RuntimeTensorShape,
): Promise<RuntimeTensorHandle> =>
  toHandle(
    shape,
    await runContentLossBackward(
      getGpuDevice(),
      runUnary,
      await fromHandle(input),
      await fromHandle(target),
      BUFFER_USAGE_STORAGE_COPY_DST,
    ),
  );
export const reluBackwardHandle = async (
  forwardInput: RuntimeTensorHandle,
  gradOutput: RuntimeTensorHandle,
  shape: RuntimeTensorShape,
): Promise<RuntimeTensorHandle> =>
  toHandle(
    shape,
    await runReluBackward(
      getGpuDevice(),
      runUnary,
      await fromHandle(forwardInput),
      await fromHandle(gradOutput),
    ),
  );
export const maxPoolBackwardHandle = async (
  forwardInput: RuntimeTensorHandle,
  inputShape: RuntimeTensorShape,
  gradOutput: RuntimeTensorHandle,
): Promise<RuntimeTensorHandle> =>
  toHandle(
    inputShape,
    await runMaxPool2dBackward(
      getGpuDevice(),
      runUnary,
      await fromHandle(forwardInput),
      inputShape,
      await fromHandle(gradOutput),
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    ),
  );
export const conv2dBackwardInputHandle = async (
  inputShape: RuntimeTensorShape,
  gradOutput: RuntimeTensorHandle,
  weightValues: Float32Array,
  weightShape: readonly [number, number, number, number],
): Promise<RuntimeTensorHandle> =>
  toHandle(
    inputShape,
    await runConv2dBackwardInput(
      getGpuDevice(),
      runUnary,
      inputShape,
      await fromHandle(gradOutput),
      weightValues,
      weightShape,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    ),
  );
export const normalizeBackwardHandle = async (
  gradOutput: RuntimeTensorHandle,
  inputShape: RuntimeTensorShape,
  std: readonly [number, number, number],
): Promise<RuntimeTensorHandle> =>
  toHandle(
    inputShape,
    await runNormalizeBackward(
      getGpuDevice(),
      runUnary,
      await fromHandle(gradOutput),
      inputShape,
      std,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    ),
  );
