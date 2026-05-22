/// <reference lib="webworker" />

import type { WorkerRequest } from "../../../../types";
import { createTensor } from "../../../index";
import { runReluForward, runReluBackward } from "../../ops/relu/relu.run";
import {
  runNormalizeBackward,
  runNormalizeForward,
} from "../../ops/normalization/normalize.run";
import {
  runMaxPool2dBackward,
  runMaxPool2dForward,
} from "../../ops/pooling/maxpool.run";
import {
  runConv2dBackwardInput,
  runConv2dForward,
} from "../../ops/convolution/conv2d.run";
import { runMse } from "../../ops/loss/mse.run";
import { runContentLossBackward } from "../../ops/loss/contentLoss.run";
import { runStyleLossBackward } from "../../ops/loss/styleLoss.run";
import {
  acquireReusableBuffer,
  releaseReusableBuffer,
} from "../../runtime/bufferPool";
import {
  BUFFER_USAGE_MAP_READ_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_SRC,
  BUFFER_USAGE_UNIFORM_COPY_DST,
  MAP_MODE_READ,
} from "../../runtime/gpuFlags";
import { runBinaryOp, runScalarBinaryOp } from "../../runtime/shaderRunner";
import { getGpuDevice } from "../../runtime/deviceState";
import {
  acquireTensorHandleFromCpu,
  chainTensorBinaryOp,
  chainClamp01,
  chainTensorScalarOp,
  readImageSnapshotBoundary,
  releaseTensorHandle,
  runUnary,
} from "../../runtime/computeContext";
import { conv2dForwardHandle, gramHandle, maxPoolForwardHandle, mseScalarFromHandles, normalizeForwardHandle, reluForwardHandle, styleLossBackwardHandle } from "../../runtime/handleOps/firstPoolHandleOps";

export const runFirstPoolOptimizer = async (
  payload: Extract<WorkerRequest, { type: "run-first-pool-optimizer" }>,
): Promise<{ losses: number[]; finalValues: number[] }> => {
  const convForward = async (
    inputValues: Float32Array,
    inputShape: readonly [number, number, number, number],
    weightValues: Float32Array,
    weightShape: readonly [number, number, number, number],
    bias: number[],
  ): Promise<Float32Array> =>
    await runConv2dForward(
      getGpuDevice(),
      runUnary,
      inputValues,
      inputShape,
      weightValues,
      weightShape,
      bias,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    );

  const conv1Weight = createTensor(
    payload.conv1Weight.shape,
    payload.conv1Weight.values,
  );
  const conv2Weight = createTensor(
    payload.conv2Weight.shape,
    payload.conv2Weight.values,
  );
  const conv3Weight = createTensor(
    payload.conv3Weight.shape,
    payload.conv3Weight.values,
  );

  const contentNorm = await runNormalizeForward(
    getGpuDevice(),
    runUnary,
    new Float32Array(payload.contentImageValues),
    payload.inputShape,
    payload.mean,
    payload.std,
    BUFFER_USAGE_STORAGE_COPY_DST,
    BUFFER_USAGE_UNIFORM_COPY_DST,
  );
  const styleNorm = await runNormalizeForward(
    getGpuDevice(),
    runUnary,
    new Float32Array(payload.styleImageValues),
    payload.inputShape,
    payload.mean,
    payload.std,
    BUFFER_USAGE_STORAGE_COPY_DST,
    BUFFER_USAGE_UNIFORM_COPY_DST,
  );

  const styleConv1 = await convForward(
    styleNorm,
    payload.inputShape,
    conv1Weight.values,
    conv1Weight.shape,
    payload.conv1Bias,
  );
  const styleRelu1 = await runReluForward(runUnary, styleConv1);
  const styleConv2 = await convForward(
    styleRelu1,
    [1, 64, 16, 16],
    conv2Weight.values,
    conv2Weight.shape,
    payload.conv2Bias,
  );
  const styleRelu2 = await runReluForward(runUnary, styleConv2);
  const stylePool = await runMaxPool2dForward(
    getGpuDevice(),
    runUnary,
    styleRelu2,
    [1, 64, 16, 16],
    BUFFER_USAGE_UNIFORM_COPY_DST,
  );
  const styleConv3 = await convForward(
    stylePool,
    [1, 64, 8, 8],
    conv3Weight.values,
    conv3Weight.shape,
    payload.conv3Bias,
  );
  const styleRelu3 = await runReluForward(runUnary, styleConv3);

  const contentConv1 = await convForward(
    contentNorm,
    payload.inputShape,
    conv1Weight.values,
    conv1Weight.shape,
    payload.conv1Bias,
  );
  const contentRelu1 = await runReluForward(runUnary, contentConv1);
  const contentConv2 = await convForward(
    contentRelu1,
    [1, 64, 16, 16],
    conv2Weight.values,
    conv2Weight.shape,
    payload.conv2Bias,
  );
  const contentRelu2 = await runReluForward(runUnary, contentConv2);

  let inputHandle = acquireTensorHandleFromCpu(
    payload.inputShape,
    new Float32Array(payload.initialInputValues),
  );
  const losses: number[] = [];
  for (let step = 0; step < payload.steps; step += 1) {
    const normHandle = await normalizeForwardHandle(
      inputHandle,
      payload.inputShape,
      payload.mean,
      payload.std,
    );
    const norm = await readImageSnapshotBoundary(normHandle);
    releaseTensorHandle(normHandle);
    const normCpuHandle = acquireTensorHandleFromCpu(payload.inputShape, norm);
    const conv1Handle = await conv2dForwardHandle(
      normCpuHandle,
      payload.inputShape,
      conv1Weight.values,
      conv1Weight.shape,
      payload.conv1Bias,
      [1, 64, 16, 16],
    );
    releaseTensorHandle(normCpuHandle);
    const conv1 = await readImageSnapshotBoundary(conv1Handle);
    releaseTensorHandle(conv1Handle);
    const conv1CpuHandle = acquireTensorHandleFromCpu([1, 64, 16, 16], conv1);
    const relu1Handle = await reluForwardHandle(conv1CpuHandle, [1, 64, 16, 16]);
    releaseTensorHandle(conv1CpuHandle);
    const relu1 = await readImageSnapshotBoundary(relu1Handle);
    releaseTensorHandle(relu1Handle);
    const relu1CpuHandle = acquireTensorHandleFromCpu([1, 64, 16, 16], relu1);
    const conv2Handle = await conv2dForwardHandle(
      relu1CpuHandle,
      [1, 64, 16, 16],
      conv2Weight.values,
      conv2Weight.shape,
      payload.conv2Bias,
      [1, 64, 16, 16],
    );
    releaseTensorHandle(relu1CpuHandle);
    const conv2 = await readImageSnapshotBoundary(conv2Handle);
    releaseTensorHandle(conv2Handle);
    const conv2CpuHandle = acquireTensorHandleFromCpu([1, 64, 16, 16], conv2);
    const relu2Handle = await reluForwardHandle(conv2CpuHandle, [1, 64, 16, 16]);
    releaseTensorHandle(conv2CpuHandle);
    const relu2 = await readImageSnapshotBoundary(relu2Handle);
    releaseTensorHandle(relu2Handle);
    const relu2CpuHandle = acquireTensorHandleFromCpu([1, 64, 16, 16], relu2);
    const poolHandle = await maxPoolForwardHandle(
      relu2CpuHandle,
      [1, 64, 16, 16],
      [1, 64, 8, 8],
    );
    releaseTensorHandle(relu2CpuHandle);
    const pool = await readImageSnapshotBoundary(poolHandle);
    releaseTensorHandle(poolHandle);
    const poolCpuHandle = acquireTensorHandleFromCpu([1, 64, 8, 8], pool);
    const conv3Handle = await conv2dForwardHandle(
      poolCpuHandle,
      [1, 64, 8, 8],
      conv3Weight.values,
      conv3Weight.shape,
      payload.conv3Bias,
      [1, 128, 8, 8],
    );
    releaseTensorHandle(poolCpuHandle);
    const conv3 = await readImageSnapshotBoundary(conv3Handle);
    releaseTensorHandle(conv3Handle);
    const conv3CpuHandle = acquireTensorHandleFromCpu([1, 128, 8, 8], conv3);
    const relu3Handle = await reluForwardHandle(conv3CpuHandle, [1, 128, 8, 8]);
    releaseTensorHandle(conv3CpuHandle);
    const relu3 = await readImageSnapshotBoundary(relu3Handle);
    releaseTensorHandle(relu3Handle);

    const relu1GramInputHandle = acquireTensorHandleFromCpu([1, 64, 16, 16], relu1);
    const relu1GramHandle = await gramHandle(
      relu1GramInputHandle,
      [1, 64, 16, 16],
      [1, 1, 64, 64],
    );
    releaseTensorHandle(relu1GramInputHandle);
    const relu1Gram = await readImageSnapshotBoundary(relu1GramHandle);
    releaseTensorHandle(relu1GramHandle);

    const styleRelu1GramInputHandle = acquireTensorHandleFromCpu([1, 64, 16, 16], styleRelu1);
    const styleRelu1GramHandle = await gramHandle(
      styleRelu1GramInputHandle,
      [1, 64, 16, 16],
      [1, 1, 64, 64],
    );
    releaseTensorHandle(styleRelu1GramInputHandle);
    const styleRelu1Gram = await readImageSnapshotBoundary(styleRelu1GramHandle);
    releaseTensorHandle(styleRelu1GramHandle);
    const relu1GramForLossHandle = acquireTensorHandleFromCpu([1, 1, 64, 64], relu1Gram);
    const styleRelu1GramForLossHandle = acquireTensorHandleFromCpu([1, 1, 64, 64], styleRelu1Gram);
    const styleLoss1 =
      (await mseScalarFromHandles(relu1GramForLossHandle, styleRelu1GramForLossHandle)) *
      payload.styleWeightConv1;
    releaseTensorHandle(relu1GramForLossHandle);
    releaseTensorHandle(styleRelu1GramForLossHandle);
    const relu3GramInputHandle = acquireTensorHandleFromCpu([1, 128, 8, 8], relu3);
    const relu3GramHandle = await gramHandle(
      relu3GramInputHandle,
      [1, 128, 8, 8],
      [1, 1, 128, 128],
    );
    releaseTensorHandle(relu3GramInputHandle);
    const styleRelu3GramInputHandle = acquireTensorHandleFromCpu([1, 128, 8, 8], styleRelu3);
    const styleRelu3GramHandle = await gramHandle(
      styleRelu3GramInputHandle,
      [1, 128, 8, 8],
      [1, 1, 128, 128],
    );
    releaseTensorHandle(styleRelu3GramInputHandle);
    const styleLoss3 =
      (await mseScalarFromHandles(relu3GramHandle, styleRelu3GramHandle)) *
      payload.styleWeightConv3;
    releaseTensorHandle(relu3GramHandle);
    releaseTensorHandle(styleRelu3GramHandle);
    const contentLoss =
      (await runMse(
        getGpuDevice(),
        acquireReusableBuffer,
        releaseReusableBuffer,
        relu2,
        contentRelu2,
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_STORAGE_COPY_SRC,
        BUFFER_USAGE_MAP_READ_COPY_DST,
        MAP_MODE_READ,
      )) * payload.contentWeight;
    losses.push(styleLoss1 + styleLoss3 + contentLoss);

    const relu1ForStyleGradHandle = acquireTensorHandleFromCpu([1, 64, 16, 16], relu1);
    const gStyle1ReluRawHandle = await styleLossBackwardHandle(
      relu1ForStyleGradHandle,
      [1, 64, 16, 16],
      styleRelu1,
    );
    releaseTensorHandle(relu1ForStyleGradHandle);
    const gStyle1ReluRaw = await readImageSnapshotBoundary(gStyle1ReluRawHandle);
    releaseTensorHandle(gStyle1ReluRawHandle);
    const gStyle1Relu = await runScalarBinaryOp(
      getGpuDevice(),
      "mul",
      gStyle1ReluRaw,
      payload.styleWeightConv1,
      false,
    );
    const gStyle3Relu = await runScalarBinaryOp(
      getGpuDevice(),
      "mul",
      await runStyleLossBackward(
        getGpuDevice(),
        runUnary,
        relu3,
        [1, 128, 8, 8],
        styleRelu3,
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_UNIFORM_COPY_DST,
      ),
      payload.styleWeightConv3,
      false,
    );
    const gContentRelu = await runScalarBinaryOp(
      getGpuDevice(),
      "mul",
      await runContentLossBackward(
        getGpuDevice(),
        runUnary,
        relu2,
        contentRelu2,
        BUFFER_USAGE_STORAGE_COPY_DST,
      ),
      payload.contentWeight,
      false,
    );
    const gStyle3 = await runReluBackward(
      getGpuDevice(),
      runUnary,
      conv3,
      gStyle3Relu,
    );
    const gContent = await runReluBackward(
      getGpuDevice(),
      runUnary,
      conv2,
      gContentRelu,
    );
    const gStyle1 = await runReluBackward(
      getGpuDevice(),
      runUnary,
      conv1,
      gStyle1Relu,
    );

    const gPool = await runConv2dBackwardInput(
      getGpuDevice(),
      runUnary,
      [1, 64, 8, 8],
      gStyle3,
      conv3Weight.values,
      conv3Weight.shape,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    );
    const gRelu2 = await runMaxPool2dBackward(
      getGpuDevice(),
      runUnary,
      relu2,
      [1, 64, 16, 16],
      gPool,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    );
    const gConv2FromPool = await runReluBackward(
      getGpuDevice(),
      runUnary,
      conv2,
      gRelu2,
    );
    const gConv2Total = await runBinaryOp(
      getGpuDevice(),
      "add",
      gConv2FromPool,
      gContent,
      "tensorTensor",
    );
    const gRelu1 = await runConv2dBackwardInput(
      getGpuDevice(),
      runUnary,
      [1, 64, 16, 16],
      gConv2Total,
      conv2Weight.values,
      conv2Weight.shape,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    );
    const gConv1FromConv2 = await runReluBackward(
      getGpuDevice(),
      runUnary,
      conv1,
      gRelu1,
    );
    const gConv1Total = await runBinaryOp(
      getGpuDevice(),
      "add",
      gConv1FromConv2,
      gStyle1,
      "tensorTensor",
    );
    const gNorm = await runConv2dBackwardInput(
      getGpuDevice(),
      runUnary,
      payload.inputShape,
      gConv1Total,
      conv1Weight.values,
      conv1Weight.shape,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    );
    const gInput = await runNormalizeBackward(
      getGpuDevice(),
      runUnary,
      gNorm,
      payload.inputShape,
      payload.std,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    );

    const gradHandle = acquireTensorHandleFromCpu(payload.inputShape, gInput);
    const lrScaledGradHandle = await chainTensorScalarOp(
      gradHandle,
      "mul",
      payload.learningRate,
    );
    releaseTensorHandle(gradHandle);
    const nextHandle = await chainTensorBinaryOp(
      inputHandle,
      lrScaledGradHandle,
      "sub",
    );
    releaseTensorHandle(inputHandle);
    releaseTensorHandle(lrScaledGradHandle);
    const clampedHandle = await chainClamp01(nextHandle);
    releaseTensorHandle(nextHandle);
    inputHandle = clampedHandle;
  }
  const finalValues = await readImageSnapshotBoundary(inputHandle);
  releaseTensorHandle(inputHandle);
  return { losses, finalValues: Array.from(finalValues) };
};
