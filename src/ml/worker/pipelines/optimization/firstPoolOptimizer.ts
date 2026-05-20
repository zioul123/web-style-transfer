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
import { runGramMatrix } from "../../ops/gram/gram.run";
import { runMse } from "../../ops/loss/mse.run";
import { runContentLossBackward } from "../../ops/loss/contentLoss.run";
import {
  runStyleLossBackward,
} from "../../ops/loss/styleLoss.run";
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
import {
  runBinaryOp,
  runScalarBinaryOp,
  runUnaryShader,
} from "../../runtime/shaderRunner";
import { getGpuDevice } from "../../runtime/deviceState";

export const runUnary = (
  code: string,
  input: Float32Array,
  outCount: number,
  extraEntries: GPUBindGroupEntry[] = [],
): Promise<Float32Array> =>
  runUnaryShader(getGpuDevice(), code, input, outCount, extraEntries);

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

  let inputValues = new Float32Array(payload.initialInputValues);
  const losses: number[] = [];
  for (let step = 0; step < payload.steps; step += 1) {
    const norm = await runNormalizeForward(
      getGpuDevice(),
      runUnary,
      inputValues,
      payload.inputShape,
      payload.mean,
      payload.std,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    );
    const conv1 = await convForward(
      norm,
      payload.inputShape,
      conv1Weight.values,
      conv1Weight.shape,
      payload.conv1Bias,
    );
    const relu1 = await runReluForward(runUnary, conv1);
    const conv2 = await convForward(
      relu1,
      [1, 64, 16, 16],
      conv2Weight.values,
      conv2Weight.shape,
      payload.conv2Bias,
    );
    const relu2 = await runReluForward(runUnary, conv2);
    const pool = await runMaxPool2dForward(
      getGpuDevice(),
      runUnary,
      relu2,
      [1, 64, 16, 16],
      BUFFER_USAGE_UNIFORM_COPY_DST,
    );
    const conv3 = await convForward(
      pool,
      [1, 64, 8, 8],
      conv3Weight.values,
      conv3Weight.shape,
      payload.conv3Bias,
    );
    const relu3 = await runReluForward(runUnary, conv3);

    const styleLoss1 =
      (await runMse(
        getGpuDevice(),
        acquireReusableBuffer,
        releaseReusableBuffer,
        await runGramMatrix(
          getGpuDevice(),
          runUnary,
          relu1,
          [1, 64, 16, 16],
          BUFFER_USAGE_UNIFORM_COPY_DST,
        ),
        await runGramMatrix(
          getGpuDevice(),
          runUnary,
          styleRelu1,
          [1, 64, 16, 16],
          BUFFER_USAGE_UNIFORM_COPY_DST,
        ),
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_STORAGE_COPY_SRC,
        BUFFER_USAGE_MAP_READ_COPY_DST,
        MAP_MODE_READ,
      )) * payload.styleWeightConv1;
    const styleLoss3 =
      (await runMse(
        getGpuDevice(),
        acquireReusableBuffer,
        releaseReusableBuffer,
        await runGramMatrix(
          getGpuDevice(),
          runUnary,
          relu3,
          [1, 128, 8, 8],
          BUFFER_USAGE_UNIFORM_COPY_DST,
        ),
        await runGramMatrix(
          getGpuDevice(),
          runUnary,
          styleRelu3,
          [1, 128, 8, 8],
          BUFFER_USAGE_UNIFORM_COPY_DST,
        ),
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_STORAGE_COPY_SRC,
        BUFFER_USAGE_MAP_READ_COPY_DST,
        MAP_MODE_READ,
      )) * payload.styleWeightConv3;
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

    const gStyle1Relu = await runScalarBinaryOp(
      getGpuDevice(),
      "mul",
      await runStyleLossBackward(
        getGpuDevice(),
        runUnary,
        relu1,
        [1, 64, 16, 16],
        styleRelu1,
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_UNIFORM_COPY_DST,
      ),
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

    const next = new Float32Array(inputValues.length);
    for (let i = 0; i < inputValues.length; i += 1)
      next[i] = Math.max(
        0,
        Math.min(1, inputValues[i] - payload.learningRate * gInput[i]),
      );
    inputValues = next;
  }
  return { losses, finalValues: Array.from(inputValues) };
};

