/// <reference lib="webworker" />

import type { WorkerRequest } from "../../types";
import { createTensor } from "../index";
import { makeConv2dReluShader } from "./ops/convolution/conv2d.shader";
import { runReluForward, runReluBackward } from "./ops/relu/relu.run";
import {
  runNormalizeBackward,
  runNormalizeForward,
} from "./ops/normalization/normalize.run";
import {
  runMaxPool2dBackward,
  runMaxPool2dForward,
} from "./ops/pooling/maxpool.run";
import {
  runConv2dBackwardInput,
  runConv2dForward,
  runConv2dReluForward,
} from "./ops/convolution/conv2d.run";
import { runGramMatrix, runGramBackward } from "./ops/gram/gram.run";
import { runMse } from "./ops/loss/mse.run";
import { runContentLossBackward } from "./ops/loss/contentLoss.run";
import {
  runStyleLossBackward,
  runStyleLossBackwardFromTargetGram,
} from "./ops/loss/styleLoss.run";
import {
  acquireReusableBuffer,
  releaseReusableBuffer,
} from "./runtime/bufferPool";
import {
  BUFFER_USAGE_MAP_READ_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_SRC,
  BUFFER_USAGE_UNIFORM_COPY_DST,
  MAP_MODE_READ,
} from "./runtime/gpuFlags";
import {
  getTensorFromOperand,
  getValuesFromOperand,
  isBinaryTensorOpPayload,
} from "./runtime/operands";
import {
  runBinaryOp,
  runClamp,
  runScalarBinaryOp,
  runUnaryShader,
} from "./runtime/shaderRunner";
import { getGpuDevice } from "./runtime/deviceState";
import { postResponse } from "./main-thread-protocol/responses";

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

export const runStyleTransfer = async (
  payload: Extract<WorkerRequest, { type: "run-style-transfer" }>,
): Promise<{
  losses: number[];
  finalValues: number[];
  stats: {
    elapsedMs: number;
    avgStepMs: number;
    forwardMs: number;
    backwardMs: number;
    lossMs: number;
    updateMs: number;
    clampMs: number;
    steps: number;
  };
}> => {
  const reluLayers: readonly number[] = [
    1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29,
  ];
  const poolLayers: readonly number[] = [4, 9, 18, 27];
  const reluLayerSet = new Set(reluLayers);
  const poolLayerSet = new Set(poolLayers);
  const convLayerCache: Record<
    number,
    | {
        shape: [number, number, number, number];
        values: Float32Array;
        bias: Float32Array;
      }
    | undefined
  > = {};
  const fusedOps = payload.fusedOps ?? false;
  const superFusedOps = payload.superFusedOps ?? false;
  const superFusedBlocks: readonly (readonly number[])[] = [
    [0, 1, 2, 3],
    [5, 6, 7, 8],
    [10, 11, 12, 13, 14, 15, 16, 17],
    [19, 20, 21, 22, 23, 24, 25, 26],
    [28, 29],
  ];
  for (let layerIndex = 0; layerIndex <= 29; layerIndex += 1) {
    const weightShapeRaw = payload.weights[`conv${layerIndex}.weightShape`];
    const weightValues = payload.weights[`conv${layerIndex}.weightValues`];
    const biasValues = payload.weights[`conv${layerIndex}.biasValues`];
    if (
      Array.isArray(weightShapeRaw) &&
      Array.isArray(weightValues) &&
      Array.isArray(biasValues)
    ) {
      convLayerCache[layerIndex] = {
        shape: weightShapeRaw as [number, number, number, number],
        values: new Float32Array(weightValues),
        bias: new Float32Array(biasValues),
      };
    }
  }

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
  const runForward = async (
    start: Float32Array,
  ): Promise<{
    reluOut: Record<number, Float32Array>;
    reluShape: Record<number, readonly [number, number, number, number]>;
    convInShape: Record<number, readonly [number, number, number, number]>;
    reluPre: Record<number, Float32Array>;
    layerInput: Record<number, Float32Array>;
    layerInputShape: Record<number, readonly [number, number, number, number]>;
    final: Float32Array;
  }> => {
    let shape: readonly [number, number, number, number] = payload.inputShape;
    let values = start;
    const reluOut: Record<number, Float32Array> = {};
    const reluShape: Record<number, readonly [number, number, number, number]> =
      {};
    const convInShape: Record<
      number,
      readonly [number, number, number, number]
    > = {};
    const reluPre: Record<number, Float32Array> = {};
    const layerInput: Record<number, Float32Array> = {};
    const layerInputShape: Record<
      number,
      readonly [number, number, number, number]
    > = {};
    if (superFusedOps) {
      for (const block of superFusedBlocks) {
        let blockShape: readonly [number, number, number, number] = shape;
        for (const layerIndex of block) {
          layerInput[layerIndex] = values;
          layerInputShape[layerIndex] = shape;
          const convLayer = convLayerCache[layerIndex];
          if (convLayer !== undefined) {
            convInShape[layerIndex] = blockShape;
            blockShape = [1, convLayer.shape[0], blockShape[2], blockShape[3]];
          }
        }
        const blockLayers: {
          layerIndex: number;
          reluLayerIndex: number | null;
          shape: readonly [number, number, number, number];
          values: Float32Array;
          bias: Float32Array;
        }[] = [];
        for (const layerIndex of block) {
          const convLayer = convLayerCache[layerIndex];
          if (convLayer !== undefined)
            blockLayers.push({
              layerIndex,
              reluLayerIndex: reluLayerSet.has(layerIndex + 1)
                ? layerIndex + 1
                : null,
              shape: convLayer.shape,
              values: convLayer.values,
              bias: convLayer.bias,
            });
        }
        const tapLayers = block.filter((layerIndex) =>
          reluLayerSet.has(layerIndex),
        );
        if (fusedOps && blockLayers.length > 0) {
          const blockResult = await runSuperFusedConvReluBlock(
            values,
            shape,
            blockLayers,
            tapLayers,
          );
          values = blockResult.finalValues;
          shape = blockResult.finalShape;
          for (const layerIndex of tapLayers) {
            const tap = blockResult.reluOutByLayer[layerIndex];
            const convLayer = convLayerCache[layerIndex - 1];
            if (tap !== undefined) {
              reluOut[layerIndex] = tap;
              reluShape[layerIndex] =
                convLayer === undefined
                  ? shape
                  : [
                      1,
                      convLayer.shape[0],
                      layerInputShape[layerIndex][2],
                      layerInputShape[layerIndex][3],
                    ];
            }
          }
        } else {
          for (const layerIndex of block) {
            const convLayer = convLayerCache[layerIndex];
            const hasRelu = reluLayerSet.has(layerIndex);
            if (convLayer !== undefined) {
              values = await runConv2dForward(
                getGpuDevice(),
                runUnary,
                values,
                shape,
                convLayer.values,
                convLayer.shape,
                convLayer.bias,
                BUFFER_USAGE_STORAGE_COPY_DST,
                BUFFER_USAGE_UNIFORM_COPY_DST,
              );
              shape = [1, convLayer.shape[0], shape[2], shape[3]];
            }
            if (hasRelu) {
              reluPre[layerIndex] = values;
              values = await runReluForward(runUnary, values);
              reluOut[layerIndex] = values;
              reluShape[layerIndex] = shape;
            }
          }
        }
        const poolLayerIndex = block[block.length - 1] + 1;
        if (poolLayerSet.has(poolLayerIndex)) {
          layerInput[poolLayerIndex] = values;
          layerInputShape[poolLayerIndex] = shape;
          values = await runMaxPool2dForward(
            getGpuDevice(),
            runUnary,
            values,
            shape,
            BUFFER_USAGE_UNIFORM_COPY_DST,
          );
          shape = [
            1,
            shape[1],
            Math.floor(shape[2] / 2),
            Math.floor(shape[3] / 2),
          ];
        }
      }
    } else {
      for (let layerIndex = 0; layerIndex <= 29; layerIndex += 1) {
        layerInput[layerIndex] = values;
        layerInputShape[layerIndex] = shape;
        const convLayer = convLayerCache[layerIndex];
        const hasRelu = reluLayerSet.has(layerIndex);
        if (convLayer !== undefined) {
          convInShape[layerIndex] = shape;
          if (fusedOps && hasRelu) {
            values = await runConv2dReluForward(
              getGpuDevice(),
              runUnary,
              values,
              shape,
              convLayer.values,
              convLayer.shape,
              convLayer.bias,
              BUFFER_USAGE_STORAGE_COPY_DST,
              BUFFER_USAGE_UNIFORM_COPY_DST,
            );
            reluOut[layerIndex] = values;
            reluShape[layerIndex] = [1, convLayer.shape[0], shape[2], shape[3]];
          } else {
            values = await runConv2dForward(
              getGpuDevice(),
              runUnary,
              values,
              shape,
              convLayer.values,
              convLayer.shape,
              convLayer.bias,
              BUFFER_USAGE_STORAGE_COPY_DST,
              BUFFER_USAGE_UNIFORM_COPY_DST,
            );
          }
          shape = [1, convLayer.shape[0], shape[2], shape[3]];
        }
        if (hasRelu && !(fusedOps && convLayer !== undefined)) {
          reluPre[layerIndex] = values;
          values = await runReluForward(runUnary, values);
          reluOut[layerIndex] = values;
          reluShape[layerIndex] = shape;
        }
        if (poolLayerSet.has(layerIndex)) {
          values = await runMaxPool2dForward(
            getGpuDevice(),
            runUnary,
            values,
            shape,
            BUFFER_USAGE_UNIFORM_COPY_DST,
          );
          shape = [
            1,
            shape[1],
            Math.floor(shape[2] / 2),
            Math.floor(shape[3] / 2),
          ];
        }
      }
    }
    return {
      reluOut,
      reluShape,
      convInShape,
      reluPre,
      layerInput,
      layerInputShape,
      final: values,
    };
  };

  const contentTargets = await runForward(contentNorm);
  const styleTargets = await runForward(styleNorm);
  const styleGramTargets: Record<number, Float32Array> = {};
  for (const reluLayerIndex of payload.styleLayerIndices) {
    const styleRelu = styleTargets.reluOut[reluLayerIndex];
    const styleShape = styleTargets.reluShape[reluLayerIndex];
    if (styleRelu === undefined || styleShape === undefined)
      throw new Error(
        `Missing precomputed style activation for ReLU layer index ${reluLayerIndex}.`,
      );
    styleGramTargets[reluLayerIndex] = await runGramMatrix(
      getGpuDevice(),
      runUnary,
      styleRelu,
      styleShape,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    );
  }
  let inputValues: Float32Array<ArrayBufferLike> = new Float32Array(
    payload.inputImageValues,
  );
  const losses: number[] = [];
  const optimizer = payload.optimizer ?? "sgd";
  const adamBeta1 = payload.adamBeta1 ?? 0.9;
  const adamBeta2 = payload.adamBeta2 ?? 0.999;
  const adamEps = payload.adamEpsilon ?? 1e-8;
  const lbfgsMemory = payload.lbfgsMemory ?? 10;
  const lbfgsEps = payload.lbfgsEpsilon ?? 1e-8;
  const adamM = new Float32Array(inputValues.length);
  const adamV = new Float32Array(inputValues.length);
  const lbfgsHistory: { s: Float32Array; y: Float32Array; rho: number }[] = [];
  let previousInput: Float32Array | null = null;
  let previousGrad: Float32Array | null = null;

  let forwardMs = 0;
  let backwardMs = 0;
  let lossMs = 0;
  let updateMs = 0;
  let clampMs = 0;
  const totalStart = performance.now();

  const updateInput = async (
    grad: Float32Array,
    step: number,
  ): Promise<Float32Array<ArrayBufferLike>> => {
    const nextValues = new Float32Array(inputValues.length);
    if (optimizer === "sgd") {
      for (let i = 0; i < inputValues.length; i += 1)
        nextValues[i] = inputValues[i] - payload.learningRate * grad[i];
    } else if (optimizer === "adam") {
      const t = step + 1;
      const b1Correction = 1 - Math.pow(adamBeta1, t);
      const b2Correction = 1 - Math.pow(adamBeta2, t);
      for (let i = 0; i < inputValues.length; i += 1) {
        const g = grad[i];
        adamM[i] = adamBeta1 * adamM[i] + (1 - adamBeta1) * g;
        adamV[i] = adamBeta2 * adamV[i] + (1 - adamBeta2) * g * g;
        const mHat = adamM[i] / b1Correction;
        const vHat = adamV[i] / b2Correction;
        nextValues[i] =
          inputValues[i] -
          payload.learningRate * (mHat / (Math.sqrt(vHat) + adamEps));
      }
    } else {
      if (previousInput !== null && previousGrad !== null) {
        const s = new Float32Array(inputValues.length);
        const y = new Float32Array(inputValues.length);
        let sy = 0;
        for (let i = 0; i < inputValues.length; i += 1) {
          s[i] = inputValues[i] - previousInput[i];
          y[i] = grad[i] - previousGrad[i];
          sy += s[i] * y[i];
        }
        if (sy > lbfgsEps) {
          lbfgsHistory.push({ s, y, rho: 1 / sy });
          if (lbfgsHistory.length > lbfgsMemory) lbfgsHistory.shift();
        }
      }
      const q = new Float32Array(grad);
      const alphas = new Float32Array(lbfgsHistory.length);
      for (let i = lbfgsHistory.length - 1; i >= 0; i -= 1) {
        const { s, y, rho } = lbfgsHistory[i];
        let sq = 0;
        for (let j = 0; j < q.length; j += 1) sq += s[j] * q[j];
        const alpha = rho * sq;
        alphas[i] = alpha;
        for (let j = 0; j < q.length; j += 1) q[j] -= alpha * y[j];
      }
      let scale = 1;
      if (lbfgsHistory.length > 0) {
        const last = lbfgsHistory[lbfgsHistory.length - 1];
        let ys = 0;
        let yy = 0;
        for (let i = 0; i < q.length; i += 1) {
          ys += last.y[i] * last.s[i];
          yy += last.y[i] * last.y[i];
        }
        if (yy > lbfgsEps) scale = ys / yy;
      }
      for (let i = 0; i < q.length; i += 1) q[i] *= scale;
      for (let i = 0; i < lbfgsHistory.length; i += 1) {
        const { s, y, rho } = lbfgsHistory[i];
        let yq = 0;
        for (let j = 0; j < q.length; j += 1) yq += y[j] * q[j];
        const beta = rho * yq;
        for (let j = 0; j < q.length; j += 1) q[j] += s[j] * (alphas[i] - beta);
      }
      for (let i = 0; i < inputValues.length; i += 1)
        nextValues[i] = inputValues[i] - payload.learningRate * q[i];
    }
    const clampStart = performance.now();
    const clamped = new Float32Array(
      await runClamp(getGpuDevice(), nextValues, 0, 1),
    );
    clampMs += performance.now() - clampStart;
    previousInput = new Float32Array(inputValues);
    previousGrad = new Float32Array(grad);
    return clamped;
  };

  for (let step = 0; step < payload.steps; step += 1) {
    const forwardStart = performance.now();
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
    const run = await runForward(norm);
    forwardMs += performance.now() - forwardStart;
    const lossStart = performance.now();
    let totalStyle = 0;
    for (const reluLayerIndex of payload.styleLayerIndices) {
      const inputRelu = run.reluOut[reluLayerIndex];
      if (inputRelu === undefined)
        throw new Error(
          `Missing ReLU activation for style layer index ${reluLayerIndex}.`,
        );
      const reluShape = run.reluShape[reluLayerIndex];
      if (reluShape === undefined)
        throw new Error(
          `Unsupported ReLU style layer index ${reluLayerIndex}.`,
        );
      const targetGram = styleGramTargets[reluLayerIndex];
      if (targetGram === undefined)
        throw new Error(
          `Missing precomputed style gram target for ReLU layer index ${reluLayerIndex}.`,
        );
      totalStyle += await runMse(
        getGpuDevice(),
        acquireReusableBuffer,
        releaseReusableBuffer,
        await runGramMatrix(
          getGpuDevice(),
          runUnary,
          inputRelu,
          reluShape,
          BUFFER_USAGE_UNIFORM_COPY_DST,
        ),
        targetGram,
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_STORAGE_COPY_SRC,
        BUFFER_USAGE_MAP_READ_COPY_DST,
        MAP_MODE_READ,
      );
    }
    const contentRelu = run.reluOut[payload.contentLayerIndex];
    const contentTargetRelu = contentTargets.reluOut[payload.contentLayerIndex];
    if (contentRelu === undefined || contentTargetRelu === undefined)
      throw new Error(
        `Missing ReLU activation for content layer index ${payload.contentLayerIndex}.`,
      );
    const contentLoss = await runMse(
      getGpuDevice(),
      acquireReusableBuffer,
      releaseReusableBuffer,
      contentRelu,
      contentTargetRelu,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_STORAGE_COPY_SRC,
      BUFFER_USAGE_MAP_READ_COPY_DST,
      MAP_MODE_READ,
    );
    losses.push(
      payload.styleWeight * totalStyle + payload.contentWeight * contentLoss,
    );
    lossMs += performance.now() - lossStart;
    const gradByReluLayer: Record<number, Float32Array> = {};
    for (const reluLayerIndex of payload.styleLayerIndices) {
      const inputRelu = run.reluOut[reluLayerIndex];
      if (inputRelu === undefined)
        throw new Error(
          `Missing ReLU activation for style gradient layer index ${reluLayerIndex}.`,
        );
      const reluShape = run.reluShape[reluLayerIndex];
      if (reluShape === undefined)
        throw new Error(
          `Unsupported ReLU style gradient layer index ${reluLayerIndex}.`,
        );
      const targetGram = styleGramTargets[reluLayerIndex];
      if (targetGram === undefined)
        throw new Error(
          `Missing precomputed style gram gradient target for ReLU layer index ${reluLayerIndex}.`,
        );
      const styleGrad = await runStyleLossBackwardFromTargetGram(
        getGpuDevice(),
        runUnary,
        inputRelu,
        reluShape,
        targetGram,
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_UNIFORM_COPY_DST,
      );
      const weightedStyleGrad =
        payload.styleWeight === 1
          ? styleGrad
          : await runScalarBinaryOp(
              getGpuDevice(),
              "mul",
              styleGrad,
              payload.styleWeight,
              false,
            );
      gradByReluLayer[reluLayerIndex] =
        gradByReluLayer[reluLayerIndex] === undefined
          ? weightedStyleGrad
          : await runBinaryOp(
              getGpuDevice(),
              "add",
              gradByReluLayer[reluLayerIndex],
              weightedStyleGrad,
              "tensorTensor",
            );
    }
    const contentGrad = await runContentLossBackward(
      getGpuDevice(),
      runUnary,
      contentRelu,
      contentTargetRelu,
      BUFFER_USAGE_STORAGE_COPY_DST,
    );
    const weightedContentGrad =
      payload.contentWeight === 1
        ? contentGrad
        : await runScalarBinaryOp(
            getGpuDevice(),
            "mul",
            contentGrad,
            payload.contentWeight,
            false,
          );
    gradByReluLayer[payload.contentLayerIndex] =
      gradByReluLayer[payload.contentLayerIndex] === undefined
        ? weightedContentGrad
        : await runBinaryOp(
            getGpuDevice(),
            "add",
            gradByReluLayer[payload.contentLayerIndex],
            weightedContentGrad,
            "tensorTensor",
          );

    const backwardStart = performance.now();
    let grad: Float32Array | null = null;
    for (let layerIndex = 29; layerIndex >= 0; layerIndex -= 1) {
      if (poolLayerSet.has(layerIndex)) {
        if (grad !== null)
          grad = await runMaxPool2dBackward(
            getGpuDevice(),
            runUnary,
            run.layerInput[layerIndex],
            run.layerInputShape[layerIndex],
            grad,
            BUFFER_USAGE_STORAGE_COPY_DST,
            BUFFER_USAGE_UNIFORM_COPY_DST,
          );
      }
      if (reluLayerSet.has(layerIndex)) {
        const tapGrad = gradByReluLayer[layerIndex];
        if (tapGrad !== undefined)
          grad =
            grad === null
              ? tapGrad
              : await runBinaryOp(
                  getGpuDevice(),
                  "add",
                  grad,
                  tapGrad,
                  "tensorTensor",
                );
        if (grad !== null)
          grad = fusedOps
            ? await runReluBackward(
                getGpuDevice(),
                runUnary,
                run.reluOut[layerIndex],
                grad,
              )
            : await runReluBackward(
                getGpuDevice(),
                runUnary,
                run.reluPre[layerIndex],
                grad,
              );
      }
      const convLayer = convLayerCache[layerIndex];
      if (convLayer !== undefined && grad !== null) {
        grad = await runConv2dBackwardInput(
          getGpuDevice(),
          runUnary,
          run.convInShape[layerIndex],
          grad,
          convLayer.values,
          convLayer.shape,
          BUFFER_USAGE_STORAGE_COPY_DST,
          BUFFER_USAGE_UNIFORM_COPY_DST,
        );
      }
    }
    if (grad === null)
      throw new Error("No gradient path reached the input image.");
    const gInput = await runNormalizeBackward(
      getGpuDevice(),
      runUnary,
      grad,
      payload.inputShape,
      payload.std,
      BUFFER_USAGE_STORAGE_COPY_DST,
      BUFFER_USAGE_UNIFORM_COPY_DST,
    );
    backwardMs += performance.now() - backwardStart;

    const updateStart = performance.now();
    inputValues = await updateInput(gInput, step);
    updateMs += performance.now() - updateStart;
  }
  const elapsedMs = performance.now() - totalStart;
  return {
    losses,
    finalValues: Array.from(inputValues),
    stats: {
      elapsedMs,
      avgStepMs: payload.steps === 0 ? 0 : elapsedMs / payload.steps,
      forwardMs,
      backwardMs,
      lossMs,
      updateMs,
      clampMs,
      steps: payload.steps,
    },
  };
};


const readGpuBuffer = async (
  device: GPUDevice,
  sourceBuffer: GPUBuffer,
  floatCount: number,
): Promise<Float32Array> => {
  const readBuffer: GPUBuffer = device.createBuffer({
    size: floatCount * 4,
    usage: BUFFER_USAGE_MAP_READ_COPY_DST,
  });
  const encoder: GPUCommandEncoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(sourceBuffer, 0, readBuffer, 0, floatCount * 4);
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(MAP_MODE_READ);
  const values = new Float32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  return values;
};

const runSuperFusedConvReluBlock = async (
  input: Float32Array,
  inputShape: readonly [number, number, number, number],
  layers: readonly {
    layerIndex: number;
    reluLayerIndex: number | null;
    shape: readonly [number, number, number, number];
    values: Float32Array;
    bias: Float32Array;
  }[],
  tapLayers: readonly number[],
): Promise<{
  finalValues: Float32Array;
  reluOutByLayer: Record<number, Float32Array>;
  finalShape: readonly [number, number, number, number];
}> => {
  if (getGpuDevice() === null) throw new Error("WebGPU is not initialized.");
  const device = getGpuDevice();
  const reluOutByLayer: Record<number, Float32Array> = {};
  let shape: readonly [number, number, number, number] = inputShape;
  let currentCount = input.length;
  let currentBuffer: GPUBuffer = device.createBuffer({
    size: input.byteLength,
    usage: BUFFER_USAGE_STORAGE_COPY_DST,
  });
  device.queue.writeBuffer(currentBuffer, 0, input);

  for (const layer of layers) {
    const outChannels = layer.shape[0];
    const outCount = outChannels * shape[2] * shape[3];
    const outBuffer: GPUBuffer = device.createBuffer({
      size: outCount * 4,
      usage: BUFFER_USAGE_STORAGE_COPY_SRC,
    });
    const weightBuffer: GPUBuffer = device.createBuffer({
      size: layer.values.byteLength,
      usage: BUFFER_USAGE_STORAGE_COPY_DST,
    });
    const biasBuffer: GPUBuffer = device.createBuffer({
      size: layer.bias.byteLength,
      usage: BUFFER_USAGE_STORAGE_COPY_DST,
    });
    const uniformBuffer: GPUBuffer = device.createBuffer({
      size: 4 * 4,
      usage: BUFFER_USAGE_UNIFORM_COPY_DST,
    });
    device.queue.writeBuffer(weightBuffer, 0, layer.values);
    device.queue.writeBuffer(biasBuffer, 0, layer.bias);
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Uint32Array([shape[1], outChannels, shape[2], shape[3]]),
    );
    const pipeline: GPUComputePipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          code: makeConv2dReluShader(outCount),
        }),
        entryPoint: "main",
      },
    });
    const bindGroup: GPUBindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: currentBuffer } },
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: biasBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer } },
        { binding: 4, resource: { buffer: outBuffer } },
      ],
    });
    const encoder: GPUCommandEncoder = device.createCommandEncoder();
    const pass: GPUComputePassEncoder = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(outCount / 64));
    pass.end();
    device.queue.submit([encoder.finish()]);
    if (
      layer.reluLayerIndex !== null &&
      tapLayers.includes(layer.reluLayerIndex)
    )
      reluOutByLayer[layer.reluLayerIndex] = await readGpuBuffer(
        device,
        outBuffer,
        outCount,
      );
    currentBuffer = outBuffer;
    currentCount = outCount;
    shape = [1, outChannels, shape[2], shape[3]];
  }
  const finalValues = await readGpuBuffer(device, currentBuffer, currentCount);
  return { finalValues, reluOutByLayer, finalShape: shape };
};
