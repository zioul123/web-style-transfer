/// <reference lib="webworker" />

import type { WorkerRequest } from "../../../../../types";
import { makeConv2dReluShader } from "../../../ops/convolution/conv2d.shader";
import { runReluForward, runReluBackward } from "../../../ops/relu/relu.run";
import {
  runNormalizeBackward,
  runNormalizeForward,
} from "../../../ops/normalization/normalize.run";
import {
  runMaxPool2dBackward,
  runMaxPool2dForward,
} from "../../../ops/pooling/maxpool.run";
import {
  runConv2dBackwardInput,
  runConv2dForward,
  runConv2dReluForward,
} from "../../../ops/convolution/conv2d.run";
import { runGramMatrix } from "../../../ops/gram/gram.run";
import { runMse } from "../../../ops/loss/mse.run";
import { runContentLossBackward } from "../../../ops/loss/contentLoss.run";
import { runStyleLossBackwardFromTargetGram } from "../../../ops/loss/styleLoss.run";
import {
  acquireReusableBuffer,
  releaseReusableBuffer,
} from "../../../runtime/bufferPool";
import {
  BUFFER_USAGE_MAP_READ_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_SRC,
  BUFFER_USAGE_UNIFORM_COPY_DST,
  MAP_MODE_READ,
} from "../../../runtime/gpuFlags";
import {
  runBinaryOp,
  runScalarBinaryOp,
} from "../../../runtime/shaderRunner";
import { getGpuDevice } from "../../../runtime/deviceState";
import { SUPER_FUSED_BLOCKS, POOL_LAYERS, RELU_LAYERS } from "../layerSchedules";
import { runUnary } from "../../../runtime/computeContext";
import { createInputOptimizer } from "../input-optimizer/createInputOptimizer";
import { createCpuVectorOps } from "../input-optimizer/cpuVectorOps";
import type { InputOptimizerConfig } from "../input-optimizer/types";

type StyleTransferPayload = Extract<WorkerRequest, { type: "run-style-transfer" }>;

type StyleTransferRunResult = {
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
};

export const runStyleTransferLegacy = async (
  payload: StyleTransferPayload,
): Promise<StyleTransferRunResult> => {
  const reluLayerSet = new Set(RELU_LAYERS);
  const poolLayerSet = new Set(POOL_LAYERS);
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
      for (const block of SUPER_FUSED_BLOCKS) {
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
  const optimizerConfig: InputOptimizerConfig = {
    optimizer: payload.optimizer ?? "sgd",
    count: inputValues.length,
    learningRate: payload.learningRate,
    adamBeta1: payload.adamBeta1 ?? 0.9,
    adamBeta2: payload.adamBeta2 ?? 0.999,
    adamEpsilon: payload.adamEpsilon ?? 1e-8,
    lbfgsMemory: payload.lbfgsMemory ?? 10,
    lbfgsEpsilon: payload.lbfgsEpsilon ?? 1e-8,
  };
  const inputOptimizer = createInputOptimizer(
    optimizerConfig,
    createCpuVectorOps(inputValues.length),
  );

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
    const clampStart = performance.now();
    const clamped = await inputOptimizer.step(inputValues, grad, step);
    clampMs += performance.now() - clampStart;
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
  if (device === null) {
    throw new Error("WebGPU device is not initialized.");
  }
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
