/// <reference lib="webworker" />

import type { WorkerRequest } from "../../../../types";
import { createTensor } from "../../../index";
import {
  acquireTensorHandleFromCpu,
  chainClamp01,
  chainTensorBinaryOp,
  chainTensorScalarOp,
  readImageSnapshotBoundary,
  releaseTensorHandle,
} from "../../runtime/computeContext";
import {
  addHandles,
  readMseScalar,
  runContentLossBackwardHandle,
  runConv2dBackwardInputHandle,
  runConv2dForwardHandle,
  runMaxPool2dBackwardHandle,
  runMaxPool2dForwardHandle,
  runNormalizeBackwardHandle,
  runNormalizeForwardHandle,
  runReluBackwardHandle,
  runReluForwardHandle,
  runStyleLossBackwardHandle,
  scaleHandle,
} from "../../runtime/handleOps/firstPoolHandleOps";

export const runFirstPoolOptimizer = async (
  payload: Extract<WorkerRequest, { type: "run-first-pool-optimizer" }>,
): Promise<{ losses: number[]; finalValues: number[] }> => {
  const conv1Weight = createTensor(payload.conv1Weight.shape, payload.conv1Weight.values);
  const conv2Weight = createTensor(payload.conv2Weight.shape, payload.conv2Weight.values);
  const conv3Weight = createTensor(payload.conv3Weight.shape, payload.conv3Weight.values);

  const styleInput = acquireTensorHandleFromCpu(payload.inputShape, new Float32Array(payload.styleImageValues));
  const contentInput = acquireTensorHandleFromCpu(payload.inputShape, new Float32Array(payload.contentImageValues));
  const styleNorm = await runNormalizeForwardHandle(styleInput, payload.mean, payload.std);
  const contentNorm = await runNormalizeForwardHandle(contentInput, payload.mean, payload.std);
  releaseTensorHandle(styleInput);
  releaseTensorHandle(contentInput);

  const styleConv1 = await runConv2dForwardHandle(styleNorm, conv1Weight.values, conv1Weight.shape, payload.conv1Bias);
  const styleRelu1 = await runReluForwardHandle(styleConv1);
  const styleConv2 = await runConv2dForwardHandle(styleRelu1, conv2Weight.values, conv2Weight.shape, payload.conv2Bias);
  const styleRelu2 = await runReluForwardHandle(styleConv2);
  const stylePool = await runMaxPool2dForwardHandle(styleRelu2);
  const styleConv3 = await runConv2dForwardHandle(stylePool, conv3Weight.values, conv3Weight.shape, payload.conv3Bias);
  const styleRelu3 = await runReluForwardHandle(styleConv3);

  const contentConv1 = await runConv2dForwardHandle(contentNorm, conv1Weight.values, conv1Weight.shape, payload.conv1Bias);
  const contentRelu1 = await runReluForwardHandle(contentConv1);
  const contentConv2 = await runConv2dForwardHandle(contentRelu1, conv2Weight.values, conv2Weight.shape, payload.conv2Bias);
  const contentRelu2 = await runReluForwardHandle(contentConv2);

  releaseTensorHandle(styleNorm); releaseTensorHandle(contentNorm); releaseTensorHandle(styleConv1); releaseTensorHandle(styleConv2); releaseTensorHandle(stylePool); releaseTensorHandle(styleConv3);
  releaseTensorHandle(contentConv1); releaseTensorHandle(contentRelu1); releaseTensorHandle(contentConv2);

  let inputHandle = acquireTensorHandleFromCpu(payload.inputShape, new Float32Array(payload.initialInputValues));
  const losses: number[] = [];

  for (let step = 0; step < payload.steps; step += 1) {
    const norm = await runNormalizeForwardHandle(inputHandle, payload.mean, payload.std);
    const conv1 = await runConv2dForwardHandle(norm, conv1Weight.values, conv1Weight.shape, payload.conv1Bias);
    const relu1 = await runReluForwardHandle(conv1);
    const conv2 = await runConv2dForwardHandle(relu1, conv2Weight.values, conv2Weight.shape, payload.conv2Bias);
    const relu2 = await runReluForwardHandle(conv2);
    const pool = await runMaxPool2dForwardHandle(relu2);
    const conv3 = await runConv2dForwardHandle(pool, conv3Weight.values, conv3Weight.shape, payload.conv3Bias);
    const relu3 = await runReluForwardHandle(conv3);

    const styleLoss1 = (await readMseScalar(relu1, styleRelu1)) * payload.styleWeightConv1;
    const styleLoss3 = (await readMseScalar(relu3, styleRelu3)) * payload.styleWeightConv3;
    const contentLoss = (await readMseScalar(relu2, contentRelu2)) * payload.contentWeight;
    losses.push(styleLoss1 + styleLoss3 + contentLoss);

    const gStyle1Relu = await scaleHandle(await runStyleLossBackwardHandle(relu1, styleRelu1), payload.styleWeightConv1);
    const gStyle3Relu = await scaleHandle(await runStyleLossBackwardHandle(relu3, styleRelu3), payload.styleWeightConv3);
    const gContentRelu = await scaleHandle(await runContentLossBackwardHandle(relu2, contentRelu2), payload.contentWeight);
    const gStyle3 = await runReluBackwardHandle(conv3, gStyle3Relu);
    const gContent = await runReluBackwardHandle(conv2, gContentRelu);
    const gStyle1 = await runReluBackwardHandle(conv1, gStyle1Relu);

    const gPool = await runConv2dBackwardInputHandle([1, 64, 8, 8], gStyle3, conv3Weight.values, conv3Weight.shape);
    const gRelu2 = await runMaxPool2dBackwardHandle(relu2, gPool);
    const gConv2FromPool = await runReluBackwardHandle(conv2, gRelu2);
    const gConv2Total = await addHandles(gConv2FromPool, gContent);
    const gRelu1 = await runConv2dBackwardInputHandle([1, 64, 16, 16], gConv2Total, conv2Weight.values, conv2Weight.shape);
    const gConv1FromConv2 = await runReluBackwardHandle(conv1, gRelu1);
    const gConv1Total = await addHandles(gConv1FromConv2, gStyle1);
    const gNorm = await runConv2dBackwardInputHandle(payload.inputShape, gConv1Total, conv1Weight.values, conv1Weight.shape);
    const gInput = await runNormalizeBackwardHandle(gNorm, payload.std);

    const lrScaledGradHandle = await chainTensorScalarOp(gInput, "mul", payload.learningRate);
    const nextHandle = await chainTensorBinaryOp(inputHandle, lrScaledGradHandle, "sub");
    const clampedHandle = await chainClamp01(nextHandle);

    [norm, conv1, relu1, conv2, relu2, pool, conv3, relu3, gStyle1Relu, gStyle3Relu, gContentRelu, gStyle3, gContent, gStyle1, gPool, gRelu2, gConv2FromPool, gConv2Total, gRelu1, gConv1FromConv2, gConv1Total, gNorm, gInput, lrScaledGradHandle, nextHandle, inputHandle].forEach(releaseTensorHandle);
    inputHandle = clampedHandle;
  }

  const finalValues = await readImageSnapshotBoundary(inputHandle);
  [inputHandle, styleRelu1, styleRelu2, styleRelu3, contentRelu2].forEach(releaseTensorHandle);
  return { losses, finalValues: Array.from(finalValues) };
};
