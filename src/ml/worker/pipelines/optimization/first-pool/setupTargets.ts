import type { WorkerRequest } from "../../../../../types";
import { createTensor } from "../../../../index";
import { runConv2dReluForwardBuffer } from "../../../ops/convolution/conv2d.run";
import { runGramMatrixBuffer } from "../../../ops/gram/gram.run";
import { runNormalizeForwardBuffer } from "../../../ops/normalization/normalize.run";
import { runMaxPool2dForwardBuffer } from "../../../ops/pooling/maxpool.run";
import {
  readGpuBufferToArray,
  releaseOwnedBuffer,
  uploadToOwnedBuffer,
} from "../../../runtime/bufferKernels";
import {
  convOutputShape,
  elementCount,
  pooledShape,
} from "../../../runtime/tensorShapes";
import type { FirstPoolPersistentContext } from "./types";

export const setupFirstPoolTargets = async (
  device: GPUDevice,
  payload: Extract<WorkerRequest, { type: "run-first-pool-optimizer" }>,
): Promise<FirstPoolPersistentContext> => {
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
  const contentInputBuffer = uploadToOwnedBuffer(
    device,
    new Float32Array(payload.contentImageValues),
  );
  const styleInputBuffer = uploadToOwnedBuffer(
    device,
    new Float32Array(payload.styleImageValues),
  );
  let contentNormBuffer;
  let styleNormBuffer;
  let styleRelu1Buffer;
  let styleRelu2Buffer;
  let stylePoolBuffer;
  let styleRelu3Buffer;
  let contentRelu1Buffer;
  let contentRelu2Buffer;
  try {
    const relu1Shape = convOutputShape(
      payload.inputShape,
      conv1Weight.shape[0],
    );
    const relu2Shape = convOutputShape(relu1Shape, conv2Weight.shape[0]);
    const poolShape = pooledShape(relu2Shape);
    const relu3Shape = convOutputShape(poolShape, conv3Weight.shape[0]);

    contentNormBuffer = await runNormalizeForwardBuffer(
      contentInputBuffer,
      payload.inputShape,
      payload.mean,
      payload.std,
    );
    styleNormBuffer = await runNormalizeForwardBuffer(
      styleInputBuffer,
      payload.inputShape,
      payload.mean,
      payload.std,
    );
    styleRelu1Buffer = await runConv2dReluForwardBuffer(
      styleNormBuffer,
      payload.inputShape,
      conv1Weight.values,
      conv1Weight.shape,
      payload.conv1Bias,
    );
    styleRelu2Buffer = await runConv2dReluForwardBuffer(
      styleRelu1Buffer,
      relu1Shape,
      conv2Weight.values,
      conv2Weight.shape,
      payload.conv2Bias,
    );
    stylePoolBuffer = await runMaxPool2dForwardBuffer(
      styleRelu2Buffer,
      relu2Shape,
    );
    styleRelu3Buffer = await runConv2dReluForwardBuffer(
      stylePoolBuffer,
      poolShape,
      conv3Weight.values,
      conv3Weight.shape,
      payload.conv3Bias,
    );
    contentRelu1Buffer = await runConv2dReluForwardBuffer(
      contentNormBuffer,
      payload.inputShape,
      conv1Weight.values,
      conv1Weight.shape,
      payload.conv1Bias,
    );
    contentRelu2Buffer = await runConv2dReluForwardBuffer(
      contentRelu1Buffer,
      relu1Shape,
      conv2Weight.values,
      conv2Weight.shape,
      payload.conv2Bias,
    );
    const styleGram1Buffer = await runGramMatrixBuffer(
      styleRelu1Buffer,
      relu1Shape,
    );
    const styleGram3Buffer = await runGramMatrixBuffer(
      styleRelu3Buffer,
      relu3Shape,
    );
    const contentRelu2TargetBuffer = uploadToOwnedBuffer(
      device,
      await readGpuBufferToArray(
        device,
        contentRelu2Buffer.buffer,
        elementCount(relu2Shape),
      ),
    );
    return {
      conv1Weight,
      conv2Weight,
      conv3Weight,
      styleGram1Buffer,
      styleGram3Buffer,
      contentRelu2Buffer: contentRelu2TargetBuffer,
      dispose: () => {
        releaseOwnedBuffer(styleGram1Buffer);
        releaseOwnedBuffer(styleGram3Buffer);
        releaseOwnedBuffer(contentRelu2TargetBuffer);
      },
    };
  } finally {
    releaseOwnedBuffer(contentInputBuffer);
    releaseOwnedBuffer(styleInputBuffer);
    if (contentNormBuffer) releaseOwnedBuffer(contentNormBuffer);
    if (styleNormBuffer) releaseOwnedBuffer(styleNormBuffer);
    if (styleRelu1Buffer) releaseOwnedBuffer(styleRelu1Buffer);
    if (styleRelu2Buffer) releaseOwnedBuffer(styleRelu2Buffer);
    if (stylePoolBuffer) releaseOwnedBuffer(stylePoolBuffer);
    if (styleRelu3Buffer) releaseOwnedBuffer(styleRelu3Buffer);
    if (contentRelu1Buffer) releaseOwnedBuffer(contentRelu1Buffer);
    if (contentRelu2Buffer) releaseOwnedBuffer(contentRelu2Buffer);
  }
};
