import type { WorkerRequest } from "../../../types";

type TensorOpRequest = Extract<WorkerRequest, { type: "tensor-op" }>;
import {
  createFeatureMatrix,
  createPointCloudKnn,
  createSurfacePoolMap,
  createTensor,
} from "../../tensor";
import {
  runConv2dBackwardInputReadback,
  runConv2dForwardReadback,
  runConv2dReluForwardReadback,
} from "../ops/convolution/conv2d.run";
import { runGramBackward, runGramMatrix } from "../ops/gram/gram.run";
import { runContentLossBackward } from "../ops/loss/contentLoss.run";
import { runMse } from "../ops/loss/mse.run";
import { runStyleLossBackward } from "../ops/loss/styleLoss.run";
import {
  runNormalizeBackward,
  runNormalizeForward,
} from "../ops/normalization/normalize.run";
import {
  runPointFeatureNormalizeBackwardReadback,
  runPointFeatureNormalizeForwardReadback,
  runPointwiseExpBackwardReadback,
  runPointwiseExpForwardReadback,
} from "../ops/pointcloud/featureMatrix.run";
import {
  runPointCloudConvBackwardFeaturesReadback,
  runPointCloudConvForwardReadback,
} from "../ops/pointcloud/pcConv.run";
import {
  runSurfacePoolBackwardReadback,
  runSurfacePoolForwardReadback,
} from "../ops/pointcloud/surfacePool.run";
import {
  runMaxPool2dBackward,
  runMaxPool2dForward,
} from "../ops/pooling/maxpool.run";
import { runReluBackward, runReluForward } from "../ops/relu/relu.run";
import {
  acquireReusableBuffer,
  releaseReusableBuffer,
} from "../runtime/bufferPool";
import {
  BUFFER_USAGE_MAP_READ_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_DST,
  BUFFER_USAGE_STORAGE_COPY_SRC,
  BUFFER_USAGE_UNIFORM_COPY_DST,
  MAP_MODE_READ,
} from "../runtime/gpuFlags";
import {
  getTensorFromOperand,
  getValuesFromOperand,
  isBinaryTensorOpPayload,
} from "../runtime/operands";
import {
  runBinaryOp,
  runClamp,
  runScalarBinaryOp,
} from "../runtime/shaderRunner";
import { runUnary } from "../runtime/computeContext";
import { getGpuDevice } from "../runtime/deviceState";
import { sendTensorOpError, sendTensorOpResult } from "./responses";

export const routeTensorOp = async (
  payload: TensorOpRequest,
): Promise<void> => {
  try {
    if (
      payload.op === "conv2d-forward" ||
      payload.op === "conv2d-relu-forward"
    ) {
      const input = createTensor(payload.input.shape, payload.input.values);
      const weight = createTensor(payload.weight.shape, payload.weight.values);
      const output =
        payload.op === "conv2d-relu-forward"
          ? await runConv2dReluForwardReadback(
              getGpuDevice(),
              runUnary,
              input.values,
              input.shape,
              weight.values,
              weight.shape,
              payload.bias,
              BUFFER_USAGE_STORAGE_COPY_DST,
              BUFFER_USAGE_UNIFORM_COPY_DST,
            )
          : await runConv2dForwardReadback(
              getGpuDevice(),
              runUnary,
              input.values,
              input.shape,
              weight.values,
              weight.shape,
              payload.bias,
            );
      sendTensorOpResult(payload.id, { values: output });
      return;
    }

    if (payload.op === "relu-forward") {
      const input = createTensor(payload.input.shape, payload.input.values);
      const output = await runReluForward(runUnary, input.values);
      sendTensorOpResult(payload.id, { values: output });
      return;
    }
    if (payload.op === "maxpool2d-forward") {
      const input = createTensor(payload.input.shape, payload.input.values);
      const output = await runMaxPool2dForward(
        getGpuDevice(),
        runUnary,
        input.values,
        input.shape,
      );
      sendTensorOpResult(payload.id, { values: output });
      return;
    }
    if (payload.op === "normalize-forward") {
      const input = createTensor(payload.input.shape, payload.input.values);
      const output = await runNormalizeForward(
        getGpuDevice(),
        runUnary,
        input.values,
        input.shape,
        payload.mean,
        payload.std,
      );
      sendTensorOpResult(payload.id, { values: output });
      return;
    }
    if (payload.op === "exp-forward") {
      const input = createFeatureMatrix(
        payload.input.pointCount,
        payload.input.channelCount,
        payload.input.values,
      );
      const output = await runPointwiseExpForwardReadback(input.values);
      sendTensorOpResult(payload.id, { values: output });
      return;
    }
    if (payload.op === "exp-backward") {
      const input = createFeatureMatrix(
        payload.input.pointCount,
        payload.input.channelCount,
        payload.input.values,
      );
      const gradOut = createFeatureMatrix(
        payload.gradOut.pointCount,
        payload.gradOut.channelCount,
        payload.gradOut.values,
      );
      if (
        input.pointCount !== gradOut.pointCount ||
        input.channelCount !== gradOut.channelCount
      ) {
        throw new Error("exp-backward input and gradOut shapes must match.");
      }
      const output = await runPointwiseExpBackwardReadback(
        input.values,
        gradOut.values,
      );
      sendTensorOpResult(payload.id, { values: output });
      return;
    }
    if (payload.op === "point-feature-normalize-forward") {
      const input = createFeatureMatrix(
        payload.input.pointCount,
        payload.input.channelCount,
        payload.input.values,
      );
      const output = await runPointFeatureNormalizeForwardReadback(
        input.values,
        { pointCount: input.pointCount, channelCount: input.channelCount },
        payload.mean,
        payload.std,
      );
      sendTensorOpResult(payload.id, { values: output });
      return;
    }
    if (payload.op === "point-feature-normalize-backward") {
      const gradOut = createFeatureMatrix(
        payload.gradOut.pointCount,
        payload.gradOut.channelCount,
        payload.gradOut.values,
      );
      const output = await runPointFeatureNormalizeBackwardReadback(
        gradOut.values,
        { pointCount: gradOut.pointCount, channelCount: gradOut.channelCount },
        payload.std,
      );
      sendTensorOpResult(payload.id, { values: output });
      return;
    }
    if (payload.op === "pc-conv-forward") {
      const input = createFeatureMatrix(
        payload.input.pointCount,
        payload.input.channelCount,
        payload.input.values,
      );
      const knn = createPointCloudKnn(
        payload.knn.sampleCount,
        payload.knn.kernelPointCount,
        payload.knn.neighborCount,
        payload.knn.indices,
        payload.knn.weights,
      );
      const weight = createTensor(payload.weight.shape, payload.weight.values);
      const output = await runPointCloudConvForwardReadback(
        input.values,
        { pointCount: input.pointCount, channelCount: input.channelCount },
        knn.indices,
        knn.weights,
        knn.sampleCount,
        knn.neighborCount,
        weight.values,
        weight.shape,
        payload.bias,
        payload.kernelIndexMap,
        payload.pcConvForwardKernel,
        payload.pcConvMaxIntermediateBytes,
      );
      sendTensorOpResult(payload.id, { values: output });
      return;
    }
    if (payload.op === "pc-conv-backward-features") {
      const gradOut = createFeatureMatrix(
        payload.gradOut.pointCount,
        payload.gradOut.channelCount,
        payload.gradOut.values,
      );
      const knn = createPointCloudKnn(
        payload.knn.sampleCount,
        payload.knn.kernelPointCount,
        payload.knn.neighborCount,
        payload.knn.indices,
        payload.knn.weights,
      );
      const weight = createTensor(payload.weight.shape, payload.weight.values);
      const output = await runPointCloudConvBackwardFeaturesReadback(
        gradOut.values,
        payload.inputShape,
        knn.indices,
        knn.weights,
        knn.sampleCount,
        knn.neighborCount,
        weight.values,
        weight.shape,
        payload.kernelIndexMap,
        payload.pcConvBackwardFeaturesKernel,
        payload.pcConvMaxIntermediateBytes,
      );
      sendTensorOpResult(payload.id, { values: output });
      return;
    }
    if (payload.op === "surface-pool-forward") {
      const input = createFeatureMatrix(
        payload.input.pointCount,
        payload.input.channelCount,
        payload.input.values,
      );
      const pool = createSurfacePoolMap(
        payload.pool.inputPointCount,
        payload.pool.outputPointCount,
        payload.pool.mapping,
      );
      const output = await runSurfacePoolForwardReadback(
        input.values,
        { pointCount: input.pointCount, channelCount: input.channelCount },
        pool.outputPointCount,
        pool.mapping,
      );
      sendTensorOpResult(payload.id, { values: output });
      return;
    }
    if (payload.op === "surface-pool-backward") {
      const input = createFeatureMatrix(
        payload.input.pointCount,
        payload.input.channelCount,
        payload.input.values,
      );
      const gradOut = createFeatureMatrix(
        payload.gradOut.pointCount,
        payload.gradOut.channelCount,
        payload.gradOut.values,
      );
      const pool = createSurfacePoolMap(
        payload.pool.inputPointCount,
        payload.pool.outputPointCount,
        payload.pool.mapping,
      );
      const output = await runSurfacePoolBackwardReadback(
        input.values,
        gradOut.values,
        { pointCount: input.pointCount, channelCount: input.channelCount },
        pool.outputPointCount,
        pool.mapping,
      );
      sendTensorOpResult(payload.id, { values: output });
      return;
    }
    if (payload.op === "reshape-chw-flatten") {
      const input = createTensor(payload.input.shape, payload.input.values);
      sendTensorOpResult(payload.id, { values: input.values });
      return;
    }
    if (payload.op === "gram-matrix") {
      const input = createTensor(payload.input.shape, payload.input.values);
      const output = await runGramMatrix(
        getGpuDevice(),
        runUnary,
        input.values,
        input.shape,
      );
      sendTensorOpResult(payload.id, { values: output });
      return;
    }
    if (payload.op === "content-loss") {
      const input = createTensor(payload.input.shape, payload.input.values);
      const target = createTensor(payload.target.shape, payload.target.values);
      const sameShape: boolean = input.shape.every(
        (value, index) => value === target.shape[index],
      );
      if (!sameShape) {
        sendTensorOpResult(payload.id, { scalar: 0 });
        return;
      }
      const mse = await runMse(
        getGpuDevice(),
        acquireReusableBuffer,
        releaseReusableBuffer,
        input.values,
        target.values,
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_STORAGE_COPY_SRC,
        BUFFER_USAGE_MAP_READ_COPY_DST,
        MAP_MODE_READ,
      );
      const contentWeight: number = payload.contentWeight ?? 1;
      sendTensorOpResult(payload.id, { scalar: mse * contentWeight });
      return;
    }

    if (payload.op === "relu-backward") {
      const input = createTensor(payload.input.shape, payload.input.values);
      const gradOut = createTensor(
        payload.gradOut.shape,
        payload.gradOut.values,
      );
      const gradIn = await runReluBackward(
        getGpuDevice(),
        runUnary,
        input.values,
        gradOut.values,
      );
      sendTensorOpResult(payload.id, { values: gradIn });
      return;
    }
    if (payload.op === "maxpool2d-backward") {
      const input = createTensor(payload.input.shape, payload.input.values);
      const gradOut = createTensor(
        payload.gradOut.shape,
        payload.gradOut.values,
      );
      const gradIn = await runMaxPool2dBackward(
        getGpuDevice(),
        runUnary,
        input.values,
        input.shape,
        gradOut.values,
      );
      sendTensorOpResult(payload.id, { values: gradIn });
      return;
    }
    if (payload.op === "normalize-backward") {
      const gradOut = createTensor(
        payload.gradOut.shape,
        payload.gradOut.values,
      );
      const gradIn = await runNormalizeBackward(
        getGpuDevice(),
        runUnary,
        gradOut.values,
        gradOut.shape,
        payload.std,
      );
      sendTensorOpResult(payload.id, { values: gradIn });
      return;
    }
    if (payload.op === "conv2d-backward-input") {
      const gradOut = createTensor(
        payload.gradOut.shape,
        payload.gradOut.values,
      );
      const weight = createTensor(payload.weight.shape, payload.weight.values);
      const gradIn = await runConv2dBackwardInputReadback(
        getGpuDevice(),
        runUnary,
        payload.inputShape,
        gradOut.values,
        weight.values,
        weight.shape,
      );
      sendTensorOpResult(payload.id, { values: gradIn });
      return;
    }
    if (payload.op === "content-loss-backward") {
      const input = createTensor(payload.input.shape, payload.input.values);
      const target = createTensor(payload.target.shape, payload.target.values);
      const gradIn = await runContentLossBackward(
        getGpuDevice(),
        runUnary,
        input.values,
        target.values,
      );
      const contentWeight: number = payload.contentWeight ?? 1;
      const weightedGradIn =
        contentWeight === 1
          ? gradIn
          : await runScalarBinaryOp(
              getGpuDevice(),
              "mul",
              gradIn,
              contentWeight,
              false,
            );
      sendTensorOpResult(payload.id, { values: weightedGradIn });
      return;
    }
    if (payload.op === "gram-backward") {
      const input = createTensor(payload.input.shape, payload.input.values);
      const gradOut = createTensor(
        payload.gradOut.shape,
        payload.gradOut.values,
      );
      const gradIn = await runGramBackward(
        getGpuDevice(),
        runUnary,
        input.values,
        input.shape,
        gradOut.values,
      );
      sendTensorOpResult(payload.id, { values: gradIn });
      return;
    }
    if (payload.op === "style-loss-backward") {
      const input = createTensor(payload.input.shape, payload.input.values);
      const target = createTensor(payload.target.shape, payload.target.values);
      const gradIn = await runStyleLossBackward(
        getGpuDevice(),
        runUnary,
        input.values,
        input.shape,
        target.values,
      );
      const styleWeight: number = payload.styleWeight ?? 1;
      const weightedGradIn =
        styleWeight === 1
          ? gradIn
          : await runScalarBinaryOp(
              getGpuDevice(),
              "mul",
              gradIn,
              styleWeight,
              false,
            );
      sendTensorOpResult(payload.id, { values: weightedGradIn });
      return;
    }
    if (payload.op === "style-loss") {
      const input = createTensor(payload.input.shape, payload.input.values);
      const target = createTensor(payload.target.shape, payload.target.values);
      const inputGram = await runGramMatrix(
        getGpuDevice(),
        runUnary,
        input.values,
        input.shape,
      );
      const targetGram = await runGramMatrix(
        getGpuDevice(),
        runUnary,
        target.values,
        target.shape,
      );
      const mse = await runMse(
        getGpuDevice(),
        acquireReusableBuffer,
        releaseReusableBuffer,
        inputGram,
        targetGram,
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_STORAGE_COPY_SRC,
        BUFFER_USAGE_MAP_READ_COPY_DST,
        MAP_MODE_READ,
      );
      const styleWeight: number = payload.styleWeight ?? 1;
      sendTensorOpResult(payload.id, { scalar: mse * styleWeight });
      return;
    }
    if (payload.op === "mse") {
      const aOperand = getTensorFromOperand(payload.a);
      if (!aOperand.ok)
        throw new Error("MSE requires both operands to be tensors.");
      const bOperand = getTensorFromOperand(payload.b);
      if (!bOperand.ok)
        throw new Error("MSE requires both operands to be tensors.");
      const mse = await runMse(
        getGpuDevice(),
        acquireReusableBuffer,
        releaseReusableBuffer,
        aOperand.values,
        bOperand.values,
        BUFFER_USAGE_STORAGE_COPY_DST,
        BUFFER_USAGE_STORAGE_COPY_SRC,
        BUFFER_USAGE_MAP_READ_COPY_DST,
        MAP_MODE_READ,
      );
      sendTensorOpResult(payload.id, { scalar: mse });
      return;
    }
    if (payload.op === "clamp") {
      const v = await runClamp(
        getGpuDevice(),
        getValuesFromOperand(payload.a),
        payload.clampMin,
        payload.clampMax,
      );
      sendTensorOpResult(payload.id, { values: v });
      return;
    }
    if (!isBinaryTensorOpPayload(payload)) {
      throw new Error(`Unsupported tensor op: ${payload.op}`);
    }
    if (payload.a.kind === "scalar" && payload.b.kind === "scalar") {
      throw new Error(
        "At least one operand must be a tensor for binary tensor ops.",
      );
    }
    let v: Float32Array;
    if (payload.a.kind === "tensor" && payload.b.kind === "tensor") {
      v = await runBinaryOp(
        getGpuDevice(),
        payload.op,
        createTensor(payload.a.tensor.shape, payload.a.tensor.values).values,
        createTensor(payload.b.tensor.shape, payload.b.tensor.values).values,
        "tensorTensor",
      );
    } else if (payload.a.kind === "tensor" && payload.b.kind === "scalar") {
      v = await runScalarBinaryOp(
        getGpuDevice(),
        payload.op,
        createTensor(payload.a.tensor.shape, payload.a.tensor.values).values,
        payload.b.scalar,
        false,
      );
    } else if (payload.a.kind === "scalar" && payload.b.kind === "tensor") {
      v = await runScalarBinaryOp(
        getGpuDevice(),
        payload.op,
        createTensor(payload.b.tensor.shape, payload.b.tensor.values).values,
        payload.a.scalar,
        true,
      );
    } else {
      throw new Error(
        "At least one operand must be a tensor for binary tensor ops.",
      );
    }
    sendTensorOpResult(payload.id, { values: v });
  } catch (error: unknown) {
    sendTensorOpError(payload.id, error, "Tensor op failed.");
  }
};
