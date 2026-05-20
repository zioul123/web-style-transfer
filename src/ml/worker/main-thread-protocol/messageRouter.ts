/// <reference lib="webworker" />

import type { WorkerRequest } from "../../../types";
import { createTensor } from "../../index";
import {
  runConv2dBackwardInput,
  runConv2dForward,
  runConv2dReluForward,
} from "../ops/convolution/conv2d.run";
import { runGramBackward, runGramMatrix } from "../ops/gram/gram.run";
import { runContentLossBackward } from "../ops/loss/contentLoss.run";
import { runMse } from "../ops/loss/mse.run";
import { runStyleLossBackward } from "../ops/loss/styleLoss.run";
import { runNormalizeBackward, runNormalizeForward } from "../ops/normalization/normalize.run";
import { runMaxPool2dBackward, runMaxPool2dForward } from "../ops/pooling/maxpool.run";
import { runReluBackward, runReluForward } from "../ops/relu/relu.run";
import { acquireReusableBuffer, releaseReusableBuffer } from "../runtime/bufferPool";
import { BUFFER_USAGE_MAP_READ_COPY_DST, BUFFER_USAGE_STORAGE_COPY_DST, BUFFER_USAGE_STORAGE_COPY_SRC, BUFFER_USAGE_UNIFORM_COPY_DST, MAP_MODE_READ } from "../runtime/gpuFlags";
import { getTensorFromOperand, getValuesFromOperand, isBinaryTensorOpPayload } from "../runtime/operands";
import { runBinaryOp, runClamp, runScalarBinaryOp } from "../runtime/shaderRunner";
import { gpuDevice, initWebGpu, postResponse, runFirstPoolOptimizer, runStyleTransfer, runUnary } from "../legacyWorker";

export const routeWorkerMessage = (event: MessageEvent<WorkerRequest>): void => {

  const payload: WorkerRequest = event.data;
  switch (payload.type) {
    case "ping":
      postResponse({ type: "pong", id: payload.id, timestamp: Date.now() });
      break;
    case "init-webgpu":
      void initWebGpu(payload.id).catch((error: unknown) =>
        postResponse({
          type: "error",
          id: payload.id,
          message:
            error instanceof Error ? error.message : "Unknown worker error",
        }),
      );
      break;
    case "tensor-roundtrip": {
      try {
        const tensor = createTensor(
          payload.tensor.shape,
          payload.tensor.values,
        );
        postResponse({
          type: "tensor-roundtrip-result",
          id: payload.id,
          ok: true,
          tensor: { shape: tensor.shape, values: Array.from(tensor.values) },
        });
      } catch (error: unknown) {
        postResponse({
          type: "tensor-roundtrip-result",
          id: payload.id,
          ok: false,
          message:
            error instanceof Error ? error.message : "Tensor roundtrip failed.",
        });
      }
      break;
    }
    case "run-first-pool-optimizer": {
      void (async (): Promise<void> => {
        try {
          const result = await runFirstPoolOptimizer(payload);
          postResponse({
            type: "run-first-pool-optimizer-result",
            id: payload.id,
            ok: true,
            losses: result.losses,
            finalValues: result.finalValues,
          });
        } catch (error: unknown) {
          postResponse({
            type: "run-first-pool-optimizer-result",
            id: payload.id,
            ok: false,
            message:
              error instanceof Error
                ? error.message
                : "First-pool optimizer failed.",
          });
        }
      })();
      break;
    }
    case "run-style-transfer": {
      void (async (): Promise<void> => {
        try {
          const result = await runStyleTransfer(payload);
          postResponse({
            type: "run-style-transfer-result",
            id: payload.id,
            ok: true,
            losses: result.losses,
            finalValues: result.finalValues,
            stats: result.stats,
          });
        } catch (error: unknown) {
          postResponse({
            type: "run-style-transfer-result",
            id: payload.id,
            ok: false,
            message:
              error instanceof Error
                ? error.message
                : "Style transfer run failed.",
          });
        }
      })();
      break;
    }
    case "tensor-op": {
      void (async (): Promise<void> => {
        try {
          if (
            payload.op === "conv2d-forward" ||
            payload.op === "conv2d-relu-forward"
          ) {
            const input = createTensor(
              payload.input.shape,
              payload.input.values,
            );
            const weight = createTensor(
              payload.weight.shape,
              payload.weight.values,
            );
            const output =
              payload.op === "conv2d-relu-forward"
                ? await runConv2dReluForward(
                    gpuDevice,
                    runUnary,
                    input.values,
                    input.shape,
                    weight.values,
                    weight.shape,
                    payload.bias,
                    BUFFER_USAGE_STORAGE_COPY_DST,
                    BUFFER_USAGE_UNIFORM_COPY_DST,
                  )
                : await runConv2dForward(
                    gpuDevice,
                    runUnary,
                    input.values,
                    input.shape,
                    weight.values,
                    weight.shape,
                    payload.bias,
                    BUFFER_USAGE_STORAGE_COPY_DST,
                    BUFFER_USAGE_UNIFORM_COPY_DST,
                  );
            postResponse({
              type: "tensor-op-result",
              id: payload.id,
              ok: true,
              values: Array.from(output),
            });
            return;
          }
          if (payload.op === "relu-forward") {
            const input = createTensor(
              payload.input.shape,
              payload.input.values,
            );
            const output = await runReluForward(runUnary, input.values);
            postResponse({
              type: "tensor-op-result",
              id: payload.id,
              ok: true,
              values: Array.from(output),
            });
            return;
          }
          if (payload.op === "maxpool2d-forward") {
            const input = createTensor(
              payload.input.shape,
              payload.input.values,
            );
            const output = await runMaxPool2dForward(
              gpuDevice,
              runUnary,
              input.values,
              input.shape,
              BUFFER_USAGE_UNIFORM_COPY_DST,
            );
            postResponse({
              type: "tensor-op-result",
              id: payload.id,
              ok: true,
              values: Array.from(output),
            });
            return;
          }
          if (payload.op === "normalize-forward") {
            const input = createTensor(
              payload.input.shape,
              payload.input.values,
            );
            const output = await runNormalizeForward(
              gpuDevice,
              runUnary,
              input.values,
              input.shape,
              payload.mean,
              payload.std,
              BUFFER_USAGE_STORAGE_COPY_DST,
              BUFFER_USAGE_UNIFORM_COPY_DST,
            );
            postResponse({
              type: "tensor-op-result",
              id: payload.id,
              ok: true,
              values: Array.from(output),
            });
            return;
          }
          if (payload.op === "reshape-chw-flatten") {
            const input = createTensor(
              payload.input.shape,
              payload.input.values,
            );
            postResponse({
              type: "tensor-op-result",
              id: payload.id,
              ok: true,
              values: Array.from(input.values),
            });
            return;
          }
          if (payload.op === "gram-matrix") {
            const input = createTensor(
              payload.input.shape,
              payload.input.values,
            );
            const output = await runGramMatrix(
              gpuDevice,
              runUnary,
              input.values,
              input.shape,
              BUFFER_USAGE_UNIFORM_COPY_DST,
            );
            postResponse({
              type: "tensor-op-result",
              id: payload.id,
              ok: true,
              values: Array.from(output),
            });
            return;
          }
          if (payload.op === "content-loss") {
            const input = createTensor(
              payload.input.shape,
              payload.input.values,
            );
            const target = createTensor(
              payload.target.shape,
              payload.target.values,
            );
            const sameShape: boolean = input.shape.every(
              (value, index) => value === target.shape[index],
            );
            if (!sameShape) {
              postResponse({
                type: "tensor-op-result",
                id: payload.id,
                ok: true,
                scalar: 0,
              });
              return;
            }
            const mse = await runMse(
              gpuDevice,
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
            postResponse({
              type: "tensor-op-result",
              id: payload.id,
              ok: true,
              scalar: mse * contentWeight,
            });
            return;
          }
          if (payload.op === "relu-backward") {
            const input = createTensor(
              payload.input.shape,
              payload.input.values,
            );
            const gradOut = createTensor(
              payload.gradOut.shape,
              payload.gradOut.values,
            );
            const gradIn = await runReluBackward(
              gpuDevice,
              runUnary,
              input.values,
              gradOut.values,
            );
            postResponse({
              type: "tensor-op-result",
              id: payload.id,
              ok: true,
              values: Array.from(gradIn),
            });
            return;
          }
          if (payload.op === "maxpool2d-backward") {
            const input = createTensor(
              payload.input.shape,
              payload.input.values,
            );
            const gradOut = createTensor(
              payload.gradOut.shape,
              payload.gradOut.values,
            );
            const gradIn = await runMaxPool2dBackward(
              gpuDevice,
              runUnary,
              input.values,
              input.shape,
              gradOut.values,
              BUFFER_USAGE_STORAGE_COPY_DST,
              BUFFER_USAGE_UNIFORM_COPY_DST,
            );
            postResponse({
              type: "tensor-op-result",
              id: payload.id,
              ok: true,
              values: Array.from(gradIn),
            });
            return;
          }
          if (payload.op === "normalize-backward") {
            const gradOut = createTensor(
              payload.gradOut.shape,
              payload.gradOut.values,
            );
            const gradIn = await runNormalizeBackward(
              gpuDevice,
              runUnary,
              gradOut.values,
              gradOut.shape,
              payload.std,
              BUFFER_USAGE_STORAGE_COPY_DST,
              BUFFER_USAGE_UNIFORM_COPY_DST,
            );
            postResponse({
              type: "tensor-op-result",
              id: payload.id,
              ok: true,
              values: Array.from(gradIn),
            });
            return;
          }
          if (payload.op === "conv2d-backward-input") {
            const gradOut = createTensor(
              payload.gradOut.shape,
              payload.gradOut.values,
            );
            const weight = createTensor(
              payload.weight.shape,
              payload.weight.values,
            );
            const gradIn = await runConv2dBackwardInput(
              gpuDevice,
              runUnary,
              payload.inputShape,
              gradOut.values,
              weight.values,
              weight.shape,
              BUFFER_USAGE_STORAGE_COPY_DST,
              BUFFER_USAGE_UNIFORM_COPY_DST,
            );
            postResponse({
              type: "tensor-op-result",
              id: payload.id,
              ok: true,
              values: Array.from(gradIn),
            });
            return;
          }
          if (payload.op === "content-loss-backward") {
            const input = createTensor(
              payload.input.shape,
              payload.input.values,
            );
            const target = createTensor(
              payload.target.shape,
              payload.target.values,
            );
            const gradIn = await runContentLossBackward(
              gpuDevice,
              runUnary,
              input.values,
              target.values,
              BUFFER_USAGE_STORAGE_COPY_DST,
            );
            const contentWeight: number = payload.contentWeight ?? 1;
            const weightedGradIn =
              contentWeight === 1
                ? gradIn
                : await runScalarBinaryOp(
                    gpuDevice,
                    "mul",
                    gradIn,
                    contentWeight,
                    false,
                  );
            postResponse({
              type: "tensor-op-result",
              id: payload.id,
              ok: true,
              values: Array.from(weightedGradIn),
            });
            return;
          }
          if (payload.op === "gram-backward") {
            const input = createTensor(
              payload.input.shape,
              payload.input.values,
            );
            const gradOut = createTensor(
              payload.gradOut.shape,
              payload.gradOut.values,
            );
            const gradIn = await runGramBackward(
              gpuDevice,
              runUnary,
              input.values,
              input.shape,
              gradOut.values,
              BUFFER_USAGE_STORAGE_COPY_DST,
              BUFFER_USAGE_UNIFORM_COPY_DST,
            );
            postResponse({
              type: "tensor-op-result",
              id: payload.id,
              ok: true,
              values: Array.from(gradIn),
            });
            return;
          }
          if (payload.op === "style-loss-backward") {
            const input = createTensor(
              payload.input.shape,
              payload.input.values,
            );
            const target = createTensor(
              payload.target.shape,
              payload.target.values,
            );
            const gradIn = await runStyleLossBackward(
              gpuDevice,
              runUnary,
              input.values,
              input.shape,
              target.values,
              BUFFER_USAGE_STORAGE_COPY_DST,
              BUFFER_USAGE_UNIFORM_COPY_DST,
            );
            const styleWeight: number = payload.styleWeight ?? 1;
            const weightedGradIn =
              styleWeight === 1
                ? gradIn
                : await runScalarBinaryOp(
                    gpuDevice,
                    "mul",
                    gradIn,
                    styleWeight,
                    false,
                  );
            postResponse({
              type: "tensor-op-result",
              id: payload.id,
              ok: true,
              values: Array.from(weightedGradIn),
            });
            return;
          }
          if (payload.op === "style-loss") {
            const input = createTensor(
              payload.input.shape,
              payload.input.values,
            );
            const target = createTensor(
              payload.target.shape,
              payload.target.values,
            );
            const inputGram = await runGramMatrix(
              gpuDevice,
              runUnary,
              input.values,
              input.shape,
              BUFFER_USAGE_UNIFORM_COPY_DST,
            );
            const targetGram = await runGramMatrix(
              gpuDevice,
              runUnary,
              target.values,
              target.shape,
              BUFFER_USAGE_UNIFORM_COPY_DST,
            );
            const mse = await runMse(
              gpuDevice,
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
            postResponse({
              type: "tensor-op-result",
              id: payload.id,
              ok: true,
              scalar: mse * styleWeight,
            });
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
              gpuDevice,
              acquireReusableBuffer,
              releaseReusableBuffer,
              aOperand.values,
              bOperand.values,
              BUFFER_USAGE_STORAGE_COPY_DST,
              BUFFER_USAGE_STORAGE_COPY_SRC,
              BUFFER_USAGE_MAP_READ_COPY_DST,
              MAP_MODE_READ,
            );
            postResponse({
              type: "tensor-op-result",
              id: payload.id,
              ok: true,
              scalar: mse,
            });
            return;
          }
          if (payload.op === "clamp") {
            const v = await runClamp(
              gpuDevice,
              getValuesFromOperand(payload.a),
              payload.clampMin,
              payload.clampMax,
            );
            postResponse({
              type: "tensor-op-result",
              id: payload.id,
              ok: true,
              values: Array.from(v),
            });
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
              gpuDevice,
              payload.op,
              createTensor(payload.a.tensor.shape, payload.a.tensor.values)
                .values,
              createTensor(payload.b.tensor.shape, payload.b.tensor.values)
                .values,
              "tensorTensor",
            );
          } else if (
            payload.a.kind === "tensor" &&
            payload.b.kind === "scalar"
          ) {
            v = await runScalarBinaryOp(
              gpuDevice,
              payload.op,
              createTensor(payload.a.tensor.shape, payload.a.tensor.values)
                .values,
              payload.b.scalar,
              false,
            );
          } else if (
            payload.a.kind === "scalar" &&
            payload.b.kind === "tensor"
          ) {
            v = await runScalarBinaryOp(
              gpuDevice,
              payload.op,
              createTensor(payload.b.tensor.shape, payload.b.tensor.values)
                .values,
              payload.a.scalar,
              true,
            );
          } else {
            throw new Error(
              "At least one operand must be a tensor for binary tensor ops.",
            );
          }
          postResponse({
            type: "tensor-op-result",
            id: payload.id,
            ok: true,
            values: Array.from(v),
          });
        } catch (error: unknown) {
          postResponse({
            type: "tensor-op-result",
            id: payload.id,
            ok: false,
            message:
              error instanceof Error ? error.message : "Tensor op failed.",
          });
        }
      })();
      break;
    }
    default: {
      const exhaustivenessCheck: never = payload;
      postResponse({
        type: "error",
        id: "unknown",
        message: `Received unsupported message type: ${String(exhaustivenessCheck)}`,
      });
    }
  }
};

export const mountMessageRouter = (): void => {
  self.onmessage = (event: MessageEvent<WorkerRequest>): void => {
    routeWorkerMessage(event);
  };
};
