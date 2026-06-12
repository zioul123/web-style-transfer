/// <reference lib="webworker" />

import type { WorkerRequest } from "../../../types";
import { createTensor } from "../../tensor";
import { runFirstPoolOptimizer } from "../pipelines/optimization/firstPoolOptimizer";
import {
  clearStyleTransferSession,
  runStyleTransfer,
} from "../pipelines/optimization/styleTransferPipeline";
import { initWebGpu } from "./initWebGpu";
import {
  sendClearStyleTransferSessionResult,
  sendPongResponse,
  sendRunFirstPoolOptimizerResult,
  sendRunStyleTransferResult,
  sendTensorRoundtripError,
  sendTensorRoundtripResult,
  sendWorkerErrorResponse,
} from "./responses";
import { routeTensorOp } from "./tensorOpRouter";

export const routeWorkerMessage = (
  event: MessageEvent<WorkerRequest>,
): void => {
  const payload: WorkerRequest = event.data;
  switch (payload.type) {
    case "ping":
      sendPongResponse(payload.id, Date.now());
      break;
    case "init-webgpu":
      void initWebGpu(payload.id).catch((error: unknown) =>
        sendWorkerErrorResponse(payload.id, error),
      );
      break;
    case "tensor-roundtrip": {
      try {
        const tensor = createTensor(
          payload.tensor.shape,
          payload.tensor.values,
        );
        sendTensorRoundtripResult(payload.id, {
          shape: tensor.shape,
          values: Array.from(tensor.values),
        });
      } catch (error: unknown) {
        sendTensorRoundtripError(payload.id, error);
      }
      break;
    }
    case "run-first-pool-optimizer": {
      void (async (): Promise<void> => {
        try {
          const result = await runFirstPoolOptimizer(payload);
          sendRunFirstPoolOptimizerResult(payload.id, {
            ok: true,
            losses: result.losses,
            finalValues: result.finalValues,
            stats: result.stats,
          });
        } catch (error: unknown) {
          sendRunFirstPoolOptimizerResult(payload.id, {
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
          sendRunStyleTransferResult(payload.id, {
            ok: true,
            losses: result.losses,
            finalValues: result.finalValues,
            stats: result.stats,
          });
        } catch (error: unknown) {
          sendRunStyleTransferResult(payload.id, {
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
    case "clear-style-transfer-session": {
      try {
        clearStyleTransferSession(payload.sessionId);
        sendClearStyleTransferSessionResult(payload.id, { ok: true });
      } catch (error: unknown) {
        sendClearStyleTransferSessionResult(payload.id, {
          ok: false,
          message:
            error instanceof Error
              ? error.message
              : "Failed to clear style transfer session.",
        });
      }
      break;
    }
    case "tensor-op": {
      void routeTensorOp(payload);
      break;
    }
    default: {
      const exhaustivenessCheck: never = payload;
      sendWorkerErrorResponse(
        "unknown",
        new Error(
          `Received unsupported message type: ${String(exhaustivenessCheck)}`,
        ),
      );
    }
  }
};

export const mountMessageRouter = (): void => {
  self.onmessage = (event: MessageEvent<WorkerRequest>): void => {
    routeWorkerMessage(event);
  };
};
