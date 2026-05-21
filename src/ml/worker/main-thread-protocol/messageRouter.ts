/// <reference lib="webworker" />

import type { WorkerRequest } from "../../../types";
import { createTensor } from "../../index";
import { runFirstPoolOptimizer } from "../pipelines/optimization/firstPoolOptimizer";
import { runStyleTransfer } from "../pipelines/optimization/styleTransferPipeline";
import { initWebGpu } from "./initWebGpu";
import { postResponse, sendRunFirstPoolOptimizerResult, sendRunStyleTransferResult } from "./responses";
import { routeTensorOp } from "./tensorOpRouter";

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
          message: error instanceof Error ? error.message : "Unknown worker error",
        }),
      );
      break;
    case "tensor-roundtrip": {
      try {
        const tensor = createTensor(payload.tensor.shape, payload.tensor.values);
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
          message: error instanceof Error ? error.message : "Tensor roundtrip failed.",
        });
      }
      break;
    }
    case "run-first-pool-optimizer": {
      void (async (): Promise<void> => {
        try {
          const result = await runFirstPoolOptimizer(payload);
          sendRunFirstPoolOptimizerResult(payload.id, { ok: true, losses: result.losses, finalValues: result.finalValues });
        } catch (error: unknown) {
          sendRunFirstPoolOptimizerResult(payload.id, { ok: false, message: error instanceof Error ? error.message : "First-pool optimizer failed." });
        }
      })();
      break;
    }
    case "run-style-transfer": {
      void (async (): Promise<void> => {
        try {
          const result = await runStyleTransfer(payload);
          sendRunStyleTransferResult(payload.id, { ok: true, losses: result.losses, finalValues: result.finalValues, stats: result.stats });
        } catch (error: unknown) {
          sendRunStyleTransferResult(payload.id, { ok: false, message: error instanceof Error ? error.message : "Style transfer run failed." });
        }
      })();
      break;
    }
    case "tensor-op": {
      void routeTensorOp(payload);
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
