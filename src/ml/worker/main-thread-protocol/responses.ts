import type {
  WorkerGpuDispatchCoverageRecord,
  WorkerResponse,
  WorkerRunStats,
  WorkerTensor,
} from "../../../types";

export const postResponse = (response: WorkerResponse): void => {
  self.postMessage(response);
};

const getErrorMessage = (error: unknown, fallback: string): string =>
  error instanceof Error ? error.message : fallback;

export const sendWorkerErrorResponse = (
  id: string,
  error: unknown,
  fallbackMessage = "Unknown worker error",
): void => {
  postResponse({
    type: "error",
    id,
    message: getErrorMessage(error, fallbackMessage),
  });
};

export const sendPongResponse = (id: string, timestamp: number): void => {
  postResponse({ type: "pong", id, timestamp });
};

export const sendTensorRoundtripResult = (
  id: string,
  tensor: WorkerTensor,
): void => {
  postResponse({
    type: "tensor-roundtrip-result",
    id,
    ok: true,
    tensor,
  });
};

export const sendTensorRoundtripError = (
  id: string,
  error: unknown,
  fallbackMessage = "Tensor roundtrip failed.",
): void => {
  postResponse({
    type: "tensor-roundtrip-result",
    id,
    ok: false,
    message: getErrorMessage(error, fallbackMessage),
  });
};

export const sendTensorOpResult = (
  id: string,
  result: { values: Float32Array } | { scalar: number },
): void => {
  if ("values" in result) {
    postResponse({
      type: "tensor-op-result",
      id,
      ok: true,
      values: Array.from(result.values),
    });
    return;
  }
  postResponse({
    type: "tensor-op-result",
    id,
    ok: true,
    scalar: result.scalar,
  });
};

export const sendTensorOpError = (
  id: string,
  error: unknown,
  fallbackMessage = "Tensor operation failed.",
): void => {
  postResponse({
    type: "tensor-op-result",
    id,
    ok: false,
    message: getErrorMessage(error, fallbackMessage),
  });
};

export const sendGpuDispatchCoverageResult = (
  id: string,
  snapshot: {
    enabled: boolean;
    records: WorkerGpuDispatchCoverageRecord[];
  },
): void => {
  postResponse({
    type: "gpu-dispatch-coverage-result",
    id,
    ok: true,
    enabled: snapshot.enabled,
    records: snapshot.records,
  });
};

export const sendResetGpuDispatchCoverageResult = (id: string): void => {
  postResponse({ type: "reset-gpu-dispatch-coverage-result", id, ok: true });
};

export const sendWebGpuInitResult = (
  id: string,
  ok: boolean,
  message: string,
): void => {
  postResponse({ type: "webgpu-init-result", id, ok, message });
};

export const sendRunFirstPoolOptimizerResult = (
  id: string,
  result:
    | {
        ok: true;
        losses: number[];
        finalValues: number[];
        stats?: {
          elapsedMs: number;
          avgStepMs: number;
          forwardMs: number;
          lossMs: number;
          backwardMs: number;
          updateMs: number;
          readbackMs: number;
          mandatoryReadbackMs: number;
          diagnosticsReadbackMs: number;
          steps: number;
        };
      }
    | { ok: false; message: string },
): void => {
  postResponse({ type: "run-first-pool-optimizer-result", id, ...result });
};

export const sendRunStyleTransferResult = (
  id: string,
  result:
    | {
        ok: true;
        losses: number[];
        finalValues: number[];
        stats: WorkerRunStats;
      }
    | { ok: false; message: string },
): void => {
  postResponse({ type: "run-style-transfer-result", id, ...result });
};

export const sendClearStyleTransferSessionResult = (
  id: string,
  result: { ok: true } | { ok: false; message: string },
): void => {
  postResponse({ type: "clear-style-transfer-session-result", id, ...result });
};
