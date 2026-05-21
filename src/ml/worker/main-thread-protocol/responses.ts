import type { WorkerResponse, WorkerRunStats, WorkerTensor } from "../../../types";

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

export const sendTensorRoundtripResult = (id: string, tensor: WorkerTensor): void => {
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
    | { ok: true; losses: number[]; finalValues: number[] }
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
