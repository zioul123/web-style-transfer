export * from "./types/worker-protocol";

import type {
  WorkerRoundtripResponse,
  WorkerTensorOpResponse,
  WorkerTensor,
} from "./types/worker-protocol/core";
import type { WorkerTensorOpRequest } from "./types/worker-protocol/tensor-ops";
import type {
  WorkerRunFirstPoolOptimizerRequest,
  WorkerRunStats,
  WorkerRunStyleTransferRequest,
} from "./types/worker-protocol/pipelines";

export type WorkerRequest =
  | { type: "ping"; id: string }
  | { type: "init-webgpu"; id: string }
  | { type: "tensor-roundtrip"; id: string; tensor: WorkerTensor }
  | WorkerRunFirstPoolOptimizerRequest
  | WorkerRunStyleTransferRequest
  | WorkerTensorOpRequest;

export type WorkerResponse =
  | { type: "pong"; id: string; timestamp: number }
  | { type: "webgpu-init-result"; id: string; ok: boolean; message: string }
  | WorkerRoundtripResponse
  | WorkerTensorOpResponse
  | { type: "error"; id: string; message: string }
  | {
      type: "run-first-pool-optimizer-result";
      id: string;
      ok: true;
      losses: number[];
      finalValues: number[];
    }
  | {
      type: "run-first-pool-optimizer-result";
      id: string;
      ok: false;
      message: string;
    }
  | {
      type: "run-style-transfer-result";
      id: string;
      ok: true;
      losses: number[];
      finalValues: number[];
      stats: WorkerRunStats;
    }
  | {
      type: "run-style-transfer-result";
      id: string;
      ok: false;
      message: string;
    };
