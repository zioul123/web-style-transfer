import type {
  WorkerRoundtripResponse,
  WorkerTensor,
  WorkerTensorOpResponse,
} from "./core";
import type {
  WorkerClearStyleTransferSessionRequest,
  WorkerFirstPoolBenchmarkStats,
  WorkerRunStats,
  WorkerRunFirstPoolOptimizerRequest,
  WorkerRunStyleTransferRequest,
} from "./pipelines";
import type { WorkerTensorOpRequest } from "./tensor-ops";

export type WorkerGpuDispatchCoverageRecord = {
  label: string;
  workgroups: readonly [number, number, number];
  count: number;
};

export type WorkerRequest =
  | { type: "ping"; id: string }
  | { type: "init-webgpu"; id: string }
  | { type: "tensor-roundtrip"; id: string; tensor: WorkerTensor }
  | { type: "get-gpu-dispatch-coverage"; id: string }
  | { type: "reset-gpu-dispatch-coverage"; id: string }
  | WorkerRunFirstPoolOptimizerRequest
  | WorkerRunStyleTransferRequest
  | WorkerClearStyleTransferSessionRequest
  | WorkerTensorOpRequest;

export type WorkerResponse =
  | { type: "pong"; id: string; timestamp: number }
  | { type: "webgpu-init-result"; id: string; ok: boolean; message: string }
  | WorkerRoundtripResponse
  | WorkerTensorOpResponse
  | {
      type: "gpu-dispatch-coverage-result";
      id: string;
      ok: true;
      enabled: boolean;
      records: WorkerGpuDispatchCoverageRecord[];
    }
  | { type: "reset-gpu-dispatch-coverage-result"; id: string; ok: true }
  | { type: "error"; id: string; message: string }
  | {
      type: "run-first-pool-optimizer-result";
      id: string;
      ok: true;
      losses: number[];
      finalValues: number[];
      stats?: WorkerFirstPoolBenchmarkStats;
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
    }
  | {
      type: "clear-style-transfer-session-result";
      id: string;
      ok: true;
    }
  | {
      type: "clear-style-transfer-session-result";
      id: string;
      ok: false;
      message: string;
    };
