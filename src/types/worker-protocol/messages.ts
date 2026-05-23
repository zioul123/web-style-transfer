import type {
  WorkerRoundtripResponse,
  WorkerTensor,
  WorkerTensorOpResponse,
} from "./core";
import type { WorkerFirstPoolBenchmarkStats, WorkerRunStats, WorkerRunFirstPoolOptimizerRequest, WorkerRunStyleTransferRequest } from "./pipelines";
import type { WorkerTensorOpRequest } from "./tensor-ops";

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
    };
