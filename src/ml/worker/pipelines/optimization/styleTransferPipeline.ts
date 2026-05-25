/// <reference lib="webworker" />

import type { WorkerRequest } from "../../../../types";
import { runStyleTransferPipeline } from "./style-transfer/pipeline";

type StyleTransferPayload = Extract<
  WorkerRequest,
  { type: "run-style-transfer" }
>;

export type StyleTransferRunResult = {
  losses: number[];
  finalValues: number[];
  stats: {
    elapsedMs: number;
    avgStepMs: number;
    forwardMs: number;
    backwardMs: number;
    lossMs: number;
    updateMs: number;
    clampMs: number;
    steps: number;
  };
};

export const runStyleTransfer = async (
  payload: StyleTransferPayload,
): Promise<StyleTransferRunResult> => runStyleTransferPipeline(payload);
