import type { Dispatch, SetStateAction } from "react";
import type { WorkerRunStats } from "../../../types/worker-protocol/pipelines";

export type ResolutionPreset = 128 | 256;
export type OptimizerMode = "sgd" | "adam" | "lbfgs";
export type FullWeights = Record<
  string,
  number[] | [number, number, number, number]
>;

export interface StyleTransferControls {
  resolution: ResolutionPreset;
  setResolution: Dispatch<SetStateAction<ResolutionPreset>>;
  stepsPerChunk: number;
  setStepsPerChunk: Dispatch<SetStateAction<number>>;
  contentWeight: number;
  setContentWeight: Dispatch<SetStateAction<number>>;
  styleWeight: number;
  setStyleWeight: Dispatch<SetStateAction<number>>;
  learningRate: number;
  setLearningRate: Dispatch<SetStateAction<number>>;
  optimizer: OptimizerMode;
  setOptimizer: Dispatch<SetStateAction<OptimizerMode>>;
  fusedOps: boolean;
  setFusedOps: Dispatch<SetStateAction<boolean>>;
  adamBeta1: number;
  setAdamBeta1: Dispatch<SetStateAction<number>>;
  adamBeta2: number;
  setAdamBeta2: Dispatch<SetStateAction<number>>;
  adamEpsilon: number;
  setAdamEpsilon: Dispatch<SetStateAction<number>>;
  lbfgsMemory: number;
  setLbfgsMemory: Dispatch<SetStateAction<number>>;
  lbfgsEpsilon: number;
  setLbfgsEpsilon: Dispatch<SetStateAction<number>>;
}

export interface StyleTransferStatus {
  workerStatus: string;
  gpuStatus: string;
  gpuInfo: string;
  iterations: number;
  isRunning: boolean;
  lastLoss: number | null;
  runStats: WorkerRunStats | null;
}

export interface StyleTransferImages {
  contentImage: string | null;
  styleImage: string | null;
  outputImage: string | null;
}

export interface UseStyleTransferControllerResult {
  controls: StyleTransferControls;
  status: StyleTransferStatus;
  images: StyleTransferImages;
  canRun: boolean;
  setIsRunning: Dispatch<SetStateAction<boolean>>;
  onUpload: (
    event: React.ChangeEvent<HTMLInputElement>,
    target: "content" | "style",
  ) => Promise<void>;
}
