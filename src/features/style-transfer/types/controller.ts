import type { Dispatch, SetStateAction } from "react";
import type { WorkerKernelOptimizationFlags } from "../../../types/worker-protocol/pipelines";
import type { WorkerRunStats } from "../../../types/worker-protocol/pipelines";
import type { ModelCachePackStatus } from "../modelCache";
import type { VggPackName } from "../modelPacks";

export type ResolutionPreset =
  | "128x128"
  | "128x160"
  | "128x192"
  | "160x128"
  | "192x128"
  | "256x256"
  | "256x320"
  | "256x384"
  | "320x256"
  | "384x256";
export type ImageResolution = {
  readonly width: number;
  readonly height: number;
};
export type OptimizerMode = "sgd" | "adam" | "lbfgs";
export type KernelGramKernel = NonNullable<
  WorkerKernelOptimizationFlags["gramKernel"]
>;
export type KernelStyleBackward = NonNullable<
  WorkerKernelOptimizationFlags["styleBackward"]
>;
export type KernelConvForward = NonNullable<
  WorkerKernelOptimizationFlags["convForwardKernel"]
>;
export type KernelConvBackwardInput = NonNullable<
  WorkerKernelOptimizationFlags["convBackwardInputKernel"]
>;
export type KernelWeightStorage = NonNullable<
  WorkerKernelOptimizationFlags["weightStorage"]
>;
export type FullWeights = Record<
  string,
  number[] | [number, number, number, number]
>;

export interface StyleTransferControls {
  contentResolution: ResolutionPreset;
  setContentResolution: Dispatch<SetStateAction<ResolutionPreset>>;
  styleResolution: ResolutionPreset;
  setStyleResolution: Dispatch<SetStateAction<ResolutionPreset>>;
  stepsPerChunk: number;
  setStepsPerChunk: Dispatch<SetStateAction<number>>;
  contentWeight: number;
  setContentWeight: Dispatch<SetStateAction<number>>;
  styleWeight: number;
  setStyleWeight: Dispatch<SetStateAction<number>>;
  learningRate: number;
  setLearningRate: Dispatch<SetStateAction<number>>;
  selectedPack: VggPackName;
  setSelectedPack: Dispatch<SetStateAction<VggPackName>>;
  optimizer: OptimizerMode;
  setOptimizer: Dispatch<SetStateAction<OptimizerMode>>;
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
  synchronizePhaseTimings: boolean;
  setSynchronizePhaseTimings: Dispatch<SetStateAction<boolean>>;
  useCachedPipelines: boolean;
  setUseCachedPipelines: Dispatch<SetStateAction<boolean>>;
  usePersistentWeightBuffers: boolean;
  setUsePersistentWeightBuffers: Dispatch<SetStateAction<boolean>>;
  useStepBufferPool: boolean;
  setUseStepBufferPool: Dispatch<SetStateAction<boolean>>;
  useVec4Pointwise: boolean;
  setUseVec4Pointwise: Dispatch<SetStateAction<boolean>>;
  usePoolBackwardScatter: boolean;
  setUsePoolBackwardScatter: Dispatch<SetStateAction<boolean>>;
  gramKernel: KernelGramKernel;
  setGramKernel: Dispatch<SetStateAction<KernelGramKernel>>;
  styleBackward: KernelStyleBackward;
  setStyleBackward: Dispatch<SetStateAction<KernelStyleBackward>>;
  convForwardKernel: KernelConvForward;
  setConvForwardKernel: Dispatch<SetStateAction<KernelConvForward>>;
  convBackwardInputKernel: KernelConvBackwardInput;
  setConvBackwardInputKernel: Dispatch<SetStateAction<KernelConvBackwardInput>>;
  weightStorage: KernelWeightStorage;
  setWeightStorage: Dispatch<SetStateAction<KernelWeightStorage>>;
  kernelConfigSummary: string;
  kernelFlags: WorkerKernelOptimizationFlags | undefined;
  resetOptimizerState: () => void;
  resetOutputImage: () => void;
  clearModelCache: () => Promise<void>;
}

export interface StyleTransferStatus {
  workerStatus: string;
  gpuStatus: string;
  gpuInfo: string;
  iterations: number;
  isRunning: boolean;
  lastLoss: number | null;
  runStats: WorkerRunStats | null;
  modelCacheState: "downloading" | "cached" | "ready";
  modelCacheBytes: number;
  modelCachePackStatuses: ModelCachePackStatus[];
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
