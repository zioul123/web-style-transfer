import type {
  KernelConvBackwardInput,
  KernelConvForward,
  KernelGramKernel,
  KernelStyleBackward,
  KernelWeightStorage,
  OptimizerMode,
  ResolutionPreset,
} from "./types/controller";
import { VGG_PACK_OPTIONS, type VggPackName } from "./modelPacks";

export const resolutionOptions: readonly ResolutionPreset[] = [
  "128x128",
  "128x160",
  "128x192",
  "160x128",
  "192x128",
  "256x256",
  "256x320",
  "256x384",
  "320x256",
  "384x256",
];
export const optimizerOptions: readonly OptimizerMode[] = [
  "sgd",
  "adam",
  "lbfgs",
];
export const hostedVggPackOptions: readonly VggPackName[] = [
  "int8-per-channel",
  "int4log-experimental",
];
export const kernelGramKernelOptions: readonly KernelGramKernel[] = [
  "scalar",
  "parallel-dot",
  "symmetric-parallel-dot",
];
export const kernelStyleBackwardOptions: readonly KernelStyleBackward[] = [
  "two-pass",
  "fused-from-gram-diff",
];
export const kernelConvForwardOptions: readonly KernelConvForward[] = [
  "scalar",
  "spatial-vec4",
  "tiled-spatial",
];
export const kernelConvBackwardInputOptions: readonly KernelConvBackwardInput[] =
  ["scalar", "spatial-vec4", "transposed-weight-spatial-vec4"];
export const kernelWeightStorageOptions: readonly KernelWeightStorage[] = [
  "fp32",
  "fp16-storage",
  "int8-dequant-experimental",
];

export type VggPackOption = (typeof VGG_PACK_OPTIONS)[number];

export const isResolutionPreset = (value: string): value is ResolutionPreset =>
  resolutionOptions.some((option) => option === value);
export const isOptimizerMode = (value: string): value is OptimizerMode =>
  optimizerOptions.some((option) => option === value);
export const isVggPackName = (value: string): value is VggPackName =>
  VGG_PACK_OPTIONS.some((option) => option.name === value);
export const isHostedVggPackName = (value: VggPackName): boolean =>
  hostedVggPackOptions.some((option) => option === value);
export const isLocalhost = (): boolean => {
  const hostname = window.location.hostname;
  return (
    hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1"
  );
};
export const isKernelGramKernel = (value: string): value is KernelGramKernel =>
  kernelGramKernelOptions.some((option) => option === value);
export const isKernelStyleBackward = (
  value: string,
): value is KernelStyleBackward =>
  kernelStyleBackwardOptions.some((option) => option === value);
export const isKernelConvForward = (
  value: string,
): value is KernelConvForward =>
  kernelConvForwardOptions.some((option) => option === value);
export const isKernelConvBackwardInput = (
  value: string,
): value is KernelConvBackwardInput =>
  kernelConvBackwardInputOptions.some((option) => option === value);
export const isKernelWeightStorage = (
  value: string,
): value is KernelWeightStorage =>
  kernelWeightStorageOptions.some((option) => option === value);

export const kernelConvForwardLabel = (option: KernelConvForward): string => {
  if (option === "spatial-vec4") return "spatial-vec4 (batched scalar)";
  if (option === "tiled-spatial") return "tiled-spatial (unsupported)";
  return option;
};

export const kernelConvBackwardInputLabel = (
  option: KernelConvBackwardInput,
): string => {
  if (option === "spatial-vec4") return "spatial-vec4 (unsupported)";
  return option;
};

export const kernelWeightStorageLabel = (
  option: KernelWeightStorage,
): string => {
  if (option === "fp16-storage") return "fp16-storage (requires persistent)";
  if (option === "int8-dequant-experimental") {
    return "int8-dequant-experimental (unsupported)";
  }
  return option;
};

export const formatBytes = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
};
