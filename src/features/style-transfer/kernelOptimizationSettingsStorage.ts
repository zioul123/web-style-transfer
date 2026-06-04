import type { WorkerKernelOptimizationFlags } from "../../types";
import {
  isKernelConvBackwardInput,
  isKernelConvForward,
  isKernelGramKernel,
  isKernelStyleBackward,
  isKernelWeightStorage,
} from "./uiOptions";
import type {
  KernelConvBackwardInput,
  KernelConvForward,
  KernelGramKernel,
  KernelStyleBackward,
  KernelWeightStorage,
} from "./types/controller";

export const KERNEL_OPTIMIZATION_SETTINGS_STORAGE_KEY =
  "web-style-transfer.kernel-optimization-settings.v1";

export type KernelOptimizationSettings = {
  readonly useCachedPipelines: boolean;
  readonly usePersistentWeightBuffers: boolean;
  readonly useStepBufferPool: boolean;
  readonly useVec4Pointwise: boolean;
  readonly usePoolBackwardScatter: boolean;
  readonly gramKernel: KernelGramKernel;
  readonly styleBackward: KernelStyleBackward;
  readonly convForwardKernel: KernelConvForward;
  readonly convBackwardInputKernel: KernelConvBackwardInput;
  readonly weightStorage: KernelWeightStorage;
};

export const BASELINE_KERNEL_OPTIMIZATION_SETTINGS: KernelOptimizationSettings =
  {
    useCachedPipelines: false,
    usePersistentWeightBuffers: false,
    useStepBufferPool: false,
    useVec4Pointwise: false,
    usePoolBackwardScatter: false,
    gramKernel: "scalar",
    styleBackward: "two-pass",
    convForwardKernel: "scalar",
    convBackwardInputKernel: "scalar",
    weightStorage: "fp32",
  };

export const DEFAULT_KERNEL_OPTIMIZATION_SETTINGS: KernelOptimizationSettings =
  {
    useCachedPipelines: true,
    usePersistentWeightBuffers: true,
    useStepBufferPool: true,
    useVec4Pointwise: true,
    usePoolBackwardScatter: true,
    gramKernel: "symmetric-parallel-dot",
    styleBackward: "fused-from-gram-diff",
    convForwardKernel: "scalar",
    convBackwardInputKernel: "scalar",
    weightStorage: "fp32",
  };

type KernelOptimizationSettingsParseResult =
  | { readonly ok: true; readonly settings: KernelOptimizationSettings }
  | { readonly ok: false };

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const readBoolean = (
  record: Record<string, unknown>,
  key: keyof KernelOptimizationSettings,
  fallback: boolean,
): boolean => {
  const value = record[key];
  return typeof value === "boolean" ? value : fallback;
};

const readStringOption = <T extends string>(
  record: Record<string, unknown>,
  key: keyof KernelOptimizationSettings,
  fallback: T,
  isOption: (value: string) => value is T,
): T => {
  const value = record[key];
  if (typeof value !== "string") return fallback;
  return isOption(value) ? value : fallback;
};

export const parseKernelOptimizationSettings = (
  value: unknown,
): KernelOptimizationSettingsParseResult => {
  if (!isRecord(value)) return { ok: false };
  return {
    ok: true,
    settings: {
      useCachedPipelines: readBoolean(
        value,
        "useCachedPipelines",
        DEFAULT_KERNEL_OPTIMIZATION_SETTINGS.useCachedPipelines,
      ),
      usePersistentWeightBuffers: readBoolean(
        value,
        "usePersistentWeightBuffers",
        DEFAULT_KERNEL_OPTIMIZATION_SETTINGS.usePersistentWeightBuffers,
      ),
      useStepBufferPool: readBoolean(
        value,
        "useStepBufferPool",
        DEFAULT_KERNEL_OPTIMIZATION_SETTINGS.useStepBufferPool,
      ),
      useVec4Pointwise: readBoolean(
        value,
        "useVec4Pointwise",
        DEFAULT_KERNEL_OPTIMIZATION_SETTINGS.useVec4Pointwise,
      ),
      usePoolBackwardScatter: readBoolean(
        value,
        "usePoolBackwardScatter",
        DEFAULT_KERNEL_OPTIMIZATION_SETTINGS.usePoolBackwardScatter,
      ),
      gramKernel: readStringOption(
        value,
        "gramKernel",
        DEFAULT_KERNEL_OPTIMIZATION_SETTINGS.gramKernel,
        isKernelGramKernel,
      ),
      styleBackward: readStringOption(
        value,
        "styleBackward",
        DEFAULT_KERNEL_OPTIMIZATION_SETTINGS.styleBackward,
        isKernelStyleBackward,
      ),
      convForwardKernel: readStringOption(
        value,
        "convForwardKernel",
        DEFAULT_KERNEL_OPTIMIZATION_SETTINGS.convForwardKernel,
        isKernelConvForward,
      ),
      convBackwardInputKernel: readStringOption(
        value,
        "convBackwardInputKernel",
        DEFAULT_KERNEL_OPTIMIZATION_SETTINGS.convBackwardInputKernel,
        isKernelConvBackwardInput,
      ),
      weightStorage: readStringOption(
        value,
        "weightStorage",
        DEFAULT_KERNEL_OPTIMIZATION_SETTINGS.weightStorage,
        isKernelWeightStorage,
      ),
    },
  };
};

export const readKernelOptimizationSettings =
  (): KernelOptimizationSettings => {
    const storedSettings = localStorage.getItem(
      KERNEL_OPTIMIZATION_SETTINGS_STORAGE_KEY,
    );
    if (storedSettings === null) return DEFAULT_KERNEL_OPTIMIZATION_SETTINGS;
    try {
      const parsed = parseKernelOptimizationSettings(
        JSON.parse(storedSettings),
      );
      return parsed.ok ? parsed.settings : DEFAULT_KERNEL_OPTIMIZATION_SETTINGS;
    } catch {
      return DEFAULT_KERNEL_OPTIMIZATION_SETTINGS;
    }
  };

export const writeKernelOptimizationSettings = (
  settings: KernelOptimizationSettings,
): void => {
  localStorage.setItem(
    KERNEL_OPTIMIZATION_SETTINGS_STORAGE_KEY,
    JSON.stringify(settings),
  );
};

export const kernelFlagsToSettings = (
  kernelFlags: WorkerKernelOptimizationFlags | undefined,
): KernelOptimizationSettings => ({
  ...BASELINE_KERNEL_OPTIMIZATION_SETTINGS,
  useCachedPipelines: kernelFlags?.useCachedPipelines ?? false,
  usePersistentWeightBuffers: kernelFlags?.usePersistentWeightBuffers ?? false,
  useStepBufferPool: kernelFlags?.useStepBufferPool ?? false,
  useVec4Pointwise: kernelFlags?.useVec4Pointwise ?? false,
  usePoolBackwardScatter: kernelFlags?.usePoolBackwardScatter ?? false,
  gramKernel: kernelFlags?.gramKernel ?? "scalar",
  styleBackward: kernelFlags?.styleBackward ?? "two-pass",
  convForwardKernel: kernelFlags?.convForwardKernel ?? "scalar",
  convBackwardInputKernel: kernelFlags?.convBackwardInputKernel ?? "scalar",
  weightStorage: kernelFlags?.weightStorage ?? "fp32",
});

export const settingsToKernelFlags = (
  settings: KernelOptimizationSettings,
): WorkerKernelOptimizationFlags | undefined => {
  const flags: WorkerKernelOptimizationFlags = {};
  if (settings.useCachedPipelines) flags.useCachedPipelines = true;
  if (settings.usePersistentWeightBuffers) {
    flags.usePersistentWeightBuffers = true;
  }
  if (settings.useStepBufferPool) flags.useStepBufferPool = true;
  if (settings.useVec4Pointwise) flags.useVec4Pointwise = true;
  if (settings.usePoolBackwardScatter) flags.usePoolBackwardScatter = true;
  if (settings.gramKernel !== "scalar") flags.gramKernel = settings.gramKernel;
  if (settings.styleBackward !== "two-pass") {
    flags.styleBackward = settings.styleBackward;
  }
  if (settings.convForwardKernel !== "scalar") {
    flags.convForwardKernel = settings.convForwardKernel;
  }
  if (settings.convBackwardInputKernel !== "scalar") {
    flags.convBackwardInputKernel = settings.convBackwardInputKernel;
  }
  if (settings.weightStorage !== "fp32")
    flags.weightStorage = settings.weightStorage;
  return Object.keys(flags).length > 0 ? flags : undefined;
};
