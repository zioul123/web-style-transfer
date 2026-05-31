import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, Dispatch, SetStateAction } from "react";
import type { WorkerRequest, WorkerResponse } from "../../../types";
import {
  VGG19_RELU_TAP_CONTENT_LAYER_INDEX,
  VGG19_RELU_TAP_STYLE_LAYER_INDICES,
  assertValidVgg19ReluTapIndices,
} from "../../../ml/constants/vgg19";
import type {
  FullWeights,
  ImageResolution,
  KernelConvBackwardInput,
  KernelConvForward,
  KernelGramKernel,
  KernelStyleBackward,
  KernelWeightStorage,
  ResolutionPreset,
  UseStyleTransferControllerResult,
} from "../types/controller";
import {
  clearCachedVggModelPacks,
  DEFAULT_VGG_PACK,
  getCachedVggModelPackStatus,
  loadVgg19ManifestWeightsForPack,
  VGG_PACK_STORAGE_KEY,
  type VggPackName,
} from "../modelPacks";
import type { ModelCachePackStatus } from "../modelCache";

const DEFAULT_CONTENT_IMAGE_URL = new URL(
  "../../../../assets/madeira_900x1600.jpeg",
  import.meta.url,
).href;
const DEFAULT_STYLE_IMAGE_URL = new URL(
  "../../../../assets/starry_night_768x970.jpg",
  import.meta.url,
).href;

const RESOLUTION_PRESETS: Record<ResolutionPreset, ImageResolution> = {
  "128x128": { width: 128, height: 128 },
  "128x160": { width: 128, height: 160 },
  "128x192": { width: 128, height: 192 },
  "160x128": { width: 160, height: 128 },
  "192x128": { width: 192, height: 128 },
  "256x256": { width: 256, height: 256 },
  "256x320": { width: 256, height: 320 },
  "256x384": { width: 256, height: 384 },
  "320x256": { width: 320, height: 256 },
  "384x256": { width: 384, height: 256 },
};

const AUTO_RESOLUTION_RATIOS = [
  { ratio: 1, portrait: "128x128", landscape: "128x128" },
  { ratio: 1.25, portrait: "128x160", landscape: "160x128" },
  { ratio: 1.5, portrait: "128x192", landscape: "192x128" },
] as const satisfies readonly {
  readonly ratio: number;
  readonly portrait: ResolutionPreset;
  readonly landscape: ResolutionPreset;
}[];

const shapeForResolution = (
  resolution: ImageResolution,
): readonly [1, 3, number, number] => [
  1,
  3,
  resolution.height,
  resolution.width,
];

const resolvePresetUpdate = (
  update: SetStateAction<ResolutionPreset>,
  current: ResolutionPreset,
): ResolutionPreset =>
  typeof update === "function" ? update(current) : update;

const nearestResolutionPresetForImage = (
  image: HTMLImageElement,
): ResolutionPreset => {
  const width = Math.max(1, image.naturalWidth || image.width);
  const height = Math.max(1, image.naturalHeight || image.height);
  const aspectRatio = Math.max(width, height) / Math.min(width, height);
  const closest = AUTO_RESOLUTION_RATIOS.reduce((best, option) =>
    Math.abs(option.ratio - aspectRatio) < Math.abs(best.ratio - aspectRatio)
      ? option
      : best,
  );
  return height >= width ? closest.portrait : closest.landscape;
};

const createMessageId = (): string =>
  `${Date.now()}-${Math.random().toString(16).slice(2)}`;

const createOptimizationSessionId = (): string =>
  `style-transfer-session-${createMessageId()}`;

const toResolvedState = <T>(update: SetStateAction<T>, current: T): T =>
  typeof update === "function"
    ? (update as (previous: T) => T)(current)
    : update;

const readFileAsImage = (file: File): Promise<HTMLImageElement> =>
  new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file);
    const image = new Image();
    image.onload = (): void => {
      URL.revokeObjectURL(objectUrl);
      resolve(image);
    };
    image.onerror = (): void => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error(`Failed to load ${file.name}`));
    };
    image.src = objectUrl;
  });

const readUrlAsImage = (url: string): Promise<HTMLImageElement> =>
  new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = (): void => {
      resolve(image);
    };
    image.onerror = (): void => {
      reject(new Error(`Failed to load ${url}`));
    };
    image.src = url;
  });

const imageToTensorValues = (
  image: HTMLImageElement,
  resolution: ImageResolution,
): number[] => {
  const canvas = document.createElement("canvas");
  canvas.width = resolution.width;
  canvas.height = resolution.height;
  const context = canvas.getContext("2d");
  if (context === null) throw new Error("2D canvas unavailable.");
  context.drawImage(image, 0, 0, resolution.width, resolution.height);
  const rgba = context.getImageData(
    0,
    0,
    resolution.width,
    resolution.height,
  ).data;
  const spatial = resolution.width * resolution.height;
  const chw = new Float32Array(3 * spatial);
  for (let i = 0; i < spatial; i += 1) {
    chw[i] = rgba[i * 4] / 255;
    chw[spatial + i] = rgba[i * 4 + 1] / 255;
    chw[spatial * 2 + i] = rgba[i * 4 + 2] / 255;
  }
  return Array.from(chw);
};

const tensorValuesToDataUrl = (
  values: number[],
  resolution: ImageResolution,
): string => {
  const canvas = document.createElement("canvas");
  canvas.width = resolution.width;
  canvas.height = resolution.height;
  const context = canvas.getContext("2d");
  if (context === null) throw new Error("2D canvas unavailable.");
  const spatial = resolution.width * resolution.height;
  const imageData = context.createImageData(
    resolution.width,
    resolution.height,
  );
  for (let i = 0; i < spatial; i += 1) {
    imageData.data[i * 4] = Math.max(
      0,
      Math.min(255, Math.round(values[i] * 255)),
    );
    imageData.data[i * 4 + 1] = Math.max(
      0,
      Math.min(255, Math.round(values[spatial + i] * 255)),
    );
    imageData.data[i * 4 + 2] = Math.max(
      0,
      Math.min(255, Math.round(values[spatial * 2 + i] * 255)),
    );
    imageData.data[i * 4 + 3] = 255;
  }
  context.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
};

const getInitialPackPreference = (): VggPackName => {
  const storedPack = localStorage.getItem(VGG_PACK_STORAGE_KEY);
  if (
    storedPack === "fp32" ||
    storedPack === "fp16" ||
    storedPack === "int8-per-channel" ||
    storedPack === "int8log-per-channel" ||
    storedPack === "int4-experimental" ||
    storedPack === "int4log-experimental"
  ) {
    return storedPack;
  }
  return DEFAULT_VGG_PACK;
};

export const useStyleTransferController =
  (): UseStyleTransferControllerResult => {
    const [workerStatus, setWorkerStatus] =
      useState<string>("Worker starting…");
    const [gpuStatus, setGpuStatus] = useState<string>(
      "Checking WebGPU support…",
    );
    const [gpuInfo, setGpuInfo] = useState<string>("GPU details pending…");
    const [contentResolution, setContentResolution] =
      useState<ResolutionPreset>("128x192");
    const [styleResolution, setStyleResolution] =
      useState<ResolutionPreset>("128x160");
    const [stepsPerChunk, setStepsPerChunk] = useState<number>(10);
    const [contentWeight, setContentWeight] = useState<number>(1);
    const [styleWeight, setStyleWeight] = useState<number>(500000);
    const [learningRate, setLearningRate] = useState<number>(1);
    const [selectedPack, setSelectedPackState] = useState<VggPackName>(
      getInitialPackPreference,
    );
    const [optimizer, setOptimizer] = useState<"sgd" | "adam" | "lbfgs">(
      "lbfgs",
    );
    const [adamBeta1, setAdamBeta1] = useState<number>(0.9);
    const [adamBeta2, setAdamBeta2] = useState<number>(0.999);
    const [adamEpsilon, setAdamEpsilon] = useState<number>(1e-8);
    const [lbfgsMemory, setLbfgsMemory] = useState<number>(10);
    const [lbfgsEpsilon, setLbfgsEpsilon] = useState<number>(1e-9);
    const [synchronizePhaseTimings, setSynchronizePhaseTimings] =
      useState<boolean>(false);
    const [useCachedPipelines, setUseCachedPipelines] = useState<boolean>(true);
    const [usePersistentWeightBuffers, setUsePersistentWeightBuffers] =
      useState<boolean>(true);
    const [useStepBufferPool, setUseStepBufferPool] = useState<boolean>(true);
    const [useVec4Pointwise, setUseVec4Pointwise] = useState<boolean>(true);
    const [usePoolBackwardScatter, setUsePoolBackwardScatter] =
      useState<boolean>(true);
    const [gramKernel, setGramKernel] = useState<KernelGramKernel>(
      "symmetric-parallel-dot",
    );
    const [styleBackward, setStyleBackward] = useState<KernelStyleBackward>(
      "fused-from-gram-diff",
    );
    const [convForwardKernel, setConvForwardKernel] =
      useState<KernelConvForward>("scalar");
    const [convBackwardInputKernel, setConvBackwardInputKernel] =
      useState<KernelConvBackwardInput>("scalar");
    const [weightStorage, setWeightStorage] =
      useState<KernelWeightStorage>("fp32");
    const [iterations, setIterations] = useState<number>(0);
    const [isRunning, setIsRunning] = useState<boolean>(false);
    const [lastLoss, setLastLoss] = useState<number | null>(null);
    const [contentImage, setContentImage] = useState<string | null>(null);
    const [styleImage, setStyleImage] = useState<string | null>(null);
    const [outputImage, setOutputImage] = useState<string | null>(null);
    const [contentSourceImage, setContentSourceImage] =
      useState<HTMLImageElement | null>(null);
    const [styleSourceImage, setStyleSourceImage] =
      useState<HTMLImageElement | null>(null);
    const [weights, setWeights] = useState<FullWeights | null>(null);
    const [contentTensor, setContentTensor] = useState<number[] | null>(null);
    const [styleTensor, setStyleTensor] = useState<number[] | null>(null);
    const [inputTensor, setInputTensor] = useState<number[] | null>(null);
    const [optimizationSessionId, setOptimizationSessionId] = useState<string>(
      createOptimizationSessionId,
    );
    const [runStats, setRunStats] = useState<
      import("../../../types").WorkerRunStats | null
    >(null);
    const [modelCacheState, setModelCacheState] = useState<
      "downloading" | "cached" | "ready"
    >("downloading");
    const [modelCacheBytes, setModelCacheBytes] = useState<number>(0);
    const [modelCachePackStatuses, setModelCachePackStatuses] = useState<
      ModelCachePackStatus[]
    >([]);

    const workerRef = useRef<Worker | null>(null);
    const contentImageResolution = RESOLUTION_PRESETS[contentResolution];
    const styleImageResolution = RESOLUTION_PRESETS[styleResolution];
    const hasWebGpuOnMainThread = useMemo<boolean>(
      () => "gpu" in navigator,
      [],
    );
    const canRun =
      contentTensor !== null && styleTensor !== null && weights !== null;
    const kernelFlags = useMemo(() => {
      const flags: import("../../../types").WorkerKernelOptimizationFlags = {};
      if (useCachedPipelines) flags.useCachedPipelines = true;
      if (usePersistentWeightBuffers) flags.usePersistentWeightBuffers = true;
      if (useStepBufferPool) flags.useStepBufferPool = true;
      if (useVec4Pointwise) flags.useVec4Pointwise = true;
      if (usePoolBackwardScatter) flags.usePoolBackwardScatter = true;
      if (gramKernel !== "scalar") flags.gramKernel = gramKernel;
      if (styleBackward !== "two-pass") flags.styleBackward = styleBackward;
      if (convForwardKernel !== "scalar")
        flags.convForwardKernel = convForwardKernel;
      if (convBackwardInputKernel !== "scalar") {
        flags.convBackwardInputKernel = convBackwardInputKernel;
      }
      if (weightStorage !== "fp32") flags.weightStorage = weightStorage;
      return Object.keys(flags).length > 0 ? flags : undefined;
    }, [
      convBackwardInputKernel,
      convForwardKernel,
      gramKernel,
      styleBackward,
      useCachedPipelines,
      usePersistentWeightBuffers,
      usePoolBackwardScatter,
      useStepBufferPool,
      useVec4Pointwise,
      weightStorage,
    ]);
    const kernelConfigSummary = useMemo(() => {
      const enabled: string[] = [];
      if (useCachedPipelines) enabled.push("cached-pipelines");
      if (usePersistentWeightBuffers) enabled.push("persistent-weights");
      if (useStepBufferPool) enabled.push("step-pool");
      if (usePoolBackwardScatter) enabled.push("pool-scatter");
      if (useVec4Pointwise) enabled.push("vec4-pointwise");
      if (gramKernel !== "scalar") enabled.push(`gram=${gramKernel}`);
      if (styleBackward !== "two-pass")
        enabled.push(`styleBackward=${styleBackward}`);
      if (convForwardKernel !== "scalar")
        enabled.push(`convFwd=${convForwardKernel}`);
      if (convBackwardInputKernel !== "scalar")
        enabled.push(`convBwdIn=${convBackwardInputKernel}`);
      if (weightStorage !== "fp32")
        enabled.push(`weightStorage=${weightStorage}`);
      return enabled.length === 0 ? "baseline" : enabled.join(", ");
    }, [
      convBackwardInputKernel,
      convForwardKernel,
      gramKernel,
      styleBackward,
      useCachedPipelines,
      usePersistentWeightBuffers,
      usePoolBackwardScatter,
      useStepBufferPool,
      useVec4Pointwise,
      weightStorage,
    ]);

    const clearWorkerSession = (sessionId: string): void => {
      if (workerRef.current === null) return;
      workerRef.current.postMessage({
        type: "clear-style-transfer-session",
        id: createMessageId(),
        sessionId,
      } satisfies WorkerRequest);
    };

    const rotateOptimizationSession = (clearCurrent: boolean): void => {
      if (clearCurrent) clearWorkerSession(optimizationSessionId);
      setOptimizationSessionId(createOptimizationSessionId());
    };

    const resetOptimizerProgress = (): void => {
      setIterations(0);
      setLastLoss(null);
      setRunStats(null);
      setIsRunning(false);
    };

    useEffect(() => {
      let isCurrent = true;
      const loadWeights = async (): Promise<void> => {
        setWorkerStatus(`Loading VGG19 weights pack: ${selectedPack}...`);
        setModelCacheState("downloading");
        try {
          const nextWeights =
            await loadVgg19ManifestWeightsForPack(selectedPack);
          if (!isCurrent) return;
          setWeights(nextWeights.weights);
          setModelCacheState(nextWeights.state);
          setModelCacheBytes(nextWeights.cacheStatus.bytes);
          setModelCachePackStatuses(nextWeights.cacheStatus.packStatuses);
          setWorkerStatus(`Loaded VGG19 weights pack: ${selectedPack}.`);
        } catch (error) {
          if (!isCurrent) return;
          const message =
            error instanceof Error ? error.message : String(error);
          setWorkerStatus(
            `Failed to load VGG19 weights pack '${selectedPack}': ${message}`,
          );
        }
      };
      void loadWeights();
      return () => {
        isCurrent = false;
      };
    }, [selectedPack]);

    useEffect(() => {
      void (async (): Promise<void> => {
        const status = await getCachedVggModelPackStatus();
        setModelCacheBytes(status.bytes);
        setModelCachePackStatuses(status.packStatuses);
      })();
    }, []);

    useEffect(() => {
      const worker = new Worker(
        new URL("../../../styleTransfer.worker.ts", import.meta.url),
        {
          type: "module",
        },
      );
      workerRef.current = worker;
      const onMessage = (event: MessageEvent<WorkerResponse>): void => {
        const payload = event.data;
        if (payload.type === "pong")
          setWorkerStatus(
            `Worker ping OK @ ${new Date(payload.timestamp).toISOString()}`,
          );
        if (payload.type === "webgpu-init-result")
          setGpuStatus(
            payload.ok ? payload.message : `Fallback: ${payload.message}`,
          );
        if (payload.type === "error") setWorkerStatus(payload.message);
        if (
          payload.type === "clear-style-transfer-session-result" &&
          !payload.ok
        )
          setWorkerStatus(payload.message);
        if (payload.type === "run-style-transfer-result") {
          if (!payload.ok) {
            setWorkerStatus(payload.message);
            setIsRunning(false);
            return;
          }
          const nextValues = payload.finalValues;
          setInputTensor(nextValues);
          setOutputImage(
            tensorValuesToDataUrl(nextValues, contentImageResolution),
          );
          setIterations((value) => value + payload.stats.steps);
          if (payload.losses.length > 0)
            setLastLoss(payload.losses[payload.losses.length - 1]);
          setRunStats(payload.stats);
        }
      };
      worker.addEventListener("message", onMessage);
      worker.postMessage({
        type: "ping",
        id: createMessageId(),
      } satisfies WorkerRequest);
      worker.postMessage({
        type: "init-webgpu",
        id: createMessageId(),
      } satisfies WorkerRequest);
      return () => {
        worker.removeEventListener("message", onMessage);
        worker.terminate();
        workerRef.current = null;
      };
    }, [contentImageResolution]);

    useEffect(() => {
      void (async (): Promise<void> => {
        if (!hasWebGpuOnMainThread) {
          setGpuInfo("No WebGPU adapter on main thread.");
          return;
        }
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter === null) {
          setGpuInfo("Adapter request failed.");
          return;
        }
        const maybeAdapter = adapter as GPUAdapter & {
          requestAdapterInfo?: () => Promise<{
            vendor?: string;
            architecture?: string;
            device?: string;
            description?: string;
          }>;
        };
        if (maybeAdapter.requestAdapterInfo === undefined) {
          setGpuInfo("Adapter info API unavailable in this browser build.");
          return;
        }
        const info = await maybeAdapter.requestAdapterInfo();
        setGpuInfo(
          `Vendor: ${info.vendor ?? "unknown"} | Architecture: ${info.architecture ?? "unknown"} | Device: ${info.device ?? "unknown"} | Description: ${info.description ?? "unknown"}`,
        );
      })();
    }, [hasWebGpuOnMainThread]);

    useEffect(() => {
      if (
        !isRunning ||
        workerRef.current === null ||
        weights === null ||
        contentTensor === null ||
        styleTensor === null ||
        inputTensor === null
      )
        return;
      assertValidVgg19ReluTapIndices(
        VGG19_RELU_TAP_STYLE_LAYER_INDICES,
        VGG19_RELU_TAP_CONTENT_LAYER_INDEX,
      );
      workerRef.current.postMessage({
        type: "run-style-transfer",
        id: createMessageId(),
        sessionId: optimizationSessionId,
        optimizer,
        adamBeta1: optimizer === "adam" ? adamBeta1 : undefined,
        adamBeta2: optimizer === "adam" ? adamBeta2 : undefined,
        adamEpsilon: optimizer === "adam" ? adamEpsilon : undefined,
        lbfgsMemory:
          optimizer === "lbfgs"
            ? Math.max(1, Math.floor(lbfgsMemory))
            : undefined,
        lbfgsEpsilon: optimizer === "lbfgs" ? lbfgsEpsilon : undefined,
        inputShape: shapeForResolution(contentImageResolution),
        contentShape: shapeForResolution(contentImageResolution),
        styleShape: shapeForResolution(styleImageResolution),
        inputImageValues: inputTensor,
        contentImageValues: contentTensor,
        styleImageValues: styleTensor,
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
        styleLayerIndices: Array.from(VGG19_RELU_TAP_STYLE_LAYER_INDICES),
        contentLayerIndex: VGG19_RELU_TAP_CONTENT_LAYER_INDEX,
        weights,
        contentWeight,
        styleWeight,
        learningRate,
        steps: stepsPerChunk,
        lossReadbackInterval: stepsPerChunk,
        synchronizePhaseTimings,
        kernelFlags,
      } satisfies WorkerRequest);
    }, [
      adamBeta1,
      adamBeta2,
      adamEpsilon,
      contentTensor,
      contentWeight,
      inputTensor,
      isRunning,
      iterations,
      lbfgsEpsilon,
      lbfgsMemory,
      learningRate,
      optimizer,
      optimizationSessionId,
      contentImageResolution,
      styleImageResolution,
      stepsPerChunk,
      styleTensor,
      styleWeight,
      synchronizePhaseTimings,
      kernelFlags,
      weights,
    ]);

    const refreshContentImage = (
      image: HTMLImageElement,
      resolution: ImageResolution,
    ): void => {
      rotateOptimizationSession(true);
      const tensor = imageToTensorValues(image, resolution);
      setContentTensor(tensor);
      setInputTensor(tensor);
      setContentImage(tensorValuesToDataUrl(tensor, resolution));
      setOutputImage(tensorValuesToDataUrl(tensor, resolution));
      resetOptimizerProgress();
    };

    const refreshStyleImage = (
      image: HTMLImageElement,
      resolution: ImageResolution,
    ): void => {
      rotateOptimizationSession(true);
      const tensor = imageToTensorValues(image, resolution);
      setStyleTensor(tensor);
      setStyleImage(tensorValuesToDataUrl(tensor, resolution));
      if (contentTensor !== null) {
        setInputTensor(contentTensor);
        setOutputImage(
          tensorValuesToDataUrl(contentTensor, contentImageResolution),
        );
      }
      resetOptimizerProgress();
    };

    useEffect(() => {
      let isCurrent = true;
      void (async (): Promise<void> => {
        const [defaultContentImage, defaultStyleImage] = await Promise.all([
          readUrlAsImage(DEFAULT_CONTENT_IMAGE_URL),
          readUrlAsImage(DEFAULT_STYLE_IMAGE_URL),
        ]);
        if (!isCurrent) return;
        const nextContentResolution =
          nearestResolutionPresetForImage(defaultContentImage);
        const nextStyleResolution =
          nearestResolutionPresetForImage(defaultStyleImage);
        const nextContentImageResolution =
          RESOLUTION_PRESETS[nextContentResolution];
        const nextStyleImageResolution =
          RESOLUTION_PRESETS[nextStyleResolution];
        const defaultContentTensor = imageToTensorValues(
          defaultContentImage,
          nextContentImageResolution,
        );
        const defaultStyleTensor = imageToTensorValues(
          defaultStyleImage,
          nextStyleImageResolution,
        );
        setContentSourceImage(defaultContentImage);
        setStyleSourceImage(defaultStyleImage);
        setContentResolution(nextContentResolution);
        setStyleResolution(nextStyleResolution);
        setContentTensor(defaultContentTensor);
        setStyleTensor(defaultStyleTensor);
        setInputTensor(defaultContentTensor);
        setContentImage(
          tensorValuesToDataUrl(
            defaultContentTensor,
            nextContentImageResolution,
          ),
        );
        setStyleImage(
          tensorValuesToDataUrl(defaultStyleTensor, nextStyleImageResolution),
        );
        setOutputImage(
          tensorValuesToDataUrl(
            defaultContentTensor,
            nextContentImageResolution,
          ),
        );
        resetOptimizerProgress();
      })();
      return () => {
        isCurrent = false;
      };
    }, []);

    const updateContentResolution: Dispatch<
      SetStateAction<ResolutionPreset>
    > = (update) => {
      const nextPreset = resolvePresetUpdate(update, contentResolution);
      setContentResolution(nextPreset);
      if (contentSourceImage !== null)
        refreshContentImage(contentSourceImage, RESOLUTION_PRESETS[nextPreset]);
    };

    const updateStyleResolution: Dispatch<SetStateAction<ResolutionPreset>> = (
      update,
    ) => {
      const nextPreset = resolvePresetUpdate(update, styleResolution);
      setStyleResolution(nextPreset);
      if (styleSourceImage !== null)
        refreshStyleImage(styleSourceImage, RESOLUTION_PRESETS[nextPreset]);
    };

    const onUpload = async (
      event: ChangeEvent<HTMLInputElement>,
      target: "content" | "style",
    ): Promise<void> => {
      const file = event.target.files?.[0];
      if (file === undefined) return;
      const image = await readFileAsImage(file);
      if (target === "content") {
        const nextPreset = nearestResolutionPresetForImage(image);
        setContentSourceImage(image);
        setContentResolution(nextPreset);
        refreshContentImage(image, RESOLUTION_PRESETS[nextPreset]);
      } else {
        const nextPreset = nearestResolutionPresetForImage(image);
        setStyleSourceImage(image);
        setStyleResolution(nextPreset);
        refreshStyleImage(image, RESOLUTION_PRESETS[nextPreset]);
      }
    };

    const updateOptimizerConfigWhenPaused = <T>(
      update: SetStateAction<T>,
      current: T,
      setter: Dispatch<SetStateAction<T>>,
    ): void => {
      const next = toResolvedState(update, current);
      if (Object.is(next, current)) return;
      if (!isRunning) rotateOptimizationSession(true);
      setter(next);
    };

    const resetOptimizerState = (): void => {
      rotateOptimizationSession(true);
      resetOptimizerProgress();
    };

    const resetOutputImage = (): void => {
      if (contentTensor === null) return;
      rotateOptimizationSession(true);
      setInputTensor(contentTensor);
      setOutputImage(
        tensorValuesToDataUrl(contentTensor, contentImageResolution),
      );
      resetOptimizerProgress();
    };

    const setSelectedPack: Dispatch<SetStateAction<VggPackName>> = (update) => {
      const nextPack = toResolvedState(update, selectedPack);
      if (nextPack === selectedPack) return;
      localStorage.setItem(VGG_PACK_STORAGE_KEY, nextPack);
      rotateOptimizationSession(true);
      resetOptimizerProgress();
      setSelectedPackState(nextPack);
    };

    const clearModelCache = async (): Promise<void> => {
      const nextStatus = await clearCachedVggModelPacks();
      setModelCacheBytes(nextStatus.bytes);
      setModelCachePackStatuses(nextStatus.packStatuses);
      setModelCacheState("ready");
      setWorkerStatus("Model cache cleared.");
    };

    return {
      controls: {
        contentResolution,
        setContentResolution: updateContentResolution,
        styleResolution,
        setStyleResolution: updateStyleResolution,
        stepsPerChunk,
        setStepsPerChunk,
        contentWeight,
        setContentWeight,
        styleWeight,
        setStyleWeight,
        learningRate,
        setLearningRate: (update) =>
          updateOptimizerConfigWhenPaused(
            update,
            learningRate,
            setLearningRate,
          ),
        selectedPack,
        setSelectedPack,
        optimizer,
        setOptimizer: (update) =>
          updateOptimizerConfigWhenPaused(update, optimizer, setOptimizer),
        adamBeta1,
        setAdamBeta1: (update) =>
          updateOptimizerConfigWhenPaused(update, adamBeta1, setAdamBeta1),
        adamBeta2,
        setAdamBeta2: (update) =>
          updateOptimizerConfigWhenPaused(update, adamBeta2, setAdamBeta2),
        adamEpsilon,
        setAdamEpsilon: (update) =>
          updateOptimizerConfigWhenPaused(update, adamEpsilon, setAdamEpsilon),
        lbfgsMemory,
        setLbfgsMemory: (update) =>
          updateOptimizerConfigWhenPaused(update, lbfgsMemory, setLbfgsMemory),
        lbfgsEpsilon,
        setLbfgsEpsilon: (update) =>
          updateOptimizerConfigWhenPaused(
            update,
            lbfgsEpsilon,
            setLbfgsEpsilon,
          ),
        synchronizePhaseTimings,
        setSynchronizePhaseTimings: (update) =>
          updateOptimizerConfigWhenPaused(
            update,
            synchronizePhaseTimings,
            setSynchronizePhaseTimings,
          ),
        useCachedPipelines,
        setUseCachedPipelines: (update) =>
          updateOptimizerConfigWhenPaused(
            update,
            useCachedPipelines,
            setUseCachedPipelines,
          ),
        usePersistentWeightBuffers,
        setUsePersistentWeightBuffers: (update) =>
          updateOptimizerConfigWhenPaused(
            update,
            usePersistentWeightBuffers,
            setUsePersistentWeightBuffers,
          ),
        useStepBufferPool,
        setUseStepBufferPool: (update) =>
          updateOptimizerConfigWhenPaused(
            update,
            useStepBufferPool,
            setUseStepBufferPool,
          ),
        useVec4Pointwise,
        setUseVec4Pointwise: (update) =>
          updateOptimizerConfigWhenPaused(
            update,
            useVec4Pointwise,
            setUseVec4Pointwise,
          ),
        usePoolBackwardScatter,
        setUsePoolBackwardScatter: (update) =>
          updateOptimizerConfigWhenPaused(
            update,
            usePoolBackwardScatter,
            setUsePoolBackwardScatter,
          ),
        gramKernel,
        setGramKernel: (update) =>
          updateOptimizerConfigWhenPaused(update, gramKernel, setGramKernel),
        styleBackward,
        setStyleBackward: (update) =>
          updateOptimizerConfigWhenPaused(
            update,
            styleBackward,
            setStyleBackward,
          ),
        convForwardKernel,
        setConvForwardKernel: (update) =>
          updateOptimizerConfigWhenPaused(
            update,
            convForwardKernel,
            setConvForwardKernel,
          ),
        convBackwardInputKernel,
        setConvBackwardInputKernel: (update) =>
          updateOptimizerConfigWhenPaused(
            update,
            convBackwardInputKernel,
            setConvBackwardInputKernel,
          ),
        weightStorage,
        setWeightStorage: (update) =>
          updateOptimizerConfigWhenPaused(
            update,
            weightStorage,
            setWeightStorage,
          ),
        kernelConfigSummary,
        kernelFlags,
        resetOptimizerState,
        resetOutputImage,
        clearModelCache,
      },
      status: {
        workerStatus,
        gpuStatus,
        gpuInfo,
        iterations,
        isRunning,
        lastLoss,
        runStats,
        modelCacheState,
        modelCacheBytes,
        modelCachePackStatuses,
      },
      images: { contentImage, styleImage, outputImage },
      canRun,
      setIsRunning,
      onUpload,
    };
  };
