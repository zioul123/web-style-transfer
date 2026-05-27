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
  ResolutionPreset,
  UseStyleTransferControllerResult,
} from "../types/controller";
import {
  parseVgg19ManifestBackedLayerCache,
  type Vgg19WeightsManifest,
} from "../../../ml/worker/models/vgg19/weights";

const RESOLUTION_PRESETS: Record<ResolutionPreset, ImageResolution> = {
  "128x128": { width: 128, height: 128 },
  "128x192": { width: 192, height: 128 },
  "192x128": { width: 128, height: 192 },
  "256x256": { width: 256, height: 256 },
  "256x384": { width: 384, height: 256 },
};

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

const createMessageId = (): string =>
  `${Date.now()}-${Math.random().toString(16).slice(2)}`;

const createOptimizationSessionId = (): string =>
  `style-transfer-session-${createMessageId()}`;

const toResolvedState = <T,>(
  update: SetStateAction<T>,
  current: T,
): T =>
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

export const useStyleTransferController =
  (): UseStyleTransferControllerResult => {
    const [workerStatus, setWorkerStatus] =
      useState<string>("Worker starting…");
    const [gpuStatus, setGpuStatus] = useState<string>(
      "Checking WebGPU support…",
    );
    const [gpuInfo, setGpuInfo] = useState<string>("GPU details pending…");
    const [contentResolution, setContentResolution] =
      useState<ResolutionPreset>("128x128");
    const [styleResolution, setStyleResolution] =
      useState<ResolutionPreset>("128x192");
    const [stepsPerChunk, setStepsPerChunk] = useState<number>(10);
    const [contentWeight, setContentWeight] = useState<number>(1);
    const [styleWeight, setStyleWeight] = useState<number>(500000);
    const [learningRate, setLearningRate] = useState<number>(1);
    const [optimizer, setOptimizer] =
      useState<"sgd" | "adam" | "lbfgs">("lbfgs");
    const [adamBeta1, setAdamBeta1] = useState<number>(0.9);
    const [adamBeta2, setAdamBeta2] = useState<number>(0.999);
    const [adamEpsilon, setAdamEpsilon] = useState<number>(1e-8);
    const [lbfgsMemory, setLbfgsMemory] = useState<number>(100);
    const [lbfgsEpsilon, setLbfgsEpsilon] = useState<number>(1e-9);
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
    const [optimizationSessionId, setOptimizationSessionId] =
      useState<string>(createOptimizationSessionId);
    const [runStats, setRunStats] = useState<
      import("../../../types").WorkerRunStats | null
    >(null);

    const workerRef = useRef<Worker | null>(null);
    const contentImageResolution = RESOLUTION_PRESETS[contentResolution];
    const styleImageResolution = RESOLUTION_PRESETS[styleResolution];
    const hasWebGpuOnMainThread = useMemo<boolean>(
      () => "gpu" in navigator,
      [],
    );
    const canRun =
      contentTensor !== null && styleTensor !== null && weights !== null;

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

    useEffect(() => {
      const loadWeights = async (): Promise<void> => {
        try {
          const manifestResponse = await fetch("/vgg19-models/fp32/manifest.json");
          if (!manifestResponse.ok) throw new Error("manifest fetch failed");
          const manifest = (await manifestResponse.json()) as Vgg19WeightsManifest;
          const shardEntries = await Promise.all(
            manifest.shards.map(async (shard) => {
              const response = await fetch(`/vgg19-models/fp32/${shard.name}`);
              if (!response.ok) throw new Error(`shard fetch failed: ${shard.name}`);
              return [shard.name, await response.arrayBuffer()] as const;
            }),
          );
          const shardBuffers: Record<string, ArrayBuffer> = Object.fromEntries(shardEntries);
          const cache = await parseVgg19ManifestBackedLayerCache(manifest, shardBuffers);
          const legacyWeights: FullWeights = {};
          for (let layerIndex = 0; layerIndex <= 29; layerIndex += 1) {
            const layer = cache[layerIndex];
            if (layer === undefined) continue;
            legacyWeights[`conv${layerIndex}.weightShape`] = [...layer.shape];
            legacyWeights[`conv${layerIndex}.weightValues`] = Array.from(layer.values);
            legacyWeights[`conv${layerIndex}.biasValues`] = Array.from(layer.bias);
          }
          setWeights(legacyWeights);
          return;
        } catch (_error) {
          const legacyResponse = await fetch("/vgg19-phase3-full-pass/vgg19_conv0_to_conv28_weights.json");
          if (!legacyResponse.ok) {
            setWorkerStatus("Failed to load VGG19 weights artifacts (manifest and legacy). ");
            return;
          }
          const legacyJson = (await legacyResponse.json()) as FullWeights;
          setWeights(legacyJson);
        }
      };
      void loadWeights();
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
        if (payload.type === "clear-style-transfer-session-result" && !payload.ok)
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
      setIterations(0);
      setLastLoss(null);
      setRunStats(null);
      setIsRunning(false);
    };

    const refreshStyleImage = (
      image: HTMLImageElement,
      resolution: ImageResolution,
    ): void => {
      rotateOptimizationSession(true);
      const tensor = imageToTensorValues(image, resolution);
      setStyleTensor(tensor);
      setStyleImage(tensorValuesToDataUrl(tensor, resolution));
      setIterations(0);
      setLastLoss(null);
      setRunStats(null);
      setIsRunning(false);
    };

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
        setContentSourceImage(image);
        refreshContentImage(image, contentImageResolution);
      } else {
        setStyleSourceImage(image);
        refreshStyleImage(image, styleImageResolution);
      }
    };

    const updateOptimizerConfigWhenPaused = <T,>(
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
      setIterations(0);
      setLastLoss(null);
      setRunStats(null);
      setIsRunning(false);
    };

    const resetOutputImage = (): void => {
      if (contentTensor === null) return;
      rotateOptimizationSession(true);
      setInputTensor(contentTensor);
      setOutputImage(tensorValuesToDataUrl(contentTensor, contentImageResolution));
      setIterations(0);
      setLastLoss(null);
      setRunStats(null);
      setIsRunning(false);
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
          updateOptimizerConfigWhenPaused(
            update,
            lbfgsMemory,
            setLbfgsMemory,
          ),
        lbfgsEpsilon,
        setLbfgsEpsilon: (update) =>
          updateOptimizerConfigWhenPaused(
            update,
            lbfgsEpsilon,
            setLbfgsEpsilon,
          ),
        resetOptimizerState,
        resetOutputImage,
      },
      status: {
        workerStatus,
        gpuStatus,
        gpuInfo,
        iterations,
        isRunning,
        lastLoss,
        runStats,
      },
      images: { contentImage, styleImage, outputImage },
      canRun,
      setIsRunning,
      onUpload,
    };
  };
