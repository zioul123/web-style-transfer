import { useState, type ReactElement } from "react";
import {
  buildPackAcceptanceRows,
  type PackComparisonRow,
} from "./features/style-transfer/benchmark/packAcceptance";
import { assetUrl, vgg19ModelUrl } from "./shared/assetUrls";
import {
  VGG_PACK_OPTIONS,
  type VggPackName,
} from "./features/style-transfer/modelPacks";
import {
  parseVgg19ManifestBackedLayerCache,
  type Vgg19WeightsManifest,
} from "./ml/worker/models/vgg19/weights";
import type {
  WorkerFirstPoolBenchmarkStats,
  WorkerKernelOptimizationFlags,
  WorkerKernelStats,
  WorkerRequest,
  WorkerResponse,
  WorkerRunStats,
} from "./types";

type VggFirstPoolWeightsFixture = {
  conv1WeightShape: [number, number, number, number];
  conv1WeightValues: number[];
  conv1BiasValues: number[];
  conv2WeightShape: [number, number, number, number];
  conv2WeightValues: number[];
  conv2BiasValues: number[];
  conv3WeightShape: [number, number, number, number];
  conv3WeightValues: number[];
  conv3BiasValues: number[];
};

type VggFirstPoolCaseFixture = {
  inputShape: [number, number, number, number];
  inputValues: number[];
  mean: [number, number, number];
  std: [number, number, number];
};

type Phase3FullPassFixture = {
  inputShape: [number, number, number, number];
  inputImageValues: number[];
  contentImageValues: number[];
  styleImageValues: number[];
  mean: [number, number, number];
  std: [number, number, number];
  styleLayerIndices: number[];
  contentLayerIndex: number;
};

type BenchmarkTab = "first-pool" | "full-style" | "kernel-lab";

type BenchmarkStats = WorkerFirstPoolBenchmarkStats | WorkerRunStats;

type BenchmarkResult = {
  losses: number[];
  stats?: BenchmarkStats;
  pack?: VggPackName;
};
type KernelLabRow = {
  name: string;
  elapsedMs: number;
  forwardMs: number;
  lossMs: number;
  backwardMs: number;
  updateMs: number;
  finalLoss: number;
  lossDeltaFromBaseline: number;
  speedupVsBaseline: number;
  ok: boolean;
  error?: string;
  kernelStats?: WorkerKernelStats;
};

type MetricSummary = { mean: number; p50: number; p95: number };

type BenchmarkRunSummary = {
  runs: number;
  elapsedMs: MetricSummary;
  forwardMs: MetricSummary;
  lossMs: MetricSummary;
  backwardMs: MetricSummary;
  updateMs: MetricSummary;
  readbackMs?: MetricSummary;
  mandatoryReadbackMs?: MetricSummary;
  diagnosticsReadbackMs?: MetricSummary;
  downloadSizeBytes?: number;
  decodeLoadMs?: number;
  firstIterationMs?: number;
  peakMemoryBytes?: number;
};
type WeightLoadStats = { downloadSizeBytes: number; decodeLoadMs: number };

const kernelLabBaselineName = (pack: VggPackName): string =>
  `baseline (default pooled ${pack})`;

const percentile = (sortedValues: readonly number[], p: number): number => {
  if (sortedValues.length === 0) return 0;
  const idx = Math.min(
    sortedValues.length - 1,
    Math.max(0, Math.ceil((p / 100) * sortedValues.length) - 1),
  );
  return sortedValues[idx];
};

const summarize = (values: readonly number[]): MetricSummary => {
  if (values.length === 0) return { mean: 0, p50: 0, p95: 0 };
  const sorted = [...values].sort((a, b) => a - b);
  const mean = values.reduce((acc, value) => acc + value, 0) / values.length;
  return { mean, p50: percentile(sorted, 50), p95: percentile(sorted, 95) };
};

const loadJson = async <T,>(url: string): Promise<T> => {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(`Failed to load ${url}: ${response.status}`);
  return (await response.json()) as T;
};

const loadManifestWeights = async (
  pack: VggPackName,
): Promise<{
  weights: Record<string, number[] | [number, number, number, number]>;
  stats: WeightLoadStats;
}> => {
  const decodeStart = performance.now();
  const manifest = await loadJson<Vgg19WeightsManifest>(
    vgg19ModelUrl(`${pack}/manifest.json`),
  );
  const shardEntries = await Promise.all(
    manifest.shards.map(async (shard) => {
      const response = await fetch(vgg19ModelUrl(`${pack}/${shard.name}`));
      if (!response.ok) throw new Error(`Missing weight shard: ${shard.name}`);
      return [shard.name, await response.arrayBuffer()] as const;
    }),
  );
  const shardMap: Record<string, ArrayBuffer> =
    Object.fromEntries(shardEntries);
  const layerCache = await parseVgg19ManifestBackedLayerCache(
    manifest,
    shardMap,
  );
  const weights: Record<string, number[] | [number, number, number, number]> =
    {};
  for (let layerIndex = 0; layerIndex <= 29; layerIndex += 1) {
    const layer = layerCache[layerIndex];
    if (layer === undefined) continue;
    weights[`conv${layerIndex}.weightShape`] = [...layer.shape] as [
      number,
      number,
      number,
      number,
    ];
    weights[`conv${layerIndex}.weightValues`] = Array.from(layer.values);
    weights[`conv${layerIndex}.biasValues`] = Array.from(layer.bias);
  }
  const downloadSizeBytes = shardEntries.reduce(
    (acc, [, buffer]) => acc + buffer.byteLength,
    0,
  );
  return {
    weights,
    stats: {
      downloadSizeBytes,
      decodeLoadMs: performance.now() - decodeStart,
    },
  };
};

const loadFirstAvailableManifestWeights = async (
  packs: readonly VggPackName[],
): Promise<{
  pack: VggPackName;
  weights: Record<string, number[] | [number, number, number, number]>;
  stats: WeightLoadStats;
}> => {
  const errors: string[] = [];
  for (const pack of packs) {
    try {
      return { pack, ...(await loadManifestWeights(pack)) };
    } catch (error) {
      errors.push(
        `${pack}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }
  throw new Error(
    `No requested VGG19 model pack is available. ${errors.join("; ")}`,
  );
};

const askWorker = (
  worker: Worker,
  payload: WorkerRequest,
): Promise<WorkerResponse> =>
  new Promise((resolve) => {
    const handler = (event: MessageEvent<WorkerResponse>): void => {
      if (event.data.id !== payload.id) return;
      worker.removeEventListener("message", handler);
      resolve(event.data);
    };
    worker.addEventListener("message", handler);
    worker.postMessage(payload);
  });

const hasReadbackStats = (
  stats: BenchmarkStats,
): stats is WorkerFirstPoolBenchmarkStats => "readbackMs" in stats;

const summarizeRuns = (
  runStats: readonly BenchmarkStats[],
): BenchmarkRunSummary => {
  const firstPoolStats = runStats.filter(hasReadbackStats);
  return {
    runs: runStats.length,
    elapsedMs: summarize(runStats.map((entry) => entry.elapsedMs)),
    forwardMs: summarize(runStats.map((entry) => entry.forwardMs)),
    lossMs: summarize(runStats.map((entry) => entry.lossMs)),
    backwardMs: summarize(runStats.map((entry) => entry.backwardMs)),
    updateMs: summarize(runStats.map((entry) => entry.updateMs)),
    readbackMs:
      firstPoolStats.length > 0
        ? summarize(firstPoolStats.map((entry) => entry.readbackMs))
        : undefined,
    mandatoryReadbackMs:
      firstPoolStats.length > 0
        ? summarize(firstPoolStats.map((entry) => entry.mandatoryReadbackMs))
        : undefined,
    diagnosticsReadbackMs:
      firstPoolStats.length > 0
        ? summarize(firstPoolStats.map((entry) => entry.diagnosticsReadbackMs))
        : undefined,
  };
};

const formatMetric = (metric: MetricSummary | undefined): string =>
  metric === undefined
    ? "-"
    : `${metric.mean.toFixed(2)} / ${metric.p50.toFixed(2)} / ${metric.p95.toFixed(2)}`;

export const BenchmarkApp = (): ReactElement => {
  const [activeTab, setActiveTab] = useState<BenchmarkTab>("first-pool");
  const [status, setStatus] = useState<string>("Ready");
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [result, setResult] = useState<BenchmarkResult | null>(null);
  const [summary, setSummary] = useState<BenchmarkRunSummary | null>(null);
  const [packComparison, setPackComparison] = useState<
    PackComparisonRow[] | null
  >(null);
  const [packAcceptance, setPackAcceptance] = useState<Array<{
    pack: VggPackName;
    verdict: "pass" | "fail";
    reason: string;
  }> | null>(null);
  const [kernelLabRows, setKernelLabRows] = useState<KernelLabRow[] | null>(
    null,
  );

  const [firstPoolSteps, setFirstPoolSteps] = useState<number>(6);
  const [firstPoolRuns, setFirstPoolRuns] = useState<number>(5);
  const [firstPoolLearningRate, setFirstPoolLearningRate] =
    useState<number>(0.15);
  const [firstPoolContentWeight, setFirstPoolContentWeight] =
    useState<number>(1);
  const [styleWeightConv1, setStyleWeightConv1] = useState<number>(30);
  const [styleWeightConv3, setStyleWeightConv3] = useState<number>(120);
  const [useFirstPoolFusedConvRelu, setUseFirstPoolFusedConvRelu] =
    useState<boolean>(true);
  const [useFirstPoolFusedUpdateClamp, setUseFirstPoolFusedUpdateClamp] =
    useState<boolean>(true);
  const [debugValidateStepShapes, setDebugValidateStepShapes] =
    useState<boolean>(false);
  const [debugReadbackGrad, setDebugReadbackGrad] = useState<boolean>(false);

  const [fullSteps, setFullSteps] = useState<number>(1);
  const [fullRuns, setFullRuns] = useState<number>(3);
  const [fullLearningRate, setFullLearningRate] = useState<number>(1e-5);
  const [fullContentWeight, setFullContentWeight] = useState<number>(1);
  const [fullStyleWeight, setFullStyleWeight] = useState<number>(1);
  const [kernelLabSteps, setKernelLabSteps] = useState<number>(1);

  const clearResults = (): void => {
    setResult(null);
    setSummary(null);
    setPackComparison(null);
    setPackAcceptance(null);
    setKernelLabRows(null);
  };

  const runKernelLabBenchmark = async (worker: Worker): Promise<void> => {
    setStatus("Loading full style-transfer fixtures for kernel lab...");
    const fixture = await loadJson<Phase3FullPassFixture>(
      assetUrl("vgg19-phase3-full-pass/vgg19_phase3_full_pass_fixture.json"),
    );
    const baselineWeights = await loadFirstAvailableManifestWeights([
      "fp32",
      "int8-per-channel",
    ]);
    const weights = baselineWeights.weights;
    const baselineName = kernelLabBaselineName(baselineWeights.pack);
    await initializeWorker(worker);
    const variants: Array<{
      name: string;
      kernelFlags?: WorkerKernelOptimizationFlags;
    }> = [
      { name: baselineName },
      { name: "cached-pipelines", kernelFlags: { useCachedPipelines: true } },
      {
        name: "cached+persistent-weights",
        kernelFlags: {
          useCachedPipelines: true,
          usePersistentWeightBuffers: true,
        },
      },
      {
        name: "cached+persistent+fp16-storage",
        kernelFlags: {
          useCachedPipelines: true,
          usePersistentWeightBuffers: true,
          weightStorage: "fp16-storage",
        },
      },
      {
        name: "cached+persistent+pool-scatter",
        kernelFlags: {
          useCachedPipelines: true,
          usePersistentWeightBuffers: true,
          usePoolBackwardScatter: true,
        },
      },
      {
        name: "cached+persistent+pool-scatter+vec4",
        kernelFlags: {
          useCachedPipelines: true,
          usePersistentWeightBuffers: true,
          usePoolBackwardScatter: true,
          useVec4Pointwise: true,
        },
      },
      {
        name: "cached+persistent+pool-scatter+vec4+gram-parallel+style-two-pass",
        kernelFlags: {
          useCachedPipelines: true,
          usePersistentWeightBuffers: true,
          usePoolBackwardScatter: true,
          useVec4Pointwise: true,
          gramKernel: "parallel-dot",
          styleBackward: "two-pass",
        },
      },
      {
        name: "cached+persistent+pool-scatter+vec4+gram-symmetric+style-fused",
        kernelFlags: {
          useCachedPipelines: true,
          usePersistentWeightBuffers: true,
          usePoolBackwardScatter: true,
          useVec4Pointwise: true,
          gramKernel: "symmetric-parallel-dot",
          styleBackward: "fused-from-gram-diff",
        },
      },
      {
        name: "cached+persistent+pool-scatter+vec4+gram-symmetric+style-fused+conv-forward-batched-scalar",
        kernelFlags: {
          useCachedPipelines: true,
          usePersistentWeightBuffers: true,
          usePoolBackwardScatter: true,
          useVec4Pointwise: true,
          gramKernel: "symmetric-parallel-dot",
          styleBackward: "fused-from-gram-diff",
          convForwardKernel: "spatial-vec4",
        },
      },
      {
        name: "cached+persistent+pool-scatter+vec4+gram-symmetric+style-fused+conv-forward-batched-scalar+conv-backward-transposed",
        kernelFlags: {
          useCachedPipelines: true,
          usePersistentWeightBuffers: true,
          usePoolBackwardScatter: true,
          useVec4Pointwise: true,
          gramKernel: "symmetric-parallel-dot",
          styleBackward: "fused-from-gram-diff",
          convForwardKernel: "spatial-vec4",
          convBackwardInputKernel: "transposed-weight-spatial-vec4",
        },
      },
      {
        name: "cached+persistent+fp16-storage+pool-scatter+vec4+gram-symmetric+style-fused+conv-forward-batched-scalar+conv-backward-transposed",
        kernelFlags: {
          useCachedPipelines: true,
          usePersistentWeightBuffers: true,
          weightStorage: "fp16-storage",
          usePoolBackwardScatter: true,
          useVec4Pointwise: true,
          gramKernel: "symmetric-parallel-dot",
          styleBackward: "fused-from-gram-diff",
          convForwardKernel: "spatial-vec4",
          convBackwardInputKernel: "transposed-weight-spatial-vec4",
        },
      },
    ];
    const smokeMode =
      new URLSearchParams(window.location.search).get("kernelLabSmoke") === "1";
    const selectedVariants = smokeMode ? [variants[0], variants[3]] : variants;
    const rows: KernelLabRow[] = [];
    let baselineLoss = 0;
    let baselineElapsed = 0;
    for (const variant of selectedVariants) {
      const warmupResponse = await askWorker(worker, {
        type: "run-style-transfer",
        id: `benchmark-kernel-lab-warmup-${variant.name}`,
        optimizer: "sgd",
        inputShape: fixture.inputShape,
        inputImageValues: fixture.inputImageValues,
        contentImageValues: fixture.contentImageValues,
        styleImageValues: fixture.styleImageValues,
        mean: fixture.mean,
        std: fixture.std,
        styleLayerIndices: fixture.styleLayerIndices,
        contentLayerIndex: fixture.contentLayerIndex,
        weights,
        contentWeight: fullContentWeight,
        styleWeight: fullStyleWeight,
        learningRate: fullLearningRate,
        steps: 1,
        kernelFlags: variant.kernelFlags,
        synchronizePhaseTimings: true,
      });
      if (
        warmupResponse.type === "run-style-transfer-result" &&
        !warmupResponse.ok
      ) {
        rows.push({
          name: variant.name,
          elapsedMs: 0,
          forwardMs: 0,
          lossMs: 0,
          backwardMs: 0,
          updateMs: 0,
          finalLoss: 0,
          lossDeltaFromBaseline: 0,
          speedupVsBaseline: 0,
          ok: false,
          error: warmupResponse.message,
        });
        continue;
      }
      setStatus(`Running kernel lab variant: ${variant.name}...`);
      const response = await askWorker(worker, {
        type: "run-style-transfer",
        id: `benchmark-kernel-lab-${variant.name}`,
        optimizer: "sgd",
        inputShape: fixture.inputShape,
        inputImageValues: fixture.inputImageValues,
        contentImageValues: fixture.contentImageValues,
        styleImageValues: fixture.styleImageValues,
        mean: fixture.mean,
        std: fixture.std,
        styleLayerIndices: fixture.styleLayerIndices,
        contentLayerIndex: fixture.contentLayerIndex,
        weights,
        contentWeight: fullContentWeight,
        styleWeight: fullStyleWeight,
        learningRate: fullLearningRate,
        steps: kernelLabSteps,
        kernelFlags: variant.kernelFlags,
        collectKernelStats: true,
        synchronizePhaseTimings: true,
      });
      if (response.type !== "run-style-transfer-result") {
        rows.push({
          name: variant.name,
          elapsedMs: 0,
          forwardMs: 0,
          lossMs: 0,
          backwardMs: 0,
          updateMs: 0,
          finalLoss: 0,
          lossDeltaFromBaseline: 0,
          speedupVsBaseline: 0,
          ok: false,
          error: "Unexpected worker response type.",
        });
        continue;
      }
      if (!response.ok) {
        rows.push({
          name: variant.name,
          elapsedMs: 0,
          forwardMs: 0,
          lossMs: 0,
          backwardMs: 0,
          updateMs: 0,
          finalLoss: 0,
          lossDeltaFromBaseline: 0,
          speedupVsBaseline: 0,
          ok: false,
          error: response.message,
        });
        continue;
      }
      const finalLoss = response.losses.at(-1) ?? 0;
      if (variant.name === baselineName) {
        baselineLoss = finalLoss;
        baselineElapsed = response.stats.elapsedMs;
      }
      rows.push({
        name: variant.name,
        elapsedMs: response.stats.elapsedMs,
        forwardMs: response.stats.forwardMs,
        lossMs: response.stats.lossMs,
        backwardMs: response.stats.backwardMs,
        updateMs: response.stats.updateMs,
        finalLoss,
        lossDeltaFromBaseline: finalLoss - baselineLoss,
        speedupVsBaseline:
          baselineElapsed > 0 ? baselineElapsed / response.stats.elapsedMs : 1,
        ok: true,
        kernelStats: response.stats.kernelStats,
      });
    }
    setKernelLabRows(rows);
  };

  const initializeWorker = async (worker: Worker): Promise<void> => {
    setStatus("Initializing WebGPU...");
    const init = await askWorker(worker, {
      type: "init-webgpu",
      id: "benchmark-init",
    });
    if (init.type !== "webgpu-init-result" || !init.ok) {
      throw new Error(
        init.type === "webgpu-init-result"
          ? init.message
          : "Unexpected init response",
      );
    }
  };

  const runFirstPoolBenchmark = async (worker: Worker): Promise<void> => {
    setStatus("Loading first-pool fixtures...");
    const [weights, caseData] = await Promise.all([
      loadJson<VggFirstPoolWeightsFixture>(
        assetUrl("vgg19-first-pool/vgg19_first_pool_weights.json"),
      ),
      loadJson<VggFirstPoolCaseFixture>(
        assetUrl("vgg19-first-pool/vgg19_first_pool_case_madeira16.json"),
      ),
    ]);
    await initializeWorker(worker);
    const cappedRuns = Math.max(1, Math.floor(firstPoolRuns));
    const runStats: WorkerFirstPoolBenchmarkStats[] = [];
    let lastResult: BenchmarkResult | null = null;

    for (let runIndex = 0; runIndex < cappedRuns; runIndex += 1) {
      setStatus(
        `Running first-pool benchmark (${runIndex + 1}/${cappedRuns})...`,
      );
      const optimized = await askWorker(worker, {
        type: "run-first-pool-optimizer",
        id: `benchmark-first-pool-${runIndex}`,
        inputShape: caseData.inputShape,
        mean: caseData.mean,
        std: caseData.std,
        contentImageValues: caseData.inputValues,
        styleImageValues: [...caseData.inputValues].reverse(),
        initialInputValues: caseData.inputValues,
        conv1Weight: {
          shape: weights.conv1WeightShape,
          values: weights.conv1WeightValues,
        },
        conv1Bias: weights.conv1BiasValues,
        conv2Weight: {
          shape: weights.conv2WeightShape,
          values: weights.conv2WeightValues,
        },
        conv2Bias: weights.conv2BiasValues,
        conv3Weight: {
          shape: weights.conv3WeightShape,
          values: weights.conv3WeightValues,
        },
        conv3Bias: weights.conv3BiasValues,
        contentWeight: firstPoolContentWeight,
        styleWeightConv1,
        styleWeightConv3,
        learningRate: firstPoolLearningRate,
        steps: firstPoolSteps,
        useFusedConvRelu: useFirstPoolFusedConvRelu,
        useFusedUpdateClamp: useFirstPoolFusedUpdateClamp,
        debugValidateStepShapes,
        debugReadbackGrad,
        collectBenchmarkStats: true,
      });
      if (optimized.type !== "run-first-pool-optimizer-result") {
        throw new Error("Unexpected first-pool run response");
      }
      if (!optimized.ok) throw new Error(optimized.message);
      lastResult = { losses: optimized.losses, stats: optimized.stats };
      if (optimized.stats !== undefined) runStats.push(optimized.stats);
    }

    if (lastResult !== null) setResult(lastResult);
    if (runStats.length > 0) setSummary(summarizeRuns(runStats));
  };

  const runFullStyleBenchmark = async (worker: Worker): Promise<void> => {
    setStatus("Loading full style-transfer fixtures...");
    const fixture = await loadJson<Phase3FullPassFixture>(
      assetUrl("vgg19-phase3-full-pass/vgg19_phase3_full_pass_fixture.json"),
    );
    const packs: VggPackName[] = VGG_PACK_OPTIONS.map((option) => option.name);
    const comparisonRows: PackComparisonRow[] = [];
    let finalWeightsResult: Record<
      string,
      number[] | [number, number, number, number]
    > | null = null;
    for (const pack of packs) {
      try {
        setStatus(`Loading ${pack} weights for pack comparison...`);
        const weightsResult = await loadManifestWeights(pack);
        if (pack === "fp32") finalWeightsResult = weightsResult.weights;
        await initializeWorker(worker);
        setStatus(`Running pack comparison for ${pack}...`);
        const singleRun = await askWorker(worker, {
          type: "run-style-transfer",
          id: `benchmark-full-style-compare-${pack}`,
          optimizer: "sgd",
          inputShape: fixture.inputShape,
          inputImageValues: fixture.inputImageValues,
          contentImageValues: fixture.contentImageValues,
          styleImageValues: fixture.styleImageValues,
          mean: fixture.mean,
          std: fixture.std,
          styleLayerIndices: fixture.styleLayerIndices,
          contentLayerIndex: fixture.contentLayerIndex,
          weights: weightsResult.weights,
          contentWeight: fullContentWeight,
          styleWeight: fullStyleWeight,
          learningRate: fullLearningRate,
          steps: fullSteps,
        });
        if (singleRun.type !== "run-style-transfer-result") {
          throw new Error(`Unexpected worker response type: ${singleRun.type}`);
        }
        if (!singleRun.ok) {
          throw new Error(singleRun.message);
        }
        comparisonRows.push({
          pack,
          elapsedMs: singleRun.stats.elapsedMs,
          avgStepMs: singleRun.stats.avgStepMs,
          finalLoss: singleRun.losses.at(-1) ?? 0,
          downloadSizeBytes: weightsResult.stats.downloadSizeBytes,
        });
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        throw new Error(`Pack comparison failed for ${pack}: ${errorMessage}`, {
          cause: error,
        });
      }
    }
    setPackComparison(comparisonRows);
    setPackAcceptance(buildPackAcceptanceRows(comparisonRows));
    const fp32WeightsResult = await loadManifestWeights("fp32");
    const resolvedWeights = finalWeightsResult ?? fp32WeightsResult.weights;
    await initializeWorker(worker);
    const cappedRuns = Math.max(1, Math.floor(fullRuns));
    const runStats: WorkerRunStats[] = [];
    let lastResult: BenchmarkResult | null = null;

    for (let runIndex = 0; runIndex < cappedRuns; runIndex += 1) {
      setStatus(
        `Running full style-transfer benchmark (${runIndex + 1}/${cappedRuns})...`,
      );
      const optimized = await askWorker(worker, {
        type: "run-style-transfer",
        id: `benchmark-full-style-${runIndex}`,
        optimizer: "sgd",
        inputShape: fixture.inputShape,
        inputImageValues: fixture.inputImageValues,
        contentImageValues: fixture.contentImageValues,
        styleImageValues: fixture.styleImageValues,
        mean: fixture.mean,
        std: fixture.std,
        styleLayerIndices: fixture.styleLayerIndices,
        contentLayerIndex: fixture.contentLayerIndex,
        weights: resolvedWeights,
        contentWeight: fullContentWeight,
        styleWeight: fullStyleWeight,
        learningRate: fullLearningRate,
        steps: fullSteps,
      });
      if (optimized.type !== "run-style-transfer-result") {
        throw new Error("Unexpected full style-transfer run response");
      }
      if (!optimized.ok) throw new Error(optimized.message);
      lastResult = {
        losses: optimized.losses,
        stats: optimized.stats,
        pack: "fp32",
      };
      runStats.push(optimized.stats);
    }

    if (lastResult !== null) setResult(lastResult);
    if (runStats.length > 0) {
      const perf = performance as Performance & {
        memory?: { usedJSHeapSize: number };
      };
      setSummary({
        ...summarizeRuns(runStats),
        downloadSizeBytes: fp32WeightsResult.stats.downloadSizeBytes,
        decodeLoadMs: fp32WeightsResult.stats.decodeLoadMs,
        firstIterationMs: runStats[0].avgStepMs,
        peakMemoryBytes: perf.memory?.usedJSHeapSize,
      });
    }
  };

  const runBenchmark = async (): Promise<void> => {
    setIsRunning(true);
    clearResults();
    const worker = new Worker(
      new URL("./styleTransfer.worker.ts", import.meta.url),
      {
        type: "module",
      },
    );
    try {
      if (activeTab === "first-pool") {
        await runFirstPoolBenchmark(worker);
      } else if (activeTab === "full-style") {
        await runFullStyleBenchmark(worker);
      } else {
        await runKernelLabBenchmark(worker);
      }
      setStatus("Done");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Benchmark failed");
    } finally {
      worker.terminate();
      setIsRunning(false);
    }
  };

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-5 px-6 py-8 text-slate-100">
      <header className="flex flex-col gap-2">
        <h1 className="text-3xl font-semibold">WebGPU Pipeline Benchmark</h1>
        <p className="text-slate-300">
          Measure first-pool and full style-transfer pipeline timings.
        </p>
      </header>

      <div className="flex w-fit overflow-hidden rounded border border-slate-700">
        <button
          className={`px-4 py-2 ${activeTab === "first-pool" ? "bg-emerald-500 text-black" : "bg-slate-900 text-slate-200"}`}
          onClick={() => {
            setActiveTab("first-pool");
            clearResults();
          }}
          type="button"
        >
          First pool
        </button>
        <button
          className={`px-4 py-2 ${activeTab === "full-style" ? "bg-emerald-500 text-black" : "bg-slate-900 text-slate-200"}`}
          onClick={() => {
            setActiveTab("full-style");
            clearResults();
          }}
          type="button"
        >
          Full style transfer
        </button>
        <button
          className={`px-4 py-2 ${activeTab === "kernel-lab" ? "bg-emerald-500 text-black" : "bg-slate-900 text-slate-200"}`}
          onClick={() => {
            setActiveTab("kernel-lab");
            clearResults();
          }}
          type="button"
        >
          Kernel lab
        </button>
      </div>

      <p className="rounded border border-slate-700 bg-slate-900/60 p-3">
        Status: {status}
      </p>

      {activeTab === "first-pool" ? (
        <section className="grid gap-3 rounded border border-slate-700 bg-slate-900/60 p-4 md:grid-cols-3">
          <label className="flex flex-col gap-1">
            Steps
            <input
              min={1}
              type="number"
              value={firstPoolSteps}
              onChange={(event) =>
                setFirstPoolSteps(Number(event.target.value))
              }
            />
          </label>
          <label className="flex flex-col gap-1">
            Runs
            <input
              min={1}
              type="number"
              value={firstPoolRuns}
              onChange={(event) => setFirstPoolRuns(Number(event.target.value))}
            />
          </label>
          <label className="flex flex-col gap-1">
            Learning rate
            <input
              step={0.001}
              type="number"
              value={firstPoolLearningRate}
              onChange={(event) =>
                setFirstPoolLearningRate(Number(event.target.value))
              }
            />
          </label>
          <label className="flex flex-col gap-1">
            Content weight
            <input
              type="number"
              value={firstPoolContentWeight}
              onChange={(event) =>
                setFirstPoolContentWeight(Number(event.target.value))
              }
            />
          </label>
          <label className="flex flex-col gap-1">
            Style weight conv1
            <input
              type="number"
              value={styleWeightConv1}
              onChange={(event) =>
                setStyleWeightConv1(Number(event.target.value))
              }
            />
          </label>
          <label className="flex flex-col gap-1">
            Style weight conv3
            <input
              type="number"
              value={styleWeightConv3}
              onChange={(event) =>
                setStyleWeightConv3(Number(event.target.value))
              }
            />
          </label>
          <label className="flex items-center gap-2">
            <input
              checked={useFirstPoolFusedConvRelu}
              type="checkbox"
              onChange={(event) =>
                setUseFirstPoolFusedConvRelu(event.target.checked)
              }
            />
            <span>Fused conv+relu</span>
          </label>
          <label className="flex items-center gap-2">
            <input
              checked={useFirstPoolFusedUpdateClamp}
              type="checkbox"
              onChange={(event) =>
                setUseFirstPoolFusedUpdateClamp(event.target.checked)
              }
            />
            <span>Fused update+clamp</span>
          </label>
          <label className="flex items-center gap-2">
            <input
              checked={debugValidateStepShapes}
              type="checkbox"
              onChange={(event) =>
                setDebugValidateStepShapes(event.target.checked)
              }
            />
            <span>Shape readback diagnostics</span>
          </label>
          <label className="flex items-center gap-2">
            <input
              checked={debugReadbackGrad}
              type="checkbox"
              onChange={(event) => setDebugReadbackGrad(event.target.checked)}
            />
            <span>Gradient readback diagnostics</span>
          </label>
        </section>
      ) : activeTab === "full-style" ? (
        <section className="grid gap-3 rounded border border-slate-700 bg-slate-900/60 p-4 md:grid-cols-3">
          <label className="flex flex-col gap-1">
            Steps
            <input
              min={1}
              type="number"
              value={fullSteps}
              onChange={(event) => setFullSteps(Number(event.target.value))}
            />
          </label>
          <label className="flex flex-col gap-1">
            Runs
            <input
              min={1}
              type="number"
              value={fullRuns}
              onChange={(event) => setFullRuns(Number(event.target.value))}
            />
          </label>
          <label className="flex flex-col gap-1">
            Learning rate
            <input
              step={0.00001}
              type="number"
              value={fullLearningRate}
              onChange={(event) =>
                setFullLearningRate(Number(event.target.value))
              }
            />
          </label>
          <label className="flex flex-col gap-1">
            Content weight
            <input
              type="number"
              value={fullContentWeight}
              onChange={(event) =>
                setFullContentWeight(Number(event.target.value))
              }
            />
          </label>
          <label className="flex flex-col gap-1">
            Style weight
            <input
              type="number"
              value={fullStyleWeight}
              onChange={(event) =>
                setFullStyleWeight(Number(event.target.value))
              }
            />
          </label>
        </section>
      ) : (
        <section className="grid gap-3 rounded border border-slate-700 bg-slate-900/60 p-4 md:grid-cols-3">
          <label className="flex flex-col gap-1">
            Steps
            <input
              min={1}
              type="number"
              value={kernelLabSteps}
              onChange={(event) =>
                setKernelLabSteps(Number(event.target.value))
              }
            />
          </label>
          <label className="flex flex-col gap-1">
            Learning rate
            <input
              step={0.00001}
              type="number"
              value={fullLearningRate}
              onChange={(event) =>
                setFullLearningRate(Number(event.target.value))
              }
            />
          </label>
        </section>
      )}

      <button
        className="w-fit rounded bg-emerald-500 px-4 py-2 font-semibold text-black disabled:cursor-not-allowed disabled:bg-slate-600 disabled:text-slate-300"
        disabled={isRunning}
        onClick={() => void runBenchmark()}
        type="button"
      >
        {isRunning ? "Running..." : "Run benchmark"}
      </button>

      <section className="rounded border border-slate-700 bg-slate-900/60 p-4">
        <div className="grid gap-2 text-sm md:grid-cols-2">
          <p>Initial loss: {result?.losses[0]?.toFixed(6) ?? "-"}</p>
          <p>Final loss: {result?.losses.at(-1)?.toFixed(6) ?? "-"}</p>
          <p>Steps in last run: {result?.stats?.steps ?? "-"}</p>
          <p>Last elapsed: {result?.stats?.elapsedMs.toFixed(2) ?? "-"} ms</p>
        </div>
        {summary === null ? null : (
          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[620px] border-collapse text-left text-sm">
              <thead className="text-slate-300">
                <tr>
                  <th className="border-b border-slate-700 py-2 pr-4">
                    Metric
                  </th>
                  <th className="border-b border-slate-700 py-2 pr-4">
                    mean / p50 / p95 ms
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="border-b border-slate-800 py-2 pr-4">
                    Elapsed
                  </td>
                  <td className="border-b border-slate-800 py-2 pr-4">
                    {formatMetric(summary.elapsedMs)}
                  </td>
                </tr>
                <tr>
                  <td className="border-b border-slate-800 py-2 pr-4">
                    Forward
                  </td>
                  <td className="border-b border-slate-800 py-2 pr-4">
                    {formatMetric(summary.forwardMs)}
                  </td>
                </tr>
                <tr>
                  <td className="border-b border-slate-800 py-2 pr-4">Loss</td>
                  <td className="border-b border-slate-800 py-2 pr-4">
                    {formatMetric(summary.lossMs)}
                  </td>
                </tr>
                <tr>
                  <td className="border-b border-slate-800 py-2 pr-4">
                    Backward
                  </td>
                  <td className="border-b border-slate-800 py-2 pr-4">
                    {formatMetric(summary.backwardMs)}
                  </td>
                </tr>
                <tr>
                  <td className="border-b border-slate-800 py-2 pr-4">
                    Update
                  </td>
                  <td className="border-b border-slate-800 py-2 pr-4">
                    {formatMetric(summary.updateMs)}
                  </td>
                </tr>
                <tr>
                  <td className="border-b border-slate-800 py-2 pr-4">
                    Readback
                  </td>
                  <td className="border-b border-slate-800 py-2 pr-4">
                    {formatMetric(summary.readbackMs)}
                  </td>
                </tr>
                <tr>
                  <td className="border-b border-slate-800 py-2 pr-4">
                    Mandatory readback
                  </td>
                  <td className="border-b border-slate-800 py-2 pr-4">
                    {formatMetric(summary.mandatoryReadbackMs)}
                  </td>
                </tr>
                <tr>
                  <td className="py-2 pr-4">Diagnostics readback</td>
                  <td className="py-2 pr-4">
                    {formatMetric(summary.diagnosticsReadbackMs)}
                  </td>
                </tr>
                <tr>
                  <td className="py-2 pr-4">Download size (bytes)</td>
                  <td className="py-2 pr-4">
                    {summary.downloadSizeBytes ?? "-"}
                  </td>
                </tr>
                <tr>
                  <td className="py-2 pr-4">Decode/load time (ms)</td>
                  <td className="py-2 pr-4">
                    {summary.decodeLoadMs?.toFixed(2) ?? "-"}
                  </td>
                </tr>
                <tr>
                  <td className="py-2 pr-4">First-iteration latency (ms)</td>
                  <td className="py-2 pr-4">
                    {summary.firstIterationMs?.toFixed(2) ?? "-"}
                  </td>
                </tr>
                <tr>
                  <td className="py-2 pr-4">Peak memory (JS heap bytes)</td>
                  <td className="py-2 pr-4">
                    {summary.peakMemoryBytes ?? "-"}
                  </td>
                </tr>
              </tbody>
            </table>
            <p className="mt-3 text-sm text-slate-400">
              Summary over {summary.runs} run{summary.runs === 1 ? "" : "s"}.
            </p>
          </div>
        )}
        {packComparison === null ? null : (
          <div className="mt-6 overflow-x-auto">
            <h3 className="mb-2 text-sm font-semibold text-slate-200">
              Pack comparison (single run each)
            </h3>
            <table className="w-full min-w-[620px] border-collapse text-left text-sm">
              <thead className="text-slate-300">
                <tr>
                  <th className="border-b border-slate-700 py-2 pr-4">Pack</th>
                  <th className="border-b border-slate-700 py-2 pr-4">
                    Download bytes
                  </th>
                  <th className="border-b border-slate-700 py-2 pr-4">
                    Elapsed ms
                  </th>
                  <th className="border-b border-slate-700 py-2 pr-4">
                    Avg step ms
                  </th>
                  <th className="border-b border-slate-700 py-2 pr-4">
                    Final loss
                  </th>
                </tr>
              </thead>
              <tbody>
                {packComparison.map((row) => (
                  <tr key={row.pack}>
                    <td className="border-b border-slate-800 py-2 pr-4">
                      {row.pack}
                    </td>
                    <td className="border-b border-slate-800 py-2 pr-4">
                      {row.downloadSizeBytes}
                    </td>
                    <td className="border-b border-slate-800 py-2 pr-4">
                      {row.elapsedMs.toFixed(2)}
                    </td>
                    <td className="border-b border-slate-800 py-2 pr-4">
                      {row.avgStepMs.toFixed(2)}
                    </td>
                    <td className="border-b border-slate-800 py-2 pr-4">
                      {row.finalLoss.toFixed(6)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {packAcceptance === null ? null : (
          <div className="mt-4 overflow-x-auto">
            <h3 className="mb-2 text-sm font-semibold text-slate-200">
              Acceptance verdicts
            </h3>
            <ul className="space-y-1 text-sm">
              {packAcceptance.map((row) => (
                <li key={row.pack}>
                  <span
                    className={
                      row.verdict === "pass"
                        ? "text-emerald-300"
                        : "text-rose-300"
                    }
                  >
                    {row.verdict.toUpperCase()}
                  </span>{" "}
                  <span className="text-slate-200">{row.pack}</span>{" "}
                  <span className="text-slate-400">({row.reason})</span>
                </li>
              ))}
            </ul>
          </div>
        )}
        {kernelLabRows === null ? null : (
          <div className="mt-6 overflow-x-auto">
            <h3 className="mb-2 text-sm font-semibold text-slate-200">
              Kernel lab (cumulative variants, default pooled)
            </h3>
            <table className="w-full min-w-[900px] border-collapse text-left text-sm">
              <thead className="text-slate-300">
                <tr>
                  <th className="border-b border-slate-700 py-2 pr-4">
                    Variant
                  </th>
                  <th className="border-b border-slate-700 py-2 pr-4">
                    Elapsed ms
                  </th>
                  <th className="border-b border-slate-700 py-2 pr-4">
                    Forward
                  </th>
                  <th className="border-b border-slate-700 py-2 pr-4">Loss</th>
                  <th className="border-b border-slate-700 py-2 pr-4">
                    Backward
                  </th>
                  <th className="border-b border-slate-700 py-2 pr-4">
                    Update
                  </th>
                  <th className="border-b border-slate-700 py-2 pr-4">
                    Final loss
                  </th>
                  <th className="border-b border-slate-700 py-2 pr-4">
                    Loss delta
                  </th>
                  <th className="border-b border-slate-700 py-2 pr-4">
                    Speedup
                  </th>
                </tr>
              </thead>
              <tbody>
                {kernelLabRows.map((row) => (
                  <tr key={row.name}>
                    <td className="border-b border-slate-800 py-2 pr-4">
                      {row.name}
                    </td>
                    <td className="border-b border-slate-800 py-2 pr-4">
                      {row.ok ? row.elapsedMs.toFixed(2) : "-"}
                    </td>
                    <td className="border-b border-slate-800 py-2 pr-4">
                      {row.ok ? row.forwardMs.toFixed(2) : "-"}
                    </td>
                    <td className="border-b border-slate-800 py-2 pr-4">
                      {row.ok ? row.lossMs.toFixed(2) : "-"}
                    </td>
                    <td className="border-b border-slate-800 py-2 pr-4">
                      {row.ok ? row.backwardMs.toFixed(2) : "-"}
                    </td>
                    <td className="border-b border-slate-800 py-2 pr-4">
                      {row.ok ? row.updateMs.toFixed(2) : "-"}
                    </td>
                    <td className="border-b border-slate-800 py-2 pr-4">
                      {row.ok ? row.finalLoss.toFixed(6) : "-"}
                    </td>
                    <td className="border-b border-slate-800 py-2 pr-4">
                      {row.ok ? row.lossDeltaFromBaseline.toFixed(6) : "-"}
                    </td>
                    <td className="border-b border-slate-800 py-2 pr-4">
                      {row.ok ? `${row.speedupVsBaseline.toFixed(3)}x` : "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {kernelLabRows.some((row) => row.error !== undefined) ? (
              <ul className="mt-3 space-y-1 text-sm text-rose-300">
                {kernelLabRows
                  .filter((row) => row.error !== undefined)
                  .map((row) => (
                    <li key={`${row.name}-error`}>
                      {row.name}: {row.error}
                    </li>
                  ))}
              </ul>
            ) : null}
          </div>
        )}
      </section>
    </main>
  );
};
