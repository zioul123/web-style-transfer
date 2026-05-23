import { useMemo, useState, type ReactElement } from "react";
import type { WorkerFirstPoolBenchmarkStats, WorkerRequest, WorkerResponse } from "./types";

type VggWeightsFixture = {
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

type VggCaseFixture = {
  inputShape: [number, number, number, number];
  inputValues: number[];
  mean: [number, number, number];
  std: [number, number, number];
};

type BenchmarkResult = {
  losses: number[];
  stats?: WorkerFirstPoolBenchmarkStats;
};

type BenchmarkPreset = "baseline" | "debug-shapes" | "debug-grad" | "debug-both";

type BenchmarkRunSummary = {
  runs: number;
  elapsedMs: { mean: number; p50: number; p95: number };
  readbackMs: { mean: number; p50: number; p95: number };
  mandatoryReadbackMs: { mean: number; p50: number; p95: number };
  diagnosticsReadbackMs: { mean: number; p50: number; p95: number };
};

const percentile = (sortedValues: readonly number[], p: number): number => {
  if (sortedValues.length === 0) return 0;
  const idx = Math.min(sortedValues.length - 1, Math.max(0, Math.ceil((p / 100) * sortedValues.length) - 1));
  return sortedValues[idx];
};

const summarize = (values: readonly number[]): { mean: number; p50: number; p95: number } => {
  if (values.length === 0) return { mean: 0, p50: 0, p95: 0 };
  const sorted = [...values].sort((a, b) => a - b);
  const mean = values.reduce((acc, value) => acc + value, 0) / values.length;
  return { mean, p50: percentile(sorted, 50), p95: percentile(sorted, 95) };
};

const loadJson = async <T,>(url: string): Promise<T> => {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to load ${url}: ${response.status}`);
  return (await response.json()) as T;
};

const askWorker = (worker: Worker, payload: WorkerRequest): Promise<WorkerResponse> =>
  new Promise((resolve) => {
    const handler = (event: MessageEvent<WorkerResponse>): void => {
      if (event.data.id !== payload.id) return;
      worker.removeEventListener("message", handler);
      resolve(event.data);
    };
    worker.addEventListener("message", handler);
    worker.postMessage(payload);
  });

export const BenchmarkApp = (): ReactElement => {
  const [preset, setPreset] = useState<BenchmarkPreset>("baseline");
  const [steps, setSteps] = useState<number>(6);
  const [runs, setRuns] = useState<number>(5);
  const [learningRate, setLearningRate] = useState<number>(0.15);
  const [contentWeight, setContentWeight] = useState<number>(1);
  const [styleWeightConv1, setStyleWeightConv1] = useState<number>(30);
  const [styleWeightConv3, setStyleWeightConv3] = useState<number>(120);
  const [debugValidateStepShapes, setDebugValidateStepShapes] = useState<boolean>(false);
  const [debugReadbackGrad, setDebugReadbackGrad] = useState<boolean>(false);
  const [status, setStatus] = useState<string>("Ready");
  const [result, setResult] = useState<BenchmarkResult | null>(null);
  const [summary, setSummary] = useState<BenchmarkRunSummary | null>(null);

  const applyPreset = (nextPreset: BenchmarkPreset): void => {
    setPreset(nextPreset);
    if (nextPreset === "baseline") {
      setDebugValidateStepShapes(false);
      setDebugReadbackGrad(false);
      return;
    }
    if (nextPreset === "debug-shapes") {
      setDebugValidateStepShapes(true);
      setDebugReadbackGrad(false);
      return;
    }
    if (nextPreset === "debug-grad") {
      setDebugValidateStepShapes(false);
      setDebugReadbackGrad(true);
      return;
    }
    setDebugValidateStepShapes(true);
    setDebugReadbackGrad(true);
  };

  const readbackBreakdown = useMemo((): string => {
    if (result?.stats === undefined) return "No stats yet";
    return `total=${result.stats.readbackMs.toFixed(2)}ms, mandatory=${result.stats.mandatoryReadbackMs.toFixed(2)}ms, diagnostics=${result.stats.diagnosticsReadbackMs.toFixed(2)}ms`;
  }, [result]);

  const runBenchmark = async (): Promise<void> => {
    setStatus("Loading fixtures...");
    setResult(null);
    setSummary(null);
    const worker = new Worker(new URL("./styleTransfer.worker.ts", import.meta.url), { type: "module" });
    try {
      const [weights, caseData] = await Promise.all([
        loadJson<VggWeightsFixture>("/vgg19-first-pool/vgg19_first_pool_weights.json"),
        loadJson<VggCaseFixture>("/vgg19-first-pool/vgg19_first_pool_case_madeira16.json"),
      ]);
      setStatus("Initializing WebGPU...");
      const init = await askWorker(worker, { type: "init-webgpu", id: "benchmark-init" });
      if (init.type !== "webgpu-init-result" || !init.ok) {
        throw new Error(init.type === "webgpu-init-result" ? init.message : "Unexpected init response");
      }

      const cappedRuns = Math.max(1, Math.floor(runs));
      const runStats: WorkerFirstPoolBenchmarkStats[] = [];
      let lastResult: BenchmarkResult | null = null;

      for (let runIndex = 0; runIndex < cappedRuns; runIndex += 1) {
        setStatus(`Running first-pool benchmark (${runIndex + 1}/${cappedRuns})...`);
        const optimized = await askWorker(worker, {
          type: "run-first-pool-optimizer",
          id: `benchmark-run-${runIndex}`,
          inputShape: caseData.inputShape,
          mean: caseData.mean,
          std: caseData.std,
          contentImageValues: caseData.inputValues,
          styleImageValues: [...caseData.inputValues].reverse(),
          initialInputValues: caseData.inputValues,
          conv1Weight: { shape: weights.conv1WeightShape, values: weights.conv1WeightValues },
          conv1Bias: weights.conv1BiasValues,
          conv2Weight: { shape: weights.conv2WeightShape, values: weights.conv2WeightValues },
          conv2Bias: weights.conv2BiasValues,
          conv3Weight: { shape: weights.conv3WeightShape, values: weights.conv3WeightValues },
          conv3Bias: weights.conv3BiasValues,
          contentWeight,
          styleWeightConv1,
          styleWeightConv3,
          learningRate,
          steps,
          collectBenchmarkStats: true,
          debugValidateStepShapes,
          debugReadbackGrad,
        });
        if (optimized.type !== "run-first-pool-optimizer-result") throw new Error("Unexpected run response");
        if (!optimized.ok) throw new Error(optimized.message);
        lastResult = { losses: optimized.losses, stats: optimized.stats };
        if (optimized.stats !== undefined) runStats.push(optimized.stats);
      }

      if (lastResult !== null) setResult(lastResult);
      if (runStats.length > 0) {
        setSummary({
          runs: runStats.length,
          elapsedMs: summarize(runStats.map((entry) => entry.elapsedMs)),
          readbackMs: summarize(runStats.map((entry) => entry.readbackMs)),
          mandatoryReadbackMs: summarize(runStats.map((entry) => entry.mandatoryReadbackMs)),
          diagnosticsReadbackMs: summarize(runStats.map((entry) => entry.diagnosticsReadbackMs)),
        });
      }
      setStatus("Done");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Benchmark failed");
    } finally {
      worker.terminate();
    }
  };

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-4xl flex-col gap-4 px-6 py-8 text-slate-100">
      <h1 className="text-3xl font-semibold">First-Pool Benchmark</h1>
      <p className="text-slate-300">Use this page to compare mandatory vs diagnostics readback overhead.</p>
      <p className="rounded border border-slate-700 bg-slate-900/60 p-3">Status: {status}</p>

      <section className="grid gap-3 rounded-xl border border-slate-700 bg-slate-900/60 p-4 md:grid-cols-3">
        <label className="flex flex-col gap-1">Preset
          <select value={preset} onChange={(event) => { const nextPreset = event.target.value; if (nextPreset === "baseline" || nextPreset === "debug-shapes" || nextPreset === "debug-grad" || nextPreset === "debug-both") applyPreset(nextPreset); }}>
            <option value="baseline">baseline</option>
            <option value="debug-shapes">debug-shapes</option>
            <option value="debug-grad">debug-grad</option>
            <option value="debug-both">debug-both</option>
          </select>
        </label>
        <label className="flex flex-col gap-1">Steps<input type="number" min={1} value={steps} onChange={(event) => setSteps(Number(event.target.value))} /></label>
        <label className="flex flex-col gap-1">Runs<input type="number" min={1} value={runs} onChange={(event) => setRuns(Number(event.target.value))} /></label>
        <label className="flex flex-col gap-1">Learning rate<input type="number" step={0.001} value={learningRate} onChange={(event) => setLearningRate(Number(event.target.value))} /></label>
        <label className="flex flex-col gap-1">Content weight<input type="number" value={contentWeight} onChange={(event) => setContentWeight(Number(event.target.value))} /></label>
        <label className="flex flex-col gap-1">Style weight conv1<input type="number" value={styleWeightConv1} onChange={(event) => setStyleWeightConv1(Number(event.target.value))} /></label>
        <label className="flex flex-col gap-1">Style weight conv3<input type="number" value={styleWeightConv3} onChange={(event) => setStyleWeightConv3(Number(event.target.value))} /></label>
        <div className="flex flex-col gap-2 pt-2">
          <label className="flex items-center gap-2"><input type="checkbox" checked={debugValidateStepShapes} onChange={(event) => setDebugValidateStepShapes(event.target.checked)} /><span>debugValidateStepShapes</span></label>
          <label className="flex items-center gap-2"><input type="checkbox" checked={debugReadbackGrad} onChange={(event) => setDebugReadbackGrad(event.target.checked)} /><span>debugReadbackGrad</span></label>
        </div>
      </section>

      <button className="w-fit rounded bg-emerald-500 px-4 py-2 font-semibold text-black" onClick={() => void runBenchmark()}>Run benchmark</button>

      <section className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
        <p>Readback breakdown: {readbackBreakdown}</p>
        <p>Initial loss: {result?.losses[0]?.toFixed(6) ?? "—"}</p>
        <p>Final loss: {(result?.losses.at(-1))?.toFixed(6) ?? "—"}</p>
        {summary === null ? null : (
          <div className="mt-3 grid gap-1 text-sm text-slate-300">
            <p>Summary over {summary.runs} runs</p>
            <p>Elapsed ms (mean/p50/p95): {summary.elapsedMs.mean.toFixed(2)} / {summary.elapsedMs.p50.toFixed(2)} / {summary.elapsedMs.p95.toFixed(2)}</p>
            <p>Total readback ms (mean/p50/p95): {summary.readbackMs.mean.toFixed(2)} / {summary.readbackMs.p50.toFixed(2)} / {summary.readbackMs.p95.toFixed(2)}</p>
            <p>Mandatory readback ms (mean/p50/p95): {summary.mandatoryReadbackMs.mean.toFixed(2)} / {summary.mandatoryReadbackMs.p50.toFixed(2)} / {summary.mandatoryReadbackMs.p95.toFixed(2)}</p>
            <p>Diagnostics readback ms (mean/p50/p95): {summary.diagnosticsReadbackMs.mean.toFixed(2)} / {summary.diagnosticsReadbackMs.p50.toFixed(2)} / {summary.diagnosticsReadbackMs.p95.toFixed(2)}</p>
          </div>
        )}
      </section>
    </main>
  );
};
