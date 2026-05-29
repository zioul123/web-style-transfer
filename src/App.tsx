import { useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import { useStyleTransferController } from "./features/style-transfer/hooks/useStyleTransferController";
import type {
  KernelConvBackwardInput,
  KernelConvForward,
  KernelGramKernel,
  KernelStyleBackward,
  KernelWeightStorage,
  OptimizerMode,
  ResolutionPreset,
} from "./features/style-transfer/types/controller";
import {
  VGG_PACK_OPTIONS,
  type VggPackName,
} from "./features/style-transfer/modelPacks";
import { assetUrl } from "./shared/assetUrls";

const resolutionOptions: readonly ResolutionPreset[] = [
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
const optimizerOptions: readonly OptimizerMode[] = ["sgd", "adam", "lbfgs"];
const hostedVggPackOptions: readonly VggPackName[] = [
  "int8-per-channel",
  "int4log-experimental",
];
const kernelGramKernelOptions: readonly KernelGramKernel[] = [
  "scalar",
  "parallel-dot",
  "symmetric-parallel-dot",
];
const kernelStyleBackwardOptions: readonly KernelStyleBackward[] = [
  "two-pass",
  "fused-from-gram-diff",
];
const kernelConvForwardOptions: readonly KernelConvForward[] = [
  "scalar",
  "spatial-vec4",
  "tiled-spatial",
];
const kernelConvBackwardInputOptions: readonly KernelConvBackwardInput[] = [
  "scalar",
  "spatial-vec4",
  "transposed-weight-spatial-vec4",
];
const kernelWeightStorageOptions: readonly KernelWeightStorage[] = [
  "fp32",
  "fp16-storage",
  "int8-dequant-experimental",
];
const isResolutionPreset = (value: string): value is ResolutionPreset =>
  resolutionOptions.some((option) => option === value);
const isOptimizerMode = (value: string): value is OptimizerMode =>
  optimizerOptions.some((option) => option === value);
const isVggPackName = (value: string): value is VggPackName =>
  VGG_PACK_OPTIONS.some((option) => option.name === value);
const isHostedVggPackName = (value: VggPackName): boolean =>
  hostedVggPackOptions.some((option) => option === value);
const isLocalhost = (): boolean => {
  const hostname = window.location.hostname;
  return (
    hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1"
  );
};
const isKernelGramKernel = (value: string): value is KernelGramKernel =>
  kernelGramKernelOptions.some((option) => option === value);
const isKernelStyleBackward = (value: string): value is KernelStyleBackward =>
  kernelStyleBackwardOptions.some((option) => option === value);
const isKernelConvForward = (value: string): value is KernelConvForward =>
  kernelConvForwardOptions.some((option) => option === value);
const isKernelConvBackwardInput = (
  value: string,
): value is KernelConvBackwardInput =>
  kernelConvBackwardInputOptions.some((option) => option === value);
const isKernelWeightStorage = (value: string): value is KernelWeightStorage =>
  kernelWeightStorageOptions.some((option) => option === value);

const kernelConvForwardLabel = (option: KernelConvForward): string => {
  if (option === "spatial-vec4") return "spatial-vec4 (batched scalar)";
  if (option === "tiled-spatial") return "tiled-spatial (unsupported)";
  return option;
};

const kernelConvBackwardInputLabel = (
  option: KernelConvBackwardInput,
): string => {
  if (option === "spatial-vec4") return "spatial-vec4 (unsupported)";
  return option;
};

const kernelWeightStorageLabel = (option: KernelWeightStorage): string => {
  if (option === "fp16-storage") return "fp16-storage (requires persistent)";
  if (option === "int8-dequant-experimental")
    return "int8-dequant-experimental (unsupported)";
  return option;
};

type ImageCard =
  | {
      kind: "source";
      label: "Content" | "Style";
      src: string | null;
      inputRef: RefObject<HTMLInputElement | null>;
      target: "content" | "style";
    }
  | {
      kind: "output";
      label: "Output";
      src: string | null;
    };

function App() {
  const { controls, status, images, canRun, setIsRunning, onUpload } =
    useStyleTransferController();
  const { selectedPack, setSelectedPack } = controls;
  const [showOptions, setShowOptions] = useState<boolean>(false);
  const contentFileInputRef = useRef<HTMLInputElement | null>(null);
  const styleFileInputRef = useRef<HTMLInputElement | null>(null);
  const cacheSizeMb = (status.modelCacheBytes / (1024 * 1024)).toFixed(2);
  const benchmarkUrl = assetUrl("benchmark");
  const canUseAllModelPacks = isLocalhost();
  const modelPackOptions = canUseAllModelPacks
    ? VGG_PACK_OPTIONS
    : VGG_PACK_OPTIONS.filter((option) => isHostedVggPackName(option.name));
  const imageCards: readonly ImageCard[] = [
    {
      kind: "source",
      label: "Content",
      src: images.contentImage,
      inputRef: contentFileInputRef,
      target: "content",
    },
    {
      kind: "source",
      label: "Style",
      src: images.styleImage,
      inputRef: styleFileInputRef,
      target: "style",
    },
    { kind: "output", label: "Output", src: images.outputImage },
  ];

  useEffect(() => {
    if (canUseAllModelPacks || isHostedVggPackName(selectedPack)) {
      return;
    }
    setSelectedPack("int8log-per-channel");
  }, [canUseAllModelPacks, selectedPack, setSelectedPack]);

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(14,165,233,0.18),_transparent_34rem),linear-gradient(135deg,_#020617_0%,_#111827_48%,_#0f172a_100%)] text-slate-100">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-5 px-4 py-5 sm:px-6 sm:py-8">
        <header className="flex flex-col gap-5 rounded-lg border border-white/10 bg-slate-950/55 p-5 shadow-2xl shadow-black/30 sm:p-6 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-2xl">
            <p className="text-sm font-semibold uppercase text-sky-300">
              Browser-native neural style transfer
            </p>
            <h1 className="mt-2 text-3xl font-semibold text-white">
              WebGPU Style Transfer
            </h1>
            <p className="mt-3 text-sm leading-6 text-slate-300">
              Stylize your photographs in the browser with WebGPU-powered
              optimization and a handful of device-tuned controls.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:flex sm:flex-wrap sm:justify-end">
            <div className="rounded-lg border border-emerald-300/20 bg-emerald-300/10 px-3 py-2">
              <p className="text-xs uppercase text-emerald-200">Iterations</p>
              <p className="text-xl font-semibold text-white">
                {status.iterations}
              </p>
            </div>
            <div className="rounded-lg border border-sky-300/20 bg-sky-300/10 px-3 py-2">
              <p className="text-xs uppercase text-sky-200">Loss</p>
              <p className="text-xl font-semibold text-white">
                {status.lastLoss === null ? "-" : status.lastLoss.toFixed(4)}
              </p>
            </div>
          </div>
        </header>

        <section className="grid gap-4 md:grid-cols-3">
          {imageCards.map((image) => (
            <figure
              className="overflow-hidden rounded-lg border border-white/10 bg-slate-950/65 shadow-xl shadow-black/20"
              key={image.label}
            >
              <div className="flex aspect-[4/5] items-center justify-center bg-slate-900/80">
                {image.src === null ? (
                  <p className="px-4 text-center text-sm text-slate-500">
                    Waiting for {image.label.toLowerCase()} image
                  </p>
                ) : (
                  <img
                    className="h-full w-full object-contain"
                    src={image.src}
                    alt={image.label}
                  />
                )}
              </div>
              <figcaption className="flex items-center justify-between border-t border-white/10 px-4 py-3 text-sm">
                <span className="font-semibold text-slate-100">
                  {image.label}
                </span>
                {image.kind === "source" ? (
                  <>
                    <input
                      className="hidden"
                      ref={image.inputRef}
                      type="file"
                      accept="image/*"
                      onChange={(event) => void onUpload(event, image.target)}
                    />
                    <button
                      className="rounded-lg bg-sky-300 px-3 py-1.5 font-semibold text-sky-950 transition hover:bg-sky-200"
                      type="button"
                      onClick={() => image.inputRef.current?.click()}
                    >
                      Select
                    </button>
                  </>
                ) : (
                  <span className="text-slate-400">Optimized</span>
                )}
              </figcaption>
            </figure>
          ))}
        </section>

        <section className="grid gap-3 rounded-lg border border-white/10 bg-slate-950/55 p-4 text-sm md:grid-cols-2">
          <p>{status.workerStatus}</p>
          <p>{status.gpuStatus}</p>
          <p>
            Model cache: {status.modelCacheState} ({cacheSizeMb} MB)
          </p>
          <p>
            {status.modelCacheState === "downloading"
              ? "Model downloading..."
              : "Model cached"}
          </p>
          <p className="text-slate-300 md:col-span-2">{status.gpuInfo}</p>
        </section>

        <section className="flex flex-col gap-3 rounded-lg border border-white/10 bg-slate-950/55 p-4 sm:flex-row sm:items-center sm:justify-between">
          <button
            className="rounded-lg bg-slate-100 px-4 py-2 font-semibold text-slate-950 transition hover:bg-white"
            type="button"
            onClick={() => setShowOptions((value) => !value)}
            aria-expanded={showOptions}
          >
            {showOptions ? "Hide options" : "Show options"}
          </button>
          <a
            className="text-sm font-medium text-sky-300 underline underline-offset-4 hover:text-sky-200"
            href={benchmarkUrl}
          >
            Benchmark settings for your device
          </a>
        </section>

        {showOptions ? (
          <section className="grid gap-4 rounded-lg border border-white/10 bg-slate-950/55 p-4 text-sm md:grid-cols-3">
            <label className="flex flex-col gap-1">
              Content image
              <input
                className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm text-slate-200"
                type="file"
                accept="image/*"
                onChange={(event) => void onUpload(event, "content")}
              />
            </label>
            <label className="flex flex-col gap-1">
              Style image
              <input
                className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm text-slate-200"
                type="file"
                accept="image/*"
                onChange={(event) => void onUpload(event, "style")}
              />
            </label>
            <label className="flex flex-col gap-1">
              Content resolution
              <select
                className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm text-slate-100"
                value={controls.contentResolution}
                onChange={(event) => {
                  const nextResolution = event.target.value;
                  if (isResolutionPreset(nextResolution)) {
                    controls.setContentResolution(nextResolution);
                  }
                }}
              >
                {resolutionOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              Style resolution
              <select
                className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm text-slate-100"
                value={controls.styleResolution}
                onChange={(event) => {
                  const nextResolution = event.target.value;
                  if (isResolutionPreset(nextResolution)) {
                    controls.setStyleResolution(nextResolution);
                  }
                }}
              >
                {resolutionOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              Model pack
              <select
                aria-label="Model pack"
                className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm text-slate-100"
                value={controls.selectedPack}
                onChange={(event) => {
                  const nextPack = event.target.value;
                  if (isVggPackName(nextPack)) {
                    controls.setSelectedPack(nextPack);
                  }
                }}
              >
                {modelPackOptions.map((option) => (
                  <option key={option.name} value={option.name}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              Chunk steps
              <input
                className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm text-slate-100"
                type="number"
                min={1}
                max={20}
                value={controls.stepsPerChunk}
                onChange={(event) =>
                  controls.setStepsPerChunk(Number(event.target.value))
                }
              />
            </label>
            <label className="flex flex-col gap-1">
              Style weight
              <input
                className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm text-slate-100"
                type="number"
                value={controls.styleWeight}
                onChange={(event) =>
                  controls.setStyleWeight(Number(event.target.value))
                }
              />
            </label>
            <label className="flex flex-col gap-1">
              Content weight
              <input
                className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm text-slate-100"
                type="number"
                value={controls.contentWeight}
                onChange={(event) =>
                  controls.setContentWeight(Number(event.target.value))
                }
              />
            </label>
            <label className="flex flex-col gap-1">
              Learning rate
              <input
                className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm text-slate-100"
                type="number"
                step={0.001}
                value={controls.learningRate}
                onChange={(event) =>
                  controls.setLearningRate(Number(event.target.value))
                }
              />
            </label>
            <label className="flex flex-col gap-1">
              Optimizer
              <select
                className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm text-slate-100"
                value={controls.optimizer}
                onChange={(event) => {
                  const nextOptimizer = event.target.value;
                  if (isOptimizerMode(nextOptimizer)) {
                    controls.setOptimizer(nextOptimizer);
                  }
                }}
              >
                <option value="sgd">SGD</option>
                <option value="adam">Adam</option>
                <option value="lbfgs">L-BFGS</option>
              </select>
            </label>
            <div className="flex flex-col gap-3 rounded-lg border border-white/10 bg-slate-900/70 p-3 md:col-span-3">
              <p className="text-sm font-semibold text-slate-200">
                Kernel Options
              </p>
              <div className="grid gap-2 text-sm md:grid-cols-3">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={controls.useCachedPipelines}
                    onChange={(event) =>
                      controls.setUseCachedPipelines(event.target.checked)
                    }
                  />
                  Cached pipelines
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={controls.usePersistentWeightBuffers}
                    onChange={(event) =>
                      controls.setUsePersistentWeightBuffers(
                        event.target.checked,
                      )
                    }
                  />
                  Persistent weight buffers
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={controls.useStepBufferPool}
                    onChange={(event) =>
                      controls.setUseStepBufferPool(event.target.checked)
                    }
                  />
                  Step buffer pool
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={controls.usePoolBackwardScatter}
                    onChange={(event) =>
                      controls.setUsePoolBackwardScatter(event.target.checked)
                    }
                  />
                  Pool backward scatter
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={controls.useVec4Pointwise}
                    onChange={(event) =>
                      controls.setUseVec4Pointwise(event.target.checked)
                    }
                  />
                  Vec4 pointwise
                </label>
              </div>
              <div className="grid gap-2 md:grid-cols-2">
                <label className="flex flex-col gap-1">
                  Gram kernel
                  <select
                    className="rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-slate-100"
                    value={controls.gramKernel}
                    onChange={(event) => {
                      const nextKernel = event.target.value;
                      if (isKernelGramKernel(nextKernel)) {
                        controls.setGramKernel(nextKernel);
                      }
                    }}
                  >
                    {kernelGramKernelOptions.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  Style backward
                  <select
                    className="rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-slate-100"
                    value={controls.styleBackward}
                    onChange={(event) => {
                      const nextStyleBackward = event.target.value;
                      if (isKernelStyleBackward(nextStyleBackward)) {
                        controls.setStyleBackward(nextStyleBackward);
                      }
                    }}
                  >
                    {kernelStyleBackwardOptions.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  Conv forward
                  <select
                    className="rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-slate-100"
                    value={controls.convForwardKernel}
                    onChange={(event) => {
                      const nextConvForward = event.target.value;
                      if (isKernelConvForward(nextConvForward)) {
                        controls.setConvForwardKernel(nextConvForward);
                      }
                    }}
                  >
                    {kernelConvForwardOptions.map((option) => (
                      <option key={option} value={option}>
                        {kernelConvForwardLabel(option)}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  Conv backward-input
                  <select
                    className="rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-slate-100"
                    value={controls.convBackwardInputKernel}
                    onChange={(event) => {
                      const nextConvBackwardInput = event.target.value;
                      if (isKernelConvBackwardInput(nextConvBackwardInput)) {
                        controls.setConvBackwardInputKernel(
                          nextConvBackwardInput,
                        );
                      }
                    }}
                  >
                    {kernelConvBackwardInputOptions.map((option) => (
                      <option key={option} value={option}>
                        {kernelConvBackwardInputLabel(option)}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  Weight storage
                  <select
                    className="rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-slate-100"
                    value={controls.weightStorage}
                    onChange={(event) => {
                      const nextWeightStorage = event.target.value;
                      if (isKernelWeightStorage(nextWeightStorage)) {
                        controls.setWeightStorage(nextWeightStorage);
                      }
                    }}
                  >
                    {kernelWeightStorageOptions.map((option) => (
                      <option key={option} value={option}>
                        {kernelWeightStorageLabel(option)}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            </div>
            {controls.optimizer === "adam" ? (
              <>
                <label className="flex flex-col gap-1">
                  Adam β1
                  <input
                    className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm text-slate-100"
                    type="number"
                    step={0.001}
                    value={controls.adamBeta1}
                    onChange={(event) =>
                      controls.setAdamBeta1(Number(event.target.value))
                    }
                  />
                </label>
                <label className="flex flex-col gap-1">
                  Adam β2
                  <input
                    className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm text-slate-100"
                    type="number"
                    step={0.001}
                    value={controls.adamBeta2}
                    onChange={(event) =>
                      controls.setAdamBeta2(Number(event.target.value))
                    }
                  />
                </label>
                <label className="flex flex-col gap-1">
                  Adam ε
                  <input
                    className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm text-slate-100"
                    type="number"
                    step={0.00000001}
                    value={controls.adamEpsilon}
                    onChange={(event) =>
                      controls.setAdamEpsilon(Number(event.target.value))
                    }
                  />
                </label>
              </>
            ) : null}
            {controls.optimizer === "lbfgs" ? (
              <>
                <label className="flex flex-col gap-1">
                  L-BFGS history
                  <input
                    className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm text-slate-100"
                    type="number"
                    min={1}
                    step={1}
                    value={controls.lbfgsMemory}
                    onChange={(event) =>
                      controls.setLbfgsMemory(Number(event.target.value))
                    }
                  />
                </label>
                <label className="flex flex-col gap-1">
                  L-BFGS tolerance change
                  <input
                    className="rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-sm text-slate-100"
                    type="number"
                    step={0.00000001}
                    value={controls.lbfgsEpsilon}
                    onChange={(event) =>
                      controls.setLbfgsEpsilon(Number(event.target.value))
                    }
                  />
                </label>
              </>
            ) : null}
            <label className="flex items-center gap-2 md:col-span-2">
              <input
                type="checkbox"
                checked={controls.synchronizePhaseTimings}
                onChange={(event) =>
                  controls.setSynchronizePhaseTimings(event.target.checked)
                }
              />
              Synchronize phase timings (profiling accuracy)
            </label>
          </section>
        ) : null}

        <section className="grid gap-3 rounded-lg border border-white/10 bg-slate-950/55 p-4 sm:grid-cols-2 lg:flex lg:items-center">
          <button
            className="rounded-lg bg-emerald-400 px-4 py-3 font-semibold text-emerald-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-50"
            onClick={() => setIsRunning(true)}
            disabled={status.isRunning || !canRun}
          >
            ▶ Play
          </button>
          <button
            className="rounded-lg bg-amber-400 px-4 py-3 font-semibold text-amber-950 transition hover:bg-amber-300 disabled:cursor-not-allowed disabled:opacity-50"
            onClick={() => setIsRunning(false)}
            disabled={!status.isRunning}
          >
            ⏸ Pause
          </button>
          <button
            className="rounded-lg bg-zinc-300 px-4 py-3 font-semibold text-zinc-950 transition hover:bg-zinc-200 disabled:cursor-not-allowed disabled:opacity-50"
            onClick={() => void controls.clearModelCache()}
            disabled={status.isRunning}
          >
            Clear model cache
          </button>
          <button
            className="rounded-lg bg-sky-400 px-4 py-3 font-semibold text-sky-950 transition hover:bg-sky-300 disabled:cursor-not-allowed disabled:opacity-50"
            onClick={controls.resetOptimizerState}
            disabled={status.isRunning}
          >
            Reset optimizer state
          </button>
          <button
            className="rounded-lg bg-rose-400 px-4 py-3 font-semibold text-rose-950 transition hover:bg-rose-300 disabled:cursor-not-allowed disabled:opacity-50"
            onClick={controls.resetOutputImage}
            disabled={images.contentImage === null}
          >
            Reset output image
          </button>
        </section>

        <details className="rounded-lg border border-white/10 bg-slate-950/55 p-4">
          <summary className="cursor-pointer font-semibold">View stats</summary>
          {status.runStats === null ? (
            <p className="mt-3 text-slate-400">
              No stats yet. Run at least one chunk.
            </p>
          ) : (
            <div className="mt-3 grid gap-2 text-sm md:grid-cols-2">
              <p>Chunk steps: {status.runStats.steps}</p>
              <p>Kernel config: {controls.kernelConfigSummary}</p>
              <p>Total: {status.runStats.elapsedMs.toFixed(1)} ms</p>
              <p>Avg step: {status.runStats.avgStepMs.toFixed(1)} ms</p>
              <p>Forward: {status.runStats.forwardMs.toFixed(1)} ms</p>
              <p>Loss: {status.runStats.lossMs.toFixed(1)} ms</p>
              <p>Backward: {status.runStats.backwardMs.toFixed(1)} ms</p>
              <p>Update + clamp: {status.runStats.updateMs.toFixed(1)} ms</p>
            </div>
          )}
        </details>

        {/* Three.js preview canvas hidden for now; restore it when it has a useful visual treatment. */}

        <section className="rounded-lg border border-dashed border-sky-400/50 bg-sky-400/5 p-4 text-sm text-sky-200">
          Reserved: add "How we made it" blog post link/teaser here.
        </section>
      </div>
    </main>
  );
}

export default App;
