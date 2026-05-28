import { Canvas, useLoader } from "@react-three/fiber";
import type { ReactElement } from "react";
import { TextureLoader } from "three";
import { useStyleTransferController } from "./features/style-transfer/hooks/useStyleTransferController";
import type {
  KernelVariantMode,
  OptimizerMode,
  ResolutionPreset,
} from "./features/style-transfer/types/controller";
import { VGG_PACK_OPTIONS, type VggPackName } from "./features/style-transfer/modelPacks";

const resolutionOptions: readonly ResolutionPreset[] = [
  "128x128",
  "128x192",
  "192x128",
  "256x256",
  "256x384",
];
const optimizerOptions: readonly OptimizerMode[] = ["sgd", "adam", "lbfgs"];
const kernelVariantOptions: readonly KernelVariantMode[] = [
  "baseline",
  "cached-pipelines",
  "cached-persistent-weights",
  "cached-persistent-weights-step-pool",
  "cached-persistent-weights-pool-scatter",
  "cached-persistent-weights-step-pool-pool-scatter",
  "cached-persistent-weights-step-pool-pool-scatter-vec4-pointwise",
];
const isOptimizerMode = (value: string): value is OptimizerMode =>
  optimizerOptions.includes(value as OptimizerMode);
const isKernelVariantMode = (value: string): value is KernelVariantMode =>
  kernelVariantOptions.includes(value as KernelVariantMode);
const isVggPackName = (value: string): value is VggPackName =>
  VGG_PACK_OPTIONS.some((option) => option.name === value);

type TextureImageSize = {
  readonly width: number;
  readonly height: number;
};

const hasTextureImageSize = (value: unknown): value is TextureImageSize => {
  if (typeof value !== "object" || value === null) return false;
  return (
    "width" in value &&
    "height" in value &&
    typeof value.width === "number" &&
    typeof value.height === "number"
  );
};

const Plane = ({ url, x }: { url: string; x: number }): ReactElement => {
  const texture = useLoader(TextureLoader, url);
  const imageWidth = hasTextureImageSize(texture.image)
    ? texture.image.width
    : 1;
  const imageHeight =
    hasTextureImageSize(texture.image) && texture.image.height > 0
      ? texture.image.height
      : 1;
  const aspect = imageWidth / imageHeight;
  const planeWidth = aspect >= 1 ? 1.75 : 1.75 * aspect;
  const planeHeight = aspect >= 1 ? 1.75 / aspect : 1.75;
  return (
    <mesh position={[x, 0, 0]}>
      <planeGeometry args={[planeWidth, planeHeight]} />
      <meshBasicMaterial map={texture} />
    </mesh>
  );
};

function App() {
  const { controls, status, images, canRun, setIsRunning, onUpload } =
    useStyleTransferController();
  const cacheSizeMb = (status.modelCacheBytes / (1024 * 1024)).toFixed(2);

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-6 px-6 py-8 text-slate-100">
      <h1 className="text-3xl font-semibold">
        WebGPU Style Transfer — Phase 6 UI
      </h1>

      <section className="grid gap-4 rounded-xl border border-slate-700 bg-slate-900/60 p-4 md:grid-cols-2">
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
        <p className="md:col-span-2 text-slate-300">{status.gpuInfo}</p>
      </section>

      <section className="grid gap-4 rounded-xl border border-slate-700 bg-slate-900/60 p-4 md:grid-cols-3">
        <label className="flex flex-col gap-1">
          Content image
          <input
            type="file"
            accept="image/*"
            onChange={(event) => void onUpload(event, "content")}
          />
        </label>
        <label className="flex flex-col gap-1">
          Style image
          <input
            type="file"
            accept="image/*"
            onChange={(event) => void onUpload(event, "style")}
          />
        </label>
        <label className="flex flex-col gap-1">
          Content resolution
          <select
            value={controls.contentResolution}
            onChange={(event) =>
              controls.setContentResolution(
                event.target.value as ResolutionPreset,
              )
            }
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
            value={controls.styleResolution}
            onChange={(event) =>
              controls.setStyleResolution(
                event.target.value as ResolutionPreset,
              )
            }
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
            value={controls.selectedPack}
            onChange={(event) => {
              const nextPack = event.target.value;
              if (isVggPackName(nextPack)) {
                controls.setSelectedPack(nextPack);
              }
            }}
          >
            {VGG_PACK_OPTIONS.map((option) => (
              <option key={option.name} value={option.name}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          Chunk steps
          <input
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
        <label className="flex flex-col gap-1">
          Kernel variant
          <select
            value={controls.kernelVariant}
            onChange={(event) => {
              const nextVariant = event.target.value;
              if (isKernelVariantMode(nextVariant)) {
                controls.setKernelVariant(nextVariant);
              }
            }}
          >
            <option value="baseline">Baseline</option>
            <option value="cached-pipelines">Cached pipelines</option>
            <option value="cached-persistent-weights">
              Cached + persistent weights
            </option>
            <option value="cached-persistent-weights-step-pool">
              Cached + persistent weights + step pool
            </option>
            <option value="cached-persistent-weights-pool-scatter">
              Cached + persistent weights + pool scatter
            </option>
            <option value="cached-persistent-weights-step-pool-pool-scatter">
              Cached + persistent weights + step pool + pool scatter
            </option>
            <option value="cached-persistent-weights-step-pool-pool-scatter-vec4-pointwise">
              Cached + persistent weights + step pool + pool scatter + vec4
            </option>
          </select>
        </label>
        {controls.optimizer === "adam" ? (
          <>
            <label className="flex flex-col gap-1">
              Adam β1
              <input
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

      <section className="flex items-center gap-4">
        <button
          className="rounded bg-emerald-500 px-4 py-2 font-semibold text-black"
          onClick={() => setIsRunning(true)}
          disabled={status.isRunning || !canRun}
        >
          ▶ Play
        </button>
        <button
          className="rounded bg-amber-500 px-4 py-2 font-semibold text-black"
          onClick={() => setIsRunning(false)}
          disabled={!status.isRunning}
        >
          ⏸ Pause
        </button>
        <button
          className="rounded bg-zinc-500 px-4 py-2 font-semibold text-black"
          onClick={() => void controls.clearModelCache()}
          disabled={status.isRunning}
        >
          Clear model cache
        </button>
        <button
          className="rounded bg-sky-500 px-4 py-2 font-semibold text-black"
          onClick={controls.resetOptimizerState}
          disabled={status.isRunning}
        >
          Reset optimizer state
        </button>
        <button
          className="rounded bg-rose-500 px-4 py-2 font-semibold text-black"
          onClick={controls.resetOutputImage}
          disabled={images.contentImage === null}
        >
          Reset output image
        </button>
        <p>Iterations: {status.iterations}</p>
        <p>
          Loss: {status.lastLoss === null ? "—" : status.lastLoss.toFixed(4)}
        </p>
      </section>

      <details className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
        <summary className="cursor-pointer font-semibold">View stats</summary>
        {status.runStats === null ? (
          <p className="mt-3 text-slate-400">
            No stats yet. Run at least one chunk.
          </p>
        ) : (
          <div className="mt-3 grid gap-2 text-sm md:grid-cols-2">
            <p>Chunk steps: {status.runStats.steps}</p>
            <p>Kernel variant: {controls.kernelVariant}</p>
            <p>Total: {status.runStats.elapsedMs.toFixed(1)} ms</p>
            <p>Avg step: {status.runStats.avgStepMs.toFixed(1)} ms</p>
            <p>Forward: {status.runStats.forwardMs.toFixed(1)} ms</p>
            <p>Loss: {status.runStats.lossMs.toFixed(1)} ms</p>
            <p>Backward: {status.runStats.backwardMs.toFixed(1)} ms</p>
            <p>Update + clamp: {status.runStats.updateMs.toFixed(1)} ms</p>
          </div>
        )}
      </details>

      <section className="grid gap-4 md:grid-cols-3">
        <img
          className="max-h-80 w-full rounded border border-slate-700 object-contain"
          src={images.contentImage ?? undefined}
          alt="Content"
        />
        <img
          className="max-h-80 w-full rounded border border-slate-700 object-contain"
          src={images.styleImage ?? undefined}
          alt="Style"
        />
        <img
          className="max-h-80 w-full rounded border border-slate-700 object-contain"
          src={images.outputImage ?? undefined}
          alt="Output"
        />
      </section>

      <section className="h-72 overflow-hidden rounded-xl border border-slate-700 bg-black/40">
        <Canvas camera={{ position: [0, 0, 3] }}>
          {images.contentImage !== null ? (
            <Plane url={images.contentImage} x={-2} />
          ) : null}
          {images.styleImage !== null ? (
            <Plane url={images.styleImage} x={0} />
          ) : null}
          {images.outputImage !== null ? (
            <Plane url={images.outputImage} x={2} />
          ) : null}
        </Canvas>
      </section>

      <section className="rounded-xl border border-dashed border-sky-400/50 bg-sky-400/5 p-4 text-sky-200">
        📝 Reserved: add “How we made it” blog post link/teaser here.
      </section>
    </main>
  );
}

export default App;
