import { Canvas, useLoader } from "@react-three/fiber";
import type { ReactElement } from "react";
import { TextureLoader } from "three";
import { useStyleTransferController } from "./features/style-transfer/hooks/useStyleTransferController";
import type {
  OptimizerMode,
  ResolutionPreset,
} from "./features/style-transfer/types/controller";

const Plane = ({ url, x }: { url: string; x: number }): ReactElement => {
  const texture = useLoader(TextureLoader, url);
  return (
    <mesh position={[x, 0, 0]}>
      <planeGeometry args={[1.75, 1.75]} />
      <meshBasicMaterial map={texture} />
    </mesh>
  );
};

function App() {
  const { controls, status, images, canRun, setIsRunning, onUpload } =
    useStyleTransferController();

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-6 px-6 py-8 text-slate-100">
      <h1 className="text-3xl font-semibold">
        WebGPU Style Transfer — Phase 6 UI
      </h1>

      <section className="grid gap-4 rounded-xl border border-slate-700 bg-slate-900/60 p-4 md:grid-cols-2">
        <p>{status.workerStatus}</p>
        <p>{status.gpuStatus}</p>
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
          Resolution
          <select
            value={controls.resolution}
            onChange={(event) =>
              controls.setResolution(
                Number(event.target.value) as ResolutionPreset,
              )
            }
          >
            <option value={128}>128</option>
            <option value={256}>256</option>
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
            onChange={(event) =>
              controls.setOptimizer(event.target.value as OptimizerMode)
            }
          >
            <option value="sgd">SGD</option>
            <option value="adam">Adam</option>
            <option value="lbfgs">L-BFGS</option>
          </select>
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={controls.fusedOps}
            onChange={(event) => controls.setFusedOps(event.target.checked)}
          />
          <span>Use fused conv+relu</span>
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={controls.superFusedOps}
            onChange={(event) =>
              controls.setSuperFusedOps(event.target.checked)
            }
          />
          <span>Use super fused blocks</span>
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
              L-BFGS memory
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
              L-BFGS ε
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
            <p>Total: {status.runStats.elapsedMs.toFixed(1)} ms</p>
            <p>Avg step: {status.runStats.avgStepMs.toFixed(1)} ms</p>
            <p>Forward: {status.runStats.forwardMs.toFixed(1)} ms</p>
            <p>Loss: {status.runStats.lossMs.toFixed(1)} ms</p>
            <p>Backward: {status.runStats.backwardMs.toFixed(1)} ms</p>
            <p>Update: {status.runStats.updateMs.toFixed(1)} ms</p>
            <p>Clamp (GPU): {status.runStats.clampMs.toFixed(1)} ms</p>
          </div>
        )}
      </details>

      <section className="grid gap-4 md:grid-cols-3">
        <img
          className="aspect-square w-full rounded border border-slate-700 object-cover"
          src={images.contentImage ?? undefined}
          alt="Content"
        />
        <img
          className="aspect-square w-full rounded border border-slate-700 object-cover"
          src={images.styleImage ?? undefined}
          alt="Style"
        />
        <img
          className="aspect-square w-full rounded border border-slate-700 object-cover"
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
