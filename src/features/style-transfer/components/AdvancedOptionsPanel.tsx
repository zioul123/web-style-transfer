import type {
  StyleTransferControls,
  StyleTransferStatus,
  UseStyleTransferControllerResult,
} from "../types/controller";
import {
  formatBytes,
  isKernelConvBackwardInput,
  isKernelConvForward,
  isKernelGramKernel,
  isKernelStyleBackward,
  isKernelWeightStorage,
  isOptimizerMode,
  isResolutionPreset,
  isVggPackName,
  kernelConvBackwardInputLabel,
  kernelConvBackwardInputOptions,
  kernelConvForwardLabel,
  kernelConvForwardOptions,
  kernelGramKernelOptions,
  kernelStyleBackwardOptions,
  kernelWeightStorageLabel,
  kernelWeightStorageOptions,
  resolutionOptions,
  type VggPackOption,
} from "../uiOptions";

type AdvancedOptionsPanelProps = {
  readonly controls: StyleTransferControls;
  readonly modelPackOptions: readonly VggPackOption[];
  readonly onUpload: UseStyleTransferControllerResult["onUpload"];
  readonly status: StyleTransferStatus;
};

export function AdvancedOptionsPanel({
  controls,
  modelPackOptions,
  onUpload,
  status,
}: AdvancedOptionsPanelProps) {
  return (
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
      <div className="flex flex-col gap-3 rounded-lg border border-white/10 bg-slate-900/70 p-3 md:col-span-3">
        <p className="text-sm font-semibold text-slate-200">Kernel Options</p>
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
                controls.setUsePersistentWeightBuffers(event.target.checked)
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
                  controls.setConvBackwardInputKernel(nextConvBackwardInput);
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
      <div className="flex flex-col gap-3 rounded-lg border border-white/10 bg-slate-900/70 p-3 md:col-span-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-sm font-semibold text-slate-200">Model Cache</p>
            <p className="text-xs text-slate-400">
              {status.modelCachePackStatuses.length} cached pack
              {status.modelCachePackStatuses.length === 1 ? "" : "s"} ·{" "}
              {formatBytes(status.modelCacheBytes)}
            </p>
          </div>
          <button
            className="rounded-lg bg-zinc-300 px-4 py-2 font-semibold text-zinc-950 transition hover:bg-zinc-200 disabled:cursor-not-allowed disabled:opacity-50"
            type="button"
            onClick={() => void controls.clearModelCache()}
            disabled={status.isRunning}
          >
            Clear model cache
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-96 text-left text-sm">
            <thead className="text-xs uppercase text-slate-400">
              <tr>
                <th className="border-b border-white/10 py-2 pr-4">Pack</th>
                <th className="border-b border-white/10 py-2 pr-4">Size</th>
                <th className="border-b border-white/10 py-2">Cached</th>
              </tr>
            </thead>
            <tbody>
              {status.modelCachePackStatuses.length === 0 ? (
                <tr>
                  <td className="py-3 text-slate-400" colSpan={3}>
                    No model packs cached yet.
                  </td>
                </tr>
              ) : (
                status.modelCachePackStatuses.map((pack) => (
                  <tr key={`${pack.tier}-${pack.updatedAtMs}`}>
                    <td className="border-b border-white/5 py-2 pr-4 font-medium text-slate-200">
                      {pack.tier}
                    </td>
                    <td className="border-b border-white/5 py-2 pr-4 text-slate-300">
                      {formatBytes(pack.bytes)}
                    </td>
                    <td className="border-b border-white/5 py-2 text-slate-300">
                      {new Date(pack.updatedAtMs).toLocaleDateString()}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
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
  );
}
