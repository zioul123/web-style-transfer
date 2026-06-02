import type { Dispatch, SetStateAction } from "react";
import type {
  StyleTransferControls,
  StyleTransferImages,
  StyleTransferStatus,
} from "../types/controller";

type RunControlsPanelProps = {
  readonly canRun: boolean;
  readonly controls: StyleTransferControls;
  readonly images: StyleTransferImages;
  readonly setIsRunning: Dispatch<SetStateAction<boolean>>;
  readonly status: StyleTransferStatus;
};

export function RunControlsPanel({
  canRun,
  controls,
  images,
  setIsRunning,
  status,
}: RunControlsPanelProps) {
  return (
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
  );
}
