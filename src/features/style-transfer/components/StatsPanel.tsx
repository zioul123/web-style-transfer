import type {
  StyleTransferControls,
  StyleTransferStatus,
} from "../types/controller";

type StatsPanelProps = {
  readonly controls: StyleTransferControls;
  readonly status: StyleTransferStatus;
};

export function StatsPanel({ controls, status }: StatsPanelProps) {
  const backendLabel =
    status.lastRunBackend === "fastapi"
      ? "fastapi/pytorch"
      : status.lastRunBackend === "webgpu"
        ? "webgpu"
        : "pending";
  const runConfig =
    status.lastRunBackend === "fastapi"
      ? "fastapi/pytorch"
      : controls.kernelConfigSummary;

  return (
    <details className="rounded-lg border border-white/10 bg-slate-950/55 p-4">
      <summary className="cursor-pointer font-semibold">View stats</summary>
      {status.runStats === null ? (
        <p className="mt-3 text-slate-400">
          No stats yet. Run at least one chunk.
        </p>
      ) : (
        <div className="mt-3 grid gap-2 text-sm md:grid-cols-2">
          <p>Chunk steps: {status.runStats.steps}</p>
          <p>Backend: {backendLabel}</p>
          <p>Run config: {runConfig}</p>
          <p>Total: {status.runStats.elapsedMs.toFixed(1)} ms</p>
          <p>Avg step: {status.runStats.avgStepMs.toFixed(1)} ms</p>
          <p>Forward: {status.runStats.forwardMs.toFixed(1)} ms</p>
          <p>Loss: {status.runStats.lossMs.toFixed(1)} ms</p>
          <p>Backward: {status.runStats.backwardMs.toFixed(1)} ms</p>
          <p>Update + clamp: {status.runStats.updateMs.toFixed(1)} ms</p>
        </div>
      )}
    </details>
  );
}
