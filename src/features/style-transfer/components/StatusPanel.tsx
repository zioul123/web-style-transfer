import type { StyleTransferStatus } from "../types/controller";

type StatusPanelProps = {
  readonly status: StyleTransferStatus;
};

export function StatusPanel({ status }: StatusPanelProps) {
  const cacheSizeMb = (status.modelCacheBytes / (1024 * 1024)).toFixed(2);

  return (
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
  );
}
