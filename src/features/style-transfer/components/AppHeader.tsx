import type { StyleTransferStatus } from "../types/controller";

type AppHeaderProps = {
  readonly status: StyleTransferStatus;
};

export function AppHeader({ status }: AppHeaderProps) {
  return (
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
  );
}
