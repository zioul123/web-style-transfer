import type { Dispatch, SetStateAction } from "react";

type OptionsTogglePanelProps = {
  readonly benchmarkUrl: string;
  readonly showOptions: boolean;
  readonly setShowOptions: Dispatch<SetStateAction<boolean>>;
};

export function OptionsTogglePanel({
  benchmarkUrl,
  showOptions,
  setShowOptions,
}: OptionsTogglePanelProps) {
  return (
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
  );
}
