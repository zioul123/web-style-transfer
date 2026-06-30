import { useId, useMemo, useState, type ChangeEvent } from "react";
import {
  parseAblationExperimentFilenames,
  summarizeAblationDimensions,
  type AblationBatchParseResult,
} from "./experimentFilenames";

type DirectorySelectedFile = File & {
  readonly webkitRelativePath?: string;
};

const emptyParseResult: AblationBatchParseResult = {
  files: [],
  failures: [],
};

const pluralize = (count: number, singular: string, plural: string): string =>
  count === 1 ? singular : plural;

const selectedFilename = (file: File): string => {
  const relativePath = (file as DirectorySelectedFile).webkitRelativePath;
  return relativePath !== undefined && relativePath.length > 0
    ? relativePath
    : file.name;
};

export function PointCloudAblationTab() {
  const folderInputId = useId();
  const fileInputId = useId();
  const [parseResult, setParseResult] =
    useState<AblationBatchParseResult>(emptyParseResult);
  const summaries = useMemo(
    () => summarizeAblationDimensions(parseResult.files),
    [parseResult.files],
  );
  const selectedCount = parseResult.files.length + parseResult.failures.length;

  const handleFileSelection = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.currentTarget.files ?? []);
    setParseResult(
      parseAblationExperimentFilenames(files.map(selectedFilename)),
    );
  };

  const setDirectoryInputRef = (node: HTMLInputElement | null) => {
    if (node !== null) {
      node.setAttribute("webkitdirectory", "");
    }
  };

  return (
    <section
      id="pointcloud-ablation-panel"
      data-testid="pointcloud-ablation-tab-panel"
      role="tabpanel"
      aria-labelledby="pointcloud-ablation-tab"
      className="min-h-0 flex-1 overflow-y-auto rounded-[1.1rem] border border-white/10 bg-slate-950/55 p-5 shadow-xl shadow-black/20"
    >
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
            Ablation status
          </p>
          <h2 className="mt-2 text-2xl font-semibold tracking-tight text-white">
            Experiment filenames
          </h2>
        </div>

        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <label
            className="inline-flex cursor-pointer items-center justify-center rounded-xl bg-amber-300 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-amber-200"
            htmlFor={folderInputId}
          >
            Import experiment folder
          </label>
          <input
            ref={setDirectoryInputRef}
            id={folderInputId}
            className="sr-only"
            type="file"
            multiple
            accept=".json,application/json"
            data-testid="pointcloud-ablation-folder-input"
            onChange={handleFileSelection}
          />
          <label
            className="inline-flex cursor-pointer items-center justify-center rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-slate-100 transition hover:bg-white/10"
            htmlFor={fileInputId}
          >
            Import JSON files
          </label>
          <input
            id={fileInputId}
            className="sr-only"
            type="file"
            multiple
            accept=".json,application/json"
            data-testid="pointcloud-ablation-file-input"
            onChange={handleFileSelection}
          />
        </div>
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-3">
        <div className="rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
            Selected
          </p>
          <output
            className="mt-2 block text-3xl font-semibold text-white"
            data-testid="pointcloud-ablation-selected-count"
          >
            {selectedCount}
          </output>
        </div>
        <div className="rounded-[0.95rem] border border-emerald-300/15 bg-emerald-400/10 px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-200">
            Parsed
          </p>
          <output
            className="mt-2 block text-3xl font-semibold text-white"
            data-testid="pointcloud-ablation-parsed-count"
          >
            {parseResult.files.length}
          </output>
        </div>
        <div className="rounded-[0.95rem] border border-rose-300/15 bg-rose-400/10 px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-rose-200">
            Unparsed
          </p>
          <output
            className="mt-2 block text-3xl font-semibold text-white"
            data-testid="pointcloud-ablation-unparsed-count"
          >
            {parseResult.failures.length}
          </output>
        </div>
      </div>

      {selectedCount === 0 ? (
        <p
          className="mt-5 rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-4 py-3 text-sm text-slate-300"
          data-testid="pointcloud-ablation-empty"
        >
          No experiment filenames selected.
        </p>
      ) : null}

      {selectedCount > 0 && summaries.length === 0 ? (
        <p
          className="mt-5 rounded-[0.95rem] border border-rose-300/15 bg-rose-400/10 px-4 py-3 text-sm text-rose-100"
          data-testid="pointcloud-ablation-no-summaries"
        >
          No valid ablation filenames parsed.
        </p>
      ) : null}

      {summaries.length > 0 ? (
        <div
          className="mt-5 grid gap-3 xl:grid-cols-2"
          data-testid="pointcloud-ablation-dimension-summaries"
        >
          {summaries.map((summary) => (
            <section
              key={summary.definition.key}
              className="rounded-[0.95rem] border border-white/10 bg-slate-900/75 p-4"
              data-testid={`pointcloud-ablation-summary-${summary.definition.key}`}
            >
              <div className="flex flex-wrap items-baseline justify-between gap-3">
                <h3 className="text-sm font-semibold text-white">
                  {summary.definition.label}
                </h3>
                <p className="text-xs font-medium uppercase tracking-[0.16em] text-slate-400">
                  {summary.values.length}{" "}
                  {pluralize(summary.values.length, "value", "values")}
                </p>
              </div>
              <ul className="mt-3 flex flex-wrap gap-2">
                {summary.values.map((value) => (
                  <li
                    key={value.key}
                    className="rounded-lg border border-white/10 bg-slate-950/65 px-3 py-2 text-sm text-slate-100"
                    data-testid={`pointcloud-ablation-summary-${summary.definition.key}-value-${value.key}`}
                  >
                    <span className="font-semibold">{value.label}</span>
                    <span className="ml-2 text-slate-400">x{value.count}</span>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      ) : null}

      {parseResult.failures.length > 0 ? (
        <details
          className="mt-5 rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-4 py-3 text-sm"
          data-testid="pointcloud-ablation-failures"
        >
          <summary className="cursor-pointer font-semibold text-rose-100">
            {parseResult.failures.length}{" "}
            {pluralize(parseResult.failures.length, "filename", "filenames")}{" "}
            unparsed
          </summary>
          <ul className="mt-3 space-y-2 text-slate-300">
            {parseResult.failures.slice(0, 8).map((failure) => (
              <li key={failure.filename} className="break-all">
                <span className="font-semibold text-slate-100">
                  {failure.filename}
                </span>
                <span className="block text-rose-100">{failure.reason}</span>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </section>
  );
}
