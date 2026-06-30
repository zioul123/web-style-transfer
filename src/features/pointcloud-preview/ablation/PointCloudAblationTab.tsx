import { useId, useMemo, useState, type ChangeEvent } from "react";
import {
  buildAblationMatrix,
  chooseDefaultAblationAxes,
  getDefaultFixedAblationValue,
  sortAblationSummariesByDefinitionOrder,
  type AblationMatrixCellFor,
} from "./ablationMatrix";
import {
  formatAblationDimensionValue,
  getAblationDimensionValueKey,
  parseAblationExperimentFilename,
  summarizeAblationDimensions,
  type AblationDimensionKey,
  type AblationDimensionValue,
  type AblationExperimentFile,
  type AblationGridSelection,
  type AblationParseFailure,
} from "./experimentFilenames";

type DirectorySelectedFile = File & {
  readonly webkitRelativePath?: string;
};

type ImportedAblationExperimentFile = AblationExperimentFile & {
  readonly file: File;
  readonly sourceLabel: string;
};

type AblationImportResult = {
  readonly files: readonly ImportedAblationExperimentFile[];
  readonly failures: readonly AblationParseFailure[];
};

type PointCloudAblationTabProps = {
  readonly onPreviewExperimentFile?: (
    file: File,
    sourceLabel: string,
  ) => Promise<boolean>;
};

const emptyParseResult: AblationImportResult = {
  files: [],
  failures: [],
};

const pluralize = (count: number, singular: string, plural: string): string =>
  count === 1 ? singular : plural;

const matrixCellClassName = (
  status: "available" | "ambiguous" | "missing",
): string => {
  switch (status) {
    case "available":
      return "border-emerald-300/25 bg-emerald-400/10 text-emerald-50";
    case "ambiguous":
      return "border-amber-300/25 bg-amber-400/10 text-amber-50";
    case "missing":
      return "border-slate-700/80 bg-slate-950/55 text-slate-500";
    default: {
      const exhaustive: never = status;
      return exhaustive;
    }
  }
};

const selectedFilename = (file: File): string => {
  const relativePath = (file as DirectorySelectedFile).webkitRelativePath;
  return relativePath !== undefined && relativePath.length > 0
    ? relativePath
    : file.name;
};

const parseSelectedExperimentFiles = (
  files: readonly File[],
): AblationImportResult => {
  const parsedFiles: ImportedAblationExperimentFile[] = [];
  const failures: AblationParseFailure[] = [];

  files.forEach((file) => {
    const sourceLabel = selectedFilename(file);
    const result = parseAblationExperimentFilename(sourceLabel);
    if (result.ok) {
      parsedFiles.push({
        ...result.file,
        file,
        sourceLabel,
      });
      return;
    }

    failures.push(result.failure);
  });

  return {
    files: parsedFiles,
    failures,
  };
};

const selectedValueForFixedFilter = (
  value: AblationDimensionValue | undefined,
): AblationDimensionValue | null => value ?? null;

export function PointCloudAblationTab({
  onPreviewExperimentFile,
}: PointCloudAblationTabProps) {
  const folderInputId = useId();
  const fileInputId = useId();
  const [parseResult, setParseResult] =
    useState<AblationImportResult>(emptyParseResult);
  const [selectedXAxis, setSelectedXAxis] =
    useState<AblationDimensionKey | null>(null);
  const [selectedYAxis, setSelectedYAxis] =
    useState<AblationDimensionKey | null>(null);
  const [selectedFixedValueKeys, setSelectedFixedValueKeys] = useState<
    Partial<Record<AblationDimensionKey, string>>
  >({});
  const [previewLoadError, setPreviewLoadError] = useState<string | null>(null);
  const summaries = useMemo(
    () =>
      sortAblationSummariesByDefinitionOrder(
        summarizeAblationDimensions(parseResult.files),
      ),
    [parseResult.files],
  );
  const availableDimensionKeys = useMemo(
    () => new Set(summaries.map((summary) => summary.definition.key)),
    [summaries],
  );
  const axisSelection = useMemo(() => {
    const defaultAxes = chooseDefaultAblationAxes(summaries);
    if (defaultAxes === null) {
      return null;
    }

    const xAxis =
      selectedXAxis !== null && availableDimensionKeys.has(selectedXAxis)
        ? selectedXAxis
        : defaultAxes.xAxis;
    const yAxis =
      selectedYAxis !== null &&
      selectedYAxis !== xAxis &&
      availableDimensionKeys.has(selectedYAxis)
        ? selectedYAxis
        : defaultAxes.yAxis !== xAxis
          ? defaultAxes.yAxis
          : (summaries.find((summary) => summary.definition.key !== xAxis)
              ?.definition.key ?? null);

    return yAxis === null
      ? null
      : {
          xAxis,
          yAxis,
        };
  }, [availableDimensionKeys, selectedXAxis, selectedYAxis, summaries]);
  const matrixSelection = useMemo((): AblationGridSelection | null => {
    if (axisSelection === null) {
      return null;
    }

    const fixedValues = new Map<AblationDimensionKey, AblationDimensionValue>();
    summaries.forEach((summary) => {
      const dimensionKey = summary.definition.key;
      if (
        dimensionKey === axisSelection.xAxis ||
        dimensionKey === axisSelection.yAxis
      ) {
        return;
      }

      const selectedValueKey = selectedFixedValueKeys[dimensionKey];
      const selectedValue = selectedValueForFixedFilter(
        summary.values.find((value) => value.key === selectedValueKey)?.value,
      );
      const fixedValue = selectedValue ?? getDefaultFixedAblationValue(summary);
      if (fixedValue !== null) {
        fixedValues.set(dimensionKey, fixedValue);
      }
    });

    return {
      xAxis: axisSelection.xAxis,
      yAxis: axisSelection.yAxis,
      fixedValues,
    };
  }, [axisSelection, selectedFixedValueKeys, summaries]);
  const matrix = useMemo(
    () =>
      matrixSelection === null
        ? null
        : buildAblationMatrix(parseResult.files, summaries, matrixSelection),
    [matrixSelection, parseResult.files, summaries],
  );
  const selectedCount = parseResult.files.length + parseResult.failures.length;

  const handleFileSelection = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.currentTarget.files ?? []);
    setParseResult(parseSelectedExperimentFiles(files));
    setPreviewLoadError(null);
    event.currentTarget.value = "";
  };

  const handleXAxisChange = (event: ChangeEvent<HTMLSelectElement>) => {
    const nextXAxis = event.currentTarget.value as AblationDimensionKey;
    setSelectedXAxis(nextXAxis);
    setSelectedYAxis((currentYAxis) =>
      currentYAxis === nextXAxis ? null : currentYAxis,
    );
  };

  const handleYAxisChange = (event: ChangeEvent<HTMLSelectElement>) => {
    const nextYAxis = event.currentTarget.value as AblationDimensionKey;
    setSelectedYAxis(nextYAxis);
    setSelectedXAxis((currentXAxis) =>
      currentXAxis === nextYAxis ? null : currentXAxis,
    );
  };

  const handleFixedFilterChange = (
    dimensionKey: AblationDimensionKey,
    valueKey: string,
  ) => {
    setSelectedFixedValueKeys((currentValues) => ({
      ...currentValues,
      [dimensionKey]: valueKey,
    }));
  };

  const previewExperimentFile = async (
    cell: AblationMatrixCellFor<ImportedAblationExperimentFile>,
  ): Promise<void> => {
    if (onPreviewExperimentFile === undefined || cell.files.length !== 1) {
      return;
    }

    const [experimentFile] = cell.files;
    setPreviewLoadError(null);
    try {
      const didLoad = await onPreviewExperimentFile(
        experimentFile.file,
        experimentFile.sourceLabel,
      );
      if (!didLoad) {
        setPreviewLoadError(`Unable to load ${experimentFile.sourceLabel}.`);
      }
    } catch (error) {
      setPreviewLoadError(
        error instanceof Error
          ? error.message
          : `Unable to load ${experimentFile.sourceLabel}.`,
      );
    }
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

      {axisSelection !== null && matrixSelection !== null && matrix !== null ? (
        <section
          className="mt-5 rounded-[0.95rem] border border-white/10 bg-slate-900/75 p-4"
          data-testid="pointcloud-ablation-matrix-browser"
        >
          <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <h3 className="text-lg font-semibold text-white">
                Ablation matrix
              </h3>
              <p className="mt-1 text-sm text-slate-400">
                {parseResult.files.length} parsed{" "}
                {pluralize(parseResult.files.length, "file", "files")}
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <label className="flex min-w-0 flex-col gap-1 text-sm font-medium text-slate-200">
                <span>X axis</span>
                <select
                  className="min-w-48 rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-white"
                  data-testid="pointcloud-ablation-x-axis-select"
                  value={axisSelection.xAxis}
                  onChange={handleXAxisChange}
                >
                  {summaries.map((summary) => (
                    <option
                      key={summary.definition.key}
                      value={summary.definition.key}
                      disabled={summary.definition.key === axisSelection.yAxis}
                    >
                      {summary.definition.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex min-w-0 flex-col gap-1 text-sm font-medium text-slate-200">
                <span>Y axis</span>
                <select
                  className="min-w-48 rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-white"
                  data-testid="pointcloud-ablation-y-axis-select"
                  value={axisSelection.yAxis}
                  onChange={handleYAxisChange}
                >
                  {summaries.map((summary) => (
                    <option
                      key={summary.definition.key}
                      value={summary.definition.key}
                      disabled={summary.definition.key === axisSelection.xAxis}
                    >
                      {summary.definition.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>

          {matrix.fixedSummaries.length > 0 ? (
            <div
              className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4"
              data-testid="pointcloud-ablation-fixed-filters"
            >
              {matrix.fixedSummaries.map((summary) => {
                const fixedValue = matrixSelection.fixedValues.get(
                  summary.definition.key,
                );
                const fixedValueKey =
                  fixedValue === undefined
                    ? ""
                    : getAblationDimensionValueKey(
                        summary.definition,
                        fixedValue,
                      );

                return (
                  <label
                    key={summary.definition.key}
                    className="flex min-w-0 flex-col gap-1 text-sm font-medium text-slate-200"
                  >
                    <span>{summary.definition.label}</span>
                    <select
                      className="min-w-0 rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-white"
                      data-testid={`pointcloud-ablation-fixed-${summary.definition.key}-select`}
                      value={fixedValueKey}
                      onChange={(event) =>
                        handleFixedFilterChange(
                          summary.definition.key,
                          event.currentTarget.value,
                        )
                      }
                    >
                      {summary.values.map((value) => (
                        <option key={value.key} value={value.key}>
                          {value.label}
                        </option>
                      ))}
                    </select>
                  </label>
                );
              })}
            </div>
          ) : null}

          {previewLoadError !== null ? (
            <p
              className="mt-4 rounded-lg border border-rose-300/15 bg-rose-400/10 px-4 py-3 text-sm text-rose-100"
              data-testid="pointcloud-ablation-preview-error"
            >
              {previewLoadError}
            </p>
          ) : null}

          <div className="mt-4 overflow-x-auto">
            <table
              className="min-w-full border-separate border-spacing-2"
              data-testid="pointcloud-ablation-matrix"
            >
              <thead>
                <tr>
                  <th className="w-48 px-3 py-2 text-left text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
                    {matrix.ySummary.definition.label}
                  </th>
                  {matrix.xSummary.values.map((value) => (
                    <th
                      key={value.key}
                      className="min-w-36 rounded-lg border border-white/10 bg-slate-950/65 px-3 py-2 text-center text-sm font-semibold text-white"
                      data-testid={`pointcloud-ablation-column-${matrix.xSummary.definition.key}-${value.key}`}
                    >
                      {formatAblationDimensionValue(
                        matrix.xSummary.definition,
                        value.value,
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {matrix.rows.map((row) => (
                  <tr key={row.yValueKey}>
                    <th
                      className="rounded-lg border border-white/10 bg-slate-950/65 px-3 py-2 text-left text-sm font-semibold text-white"
                      data-testid={`pointcloud-ablation-row-${matrix.ySummary.definition.key}-${row.yValueKey}`}
                      scope="row"
                    >
                      {row.yLabel}
                    </th>
                    {row.cells.map((cell) => (
                      <td
                        key={cell.key}
                        className={`h-20 min-w-36 rounded-lg border p-2 text-center align-middle ${matrixCellClassName(
                          cell.status,
                        )}`}
                        data-status={cell.status}
                        data-testid={`pointcloud-ablation-cell-${cell.key}`}
                      >
                        {cell.status === "available" ? (
                          <button
                            className="h-full w-full rounded-md border border-emerald-200/20 bg-emerald-300/10 px-3 py-2 text-sm font-semibold text-emerald-50 transition hover:bg-emerald-300/20 disabled:cursor-not-allowed disabled:opacity-60"
                            type="button"
                            data-testid={`pointcloud-ablation-cell-preview-${cell.key}`}
                            disabled={onPreviewExperimentFile === undefined}
                            onClick={() => {
                              void previewExperimentFile(cell);
                            }}
                          >
                            Available
                          </button>
                        ) : cell.status === "ambiguous" ? (
                          <span className="block text-sm font-semibold">
                            Ambiguous x{cell.files.length}
                          </span>
                        ) : (
                          <span className="block text-sm font-semibold">
                            Missing
                          </span>
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : summaries.length > 0 ? (
        <p
          className="mt-5 rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-4 py-3 text-sm text-slate-300"
          data-testid="pointcloud-ablation-matrix-unavailable"
        >
          Select at least two parsed dimensions to build the ablation matrix.
        </p>
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
