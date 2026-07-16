import { useEffect, useId, useMemo, useState, type ChangeEvent } from "react";
import type {
  PointCloudAblationGridExportRequest,
  PointCloudAblationGridExportRow,
  PointCloudAblationMultiViewExportRequest,
} from "../usePointCloudScreenshotsController";
import type { SavedViewpoint } from "../pointCloudPreviewModels";
import {
  buildAblationMatrix,
  chooseDefaultAblationAxes,
  getDefaultFixedAblationValue,
  sortAblationSummariesByDefinitionOrder,
  type AblationMatrixCellFor,
} from "./ablationMatrix";
import {
  ablationDimensionDefinitions,
  buildAblationExperimentKey,
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

type SelectableAblationExperiment = {
  readonly key: string;
  readonly file: ImportedAblationExperimentFile;
};

type PointCloudAblationTabProps = {
  readonly savedViewpoints: readonly SavedViewpoint[];
  readonly onPreviewExperimentFile?: (
    file: File,
    sourceLabel: string,
  ) => Promise<boolean>;
  readonly onExportAblationGrid?: (
    request: PointCloudAblationGridExportRequest,
  ) => Promise<void>;
  readonly onExportAblationViewpoints?: (
    request: PointCloudAblationMultiViewExportRequest,
  ) => Promise<void>;
};

const emptyParseResult: AblationImportResult = {
  files: [],
  failures: [],
};

const ABLATION_BROWSER_OPTIONS_STORAGE_KEY =
  "web-style-transfer.pointcloud-ablation-options.v1";

type StoredAblationBrowserOptions = {
  readonly xAxis: AblationDimensionKey | null;
  readonly yAxis: AblationDimensionKey | null;
  readonly fixedValueKeys: Partial<
    Record<AblationDimensionKey, readonly string[]>
  >;
  readonly exportViewpointId: number | null;
  readonly multiViewExperimentKey: string | null;
  readonly multiViewpointIds: readonly number[] | null;
};

const emptyStoredAblationBrowserOptions: StoredAblationBrowserOptions = {
  xAxis: null,
  yAxis: null,
  fixedValueKeys: {},
  exportViewpointId: null,
  multiViewExperimentKey: null,
  multiViewpointIds: null,
};

const ablationDimensionKeySet = new Set<AblationDimensionKey>(
  ablationDimensionDefinitions.map((definition) => definition.key),
);

const pluralize = (count: number, singular: string, plural: string): string =>
  count === 1 ? singular : plural;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isAblationDimensionKey = (
  value: unknown,
): value is AblationDimensionKey =>
  typeof value === "string" &&
  ablationDimensionKeySet.has(value as AblationDimensionKey);

const parseStoredAblationBrowserOptions = (
  value: unknown,
): StoredAblationBrowserOptions => {
  if (!isRecord(value)) {
    return emptyStoredAblationBrowserOptions;
  }

  const fixedValueKeys: Partial<
    Record<AblationDimensionKey, readonly string[]>
  > = {};
  const storedFixedValueKeys = value.fixedValueKeys;
  if (isRecord(storedFixedValueKeys)) {
    Object.entries(storedFixedValueKeys).forEach(
      ([dimensionKey, valueKeys]) => {
        if (!isAblationDimensionKey(dimensionKey)) {
          return;
        }

        if (typeof valueKeys === "string") {
          fixedValueKeys[dimensionKey] = [valueKeys];
          return;
        }

        if (Array.isArray(valueKeys)) {
          const stringValueKeys = valueKeys.filter(
            (valueKey): valueKey is string => typeof valueKey === "string",
          );
          if (stringValueKeys.length > 0) {
            fixedValueKeys[dimensionKey] = stringValueKeys;
          }
        }
      },
    );
  }

  return {
    xAxis: isAblationDimensionKey(value.xAxis) ? value.xAxis : null,
    yAxis: isAblationDimensionKey(value.yAxis) ? value.yAxis : null,
    fixedValueKeys,
    exportViewpointId:
      typeof value.exportViewpointId === "number" &&
      Number.isSafeInteger(value.exportViewpointId)
        ? value.exportViewpointId
        : null,
    multiViewExperimentKey:
      typeof value.multiViewExperimentKey === "string"
        ? value.multiViewExperimentKey
        : null,
    multiViewpointIds: Array.isArray(value.multiViewpointIds)
      ? value.multiViewpointIds.filter(
          (viewpointId): viewpointId is number =>
            typeof viewpointId === "number" &&
            Number.isSafeInteger(viewpointId),
        )
      : null,
  };
};

const readStoredAblationBrowserOptions = (): StoredAblationBrowserOptions => {
  if (typeof window === "undefined") {
    return emptyStoredAblationBrowserOptions;
  }

  const storedOptions = window.localStorage.getItem(
    ABLATION_BROWSER_OPTIONS_STORAGE_KEY,
  );
  if (storedOptions === null) {
    return emptyStoredAblationBrowserOptions;
  }

  try {
    return parseStoredAblationBrowserOptions(JSON.parse(storedOptions));
  } catch {
    return emptyStoredAblationBrowserOptions;
  }
};

const writeStoredAblationBrowserOptions = (
  options: StoredAblationBrowserOptions,
): void => {
  try {
    window.localStorage.setItem(
      ABLATION_BROWSER_OPTIONS_STORAGE_KEY,
      JSON.stringify(options),
    );
  } catch {
    // localStorage may be unavailable in private or restricted browser contexts.
  }
};

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
  savedViewpoints,
  onPreviewExperimentFile,
  onExportAblationGrid,
  onExportAblationViewpoints,
}: PointCloudAblationTabProps) {
  const folderInputId = useId();
  const fileInputId = useId();
  const [initialStoredOptions] = useState(readStoredAblationBrowserOptions);
  const [parseResult, setParseResult] =
    useState<AblationImportResult>(emptyParseResult);
  const [selectedXAxis, setSelectedXAxis] =
    useState<AblationDimensionKey | null>(initialStoredOptions.xAxis);
  const [selectedYAxis, setSelectedYAxis] =
    useState<AblationDimensionKey | null>(initialStoredOptions.yAxis);
  const [selectedFixedValueKeys, setSelectedFixedValueKeys] = useState<
    Partial<Record<AblationDimensionKey, readonly string[]>>
  >(initialStoredOptions.fixedValueKeys);
  const [selectedExportViewpointId, setSelectedExportViewpointId] = useState<
    number | null
  >(initialStoredOptions.exportViewpointId);
  const [selectedMultiViewExperimentKey, setSelectedMultiViewExperimentKey] =
    useState<string | null>(initialStoredOptions.multiViewExperimentKey);
  const [selectedMultiViewpointIds, setSelectedMultiViewpointIds] = useState<
    readonly number[] | null
  >(initialStoredOptions.multiViewpointIds);
  const [previewLoadError, setPreviewLoadError] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState<boolean>(false);
  const [multiViewExportError, setMultiViewExportError] = useState<
    string | null
  >(null);
  const [isExportingMultiView, setIsExportingMultiView] =
    useState<boolean>(false);
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

    const fixedValues = new Map<
      AblationDimensionKey,
      readonly AblationDimensionValue[]
    >();
    summaries.forEach((summary) => {
      const dimensionKey = summary.definition.key;
      if (
        dimensionKey === axisSelection.xAxis ||
        dimensionKey === axisSelection.yAxis
      ) {
        return;
      }

      const selectedValueKeys = selectedFixedValueKeys[dimensionKey] ?? [];
      const selectedValues = selectedValueKeys
        .map((selectedValueKey) =>
          selectedValueForFixedFilter(
            summary.values.find((value) => value.key === selectedValueKey)
              ?.value,
          ),
        )
        .filter((value): value is AblationDimensionValue => value !== null);
      const defaultValue = getDefaultFixedAblationValue(summary);
      const fixedValueSet =
        selectedValues.length > 0
          ? selectedValues
          : defaultValue === null
            ? []
            : [defaultValue];
      if (fixedValueSet.length > 0) {
        fixedValues.set(dimensionKey, fixedValueSet);
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
  const groupedExperiments = useMemo(() => {
    const grouped = new Map<string, ImportedAblationExperimentFile[]>();
    parseResult.files.forEach((file) => {
      const experimentKey = buildAblationExperimentKey(file);
      const matchingFiles = grouped.get(experimentKey) ?? [];
      matchingFiles.push(file);
      grouped.set(experimentKey, matchingFiles);
    });
    return grouped;
  }, [parseResult.files]);
  const selectableExperiments = useMemo(
    (): readonly SelectableAblationExperiment[] =>
      Array.from(groupedExperiments.entries())
        .flatMap(([key, files]) => {
          const [file] = files;
          return files.length === 1 && file !== undefined
            ? [{ key, file }]
            : [];
        })
        .sort((left, right) =>
          left.file.sourceLabel.localeCompare(right.file.sourceLabel),
        ),
    [groupedExperiments],
  );
  const ambiguousExperimentCount = useMemo(
    () =>
      Array.from(groupedExperiments.values()).filter(
        (matchingFiles) => matchingFiles.length > 1,
      ).length,
    [groupedExperiments],
  );
  const ambiguousCellCount = useMemo(
    () =>
      matrix?.rows.reduce(
        (count, row) =>
          count +
          row.cells.filter((cell) => cell.status === "ambiguous").length,
        0,
      ) ?? 0,
    [matrix],
  );
  const selectedExportViewpoint =
    savedViewpoints.find(
      (viewpoint) => viewpoint.id === selectedExportViewpointId,
    ) ??
    savedViewpoints[0] ??
    null;
  const selectedMultiViewExperiment =
    selectableExperiments.find(
      (experiment) => experiment.key === selectedMultiViewExperimentKey,
    ) ??
    selectableExperiments[0] ??
    null;
  const effectiveMultiViewpointIds =
    selectedMultiViewpointIds ??
    savedViewpoints.map((viewpoint) => viewpoint.id);
  const selectedMultiViewpoints = savedViewpoints.filter((viewpoint) =>
    effectiveMultiViewpointIds.includes(viewpoint.id),
  );
  const canExportGrid =
    onExportAblationGrid !== undefined &&
    matrix !== null &&
    selectedExportViewpoint !== null &&
    ambiguousCellCount === 0 &&
    !isExporting;
  const canExportMultiView =
    onExportAblationViewpoints !== undefined &&
    selectedMultiViewExperiment !== null &&
    selectedMultiViewpoints.length > 0 &&
    !isExportingMultiView;

  useEffect(() => {
    writeStoredAblationBrowserOptions({
      xAxis: selectedXAxis,
      yAxis: selectedYAxis,
      fixedValueKeys: selectedFixedValueKeys,
      exportViewpointId: selectedExportViewpointId,
      multiViewExperimentKey: selectedMultiViewExperimentKey,
      multiViewpointIds: selectedMultiViewpointIds,
    });
  }, [
    selectedMultiViewExperimentKey,
    selectedMultiViewpointIds,
    selectedExportViewpointId,
    selectedFixedValueKeys,
    selectedXAxis,
    selectedYAxis,
  ]);

  const handleFileSelection = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.currentTarget.files ?? []);
    setParseResult(parseSelectedExperimentFiles(files));
    setPreviewLoadError(null);
    setExportError(null);
    setMultiViewExportError(null);
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
    valueKeys: readonly string[],
  ) => {
    setSelectedFixedValueKeys((currentValues) => ({
      ...currentValues,
      [dimensionKey]: valueKeys,
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

  const exportAblationGrid = async (): Promise<void> => {
    if (
      onExportAblationGrid === undefined ||
      matrix === null ||
      selectedExportViewpoint === null
    ) {
      return;
    }

    if (ambiguousCellCount > 0) {
      setExportError("Resolve ambiguous matrix cells before exporting.");
      return;
    }

    const rows: PointCloudAblationGridExportRow[] = matrix.rows.map((row) => ({
      yLabel: row.yLabel,
      cells: row.cells.map((cell, columnIndex) => {
        const xLabel = matrix.xSummary.values[columnIndex]?.label ?? "";
        if (cell.status === "missing") {
          return {
            status: "missing",
            xLabel,
            yLabel: row.yLabel,
          };
        }

        const [experimentFile] = cell.files;
        if (experimentFile === undefined || cell.status !== "available") {
          return {
            status: "missing",
            xLabel,
            yLabel: row.yLabel,
          };
        }

        return {
          status: "available",
          xLabel,
          yLabel: row.yLabel,
          file: experimentFile.file,
          sourceLabel: experimentFile.sourceLabel,
        };
      }),
    }));

    setExportError(null);
    setIsExporting(true);
    try {
      await onExportAblationGrid({
        xAxisLabel: matrix.xSummary.definition.label,
        yAxisLabel: matrix.ySummary.definition.label,
        xLabels: matrix.xSummary.values.map((value) => value.label),
        rows,
        viewpoint: selectedExportViewpoint,
      });
    } catch (error) {
      setExportError(
        error instanceof Error
          ? error.message
          : "Unable to export the ablation grid.",
      );
    } finally {
      setIsExporting(false);
    }
  };

  const exportAblationViewpoints = async (): Promise<void> => {
    if (
      onExportAblationViewpoints === undefined ||
      selectedMultiViewExperiment === null ||
      selectedMultiViewpoints.length === 0
    ) {
      return;
    }

    setMultiViewExportError(null);
    setIsExportingMultiView(true);
    try {
      await onExportAblationViewpoints({
        file: selectedMultiViewExperiment.file.file,
        sourceLabel: selectedMultiViewExperiment.file.sourceLabel,
        viewpoints: selectedMultiViewpoints,
      });
    } catch (error) {
      setMultiViewExportError(
        error instanceof Error
          ? error.message
          : "Unable to export the saved viewpoints.",
      );
    } finally {
      setIsExportingMultiView(false);
    }
  };

  const setDirectoryInputRef = (node: HTMLInputElement | null) => {
    if (node !== null) {
      node.setAttribute("webkitdirectory", "");
    }
  };

  return (
    <section
      className="min-h-0 flex-1 overflow-y-auto rounded-[1.1rem] border border-white/10 bg-slate-950/55 p-5 shadow-xl shadow-black/20"
      data-testid="pointcloud-ablation-scroll-region"
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
                const fixedValues = matrixSelection.fixedValues.get(
                  summary.definition.key,
                );
                const fixedValueKeys =
                  fixedValues?.map((fixedValue) =>
                    getAblationDimensionValueKey(
                      summary.definition,
                      fixedValue,
                    ),
                  ) ?? [];
                const isOutputStepFilter =
                  summary.definition.key === "outputStep";

                return (
                  <label
                    key={summary.definition.key}
                    className="flex min-w-0 flex-col gap-1 text-sm font-medium text-slate-200"
                  >
                    <span>{summary.definition.label}</span>
                    <select
                      className="min-w-0 rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-white"
                      data-testid={`pointcloud-ablation-fixed-${summary.definition.key}-select`}
                      multiple={isOutputStepFilter}
                      size={
                        isOutputStepFilter
                          ? Math.min(Math.max(summary.values.length, 2), 6)
                          : undefined
                      }
                      value={
                        isOutputStepFilter
                          ? fixedValueKeys
                          : (fixedValueKeys[0] ?? "")
                      }
                      onChange={(event) =>
                        handleFixedFilterChange(
                          summary.definition.key,
                          isOutputStepFilter
                            ? Array.from(
                                event.currentTarget.selectedOptions,
                                (option) => option.value,
                              )
                            : [event.currentTarget.value],
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

          <div className="mt-4 rounded-lg border border-white/10 bg-slate-950/65 p-4">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
              <label className="flex min-w-0 flex-col gap-1 text-sm font-medium text-slate-200">
                <span>Export viewpoint</span>
                <select
                  className="min-w-56 rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-white disabled:cursor-not-allowed disabled:opacity-60"
                  data-testid="pointcloud-ablation-export-viewpoint-select"
                  value={selectedExportViewpoint?.id ?? ""}
                  disabled={savedViewpoints.length === 0 || isExporting}
                  onChange={(event) =>
                    setSelectedExportViewpointId(
                      event.currentTarget.value.length === 0
                        ? null
                        : Number(event.currentTarget.value),
                    )
                  }
                >
                  {savedViewpoints.length === 0 ? (
                    <option value="">No saved viewpoints</option>
                  ) : null}
                  {savedViewpoints.map((viewpoint) => (
                    <option key={viewpoint.id} value={viewpoint.id}>
                      {viewpoint.label}
                    </option>
                  ))}
                </select>
              </label>
              <button
                className="inline-flex items-center justify-center rounded-lg bg-sky-300 px-4 py-2 text-sm font-semibold text-sky-950 transition hover:bg-sky-200 disabled:cursor-not-allowed disabled:opacity-60"
                type="button"
                data-testid="pointcloud-ablation-export-button"
                disabled={!canExportGrid}
                onClick={() => {
                  void exportAblationGrid();
                }}
              >
                {isExporting ? "Exporting grid..." : "Export grid PNG"}
              </button>
            </div>
            {savedViewpoints.length === 0 ? (
              <p
                className="mt-3 text-sm text-slate-400"
                data-testid="pointcloud-ablation-export-empty"
              >
                Save a viewpoint on the Preview tab before exporting a grid.
              </p>
            ) : ambiguousCellCount > 0 ? (
              <p
                className="mt-3 text-sm text-amber-100"
                data-testid="pointcloud-ablation-export-blocked"
              >
                Resolve {ambiguousCellCount} ambiguous{" "}
                {pluralize(ambiguousCellCount, "cell", "cells")} before export.
              </p>
            ) : null}
            {exportError !== null ? (
              <p
                className="mt-3 rounded-lg border border-rose-300/15 bg-rose-400/10 px-4 py-3 text-sm text-rose-100"
                data-testid="pointcloud-ablation-export-error"
              >
                {exportError}
              </p>
            ) : null}
          </div>

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

      {parseResult.files.length > 0 ? (
        <section
          className="mt-5 rounded-[0.95rem] border border-white/10 bg-slate-900/75 p-4"
          data-testid="pointcloud-ablation-multi-view-export"
        >
          <div>
            <h3 className="text-lg font-semibold text-white">
              One setting, multiple viewpoints
            </h3>
            <p className="mt-1 text-sm text-slate-400">
              Choose one unique experiment setting and export it from your
              selected saved viewpoints as a labelled PNG.
            </p>
          </div>

          <label className="mt-4 flex min-w-0 flex-col gap-1 text-sm font-medium text-slate-200">
            <span>Experiment setting</span>
            <select
              className="min-w-0 rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-white disabled:cursor-not-allowed disabled:opacity-60"
              data-testid="pointcloud-ablation-multi-view-setting-select"
              value={selectedMultiViewExperiment?.key ?? ""}
              disabled={
                selectableExperiments.length === 0 || isExportingMultiView
              }
              onChange={(event) =>
                setSelectedMultiViewExperimentKey(event.currentTarget.value)
              }
            >
              {selectableExperiments.length === 0 ? (
                <option value="">No unique experiment settings</option>
              ) : null}
              {selectableExperiments.map((experiment) => (
                <option key={experiment.key} value={experiment.key}>
                  {experiment.file.sourceLabel}
                </option>
              ))}
            </select>
          </label>

          {ambiguousExperimentCount > 0 ? (
            <p
              className="mt-3 text-sm text-amber-100"
              data-testid="pointcloud-ablation-multi-view-ambiguous"
            >
              {ambiguousExperimentCount} duplicate{" "}
              {pluralize(
                ambiguousExperimentCount,
                "setting is",
                "settings are",
              )}{" "}
              excluded because more than one file matches.
            </p>
          ) : null}

          {savedViewpoints.length > 0 ? (
            <>
              <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
                <span
                  className="text-sm font-semibold text-slate-200"
                  data-testid="pointcloud-ablation-multi-view-selected-count"
                >
                  {selectedMultiViewpoints.length} of {savedViewpoints.length}{" "}
                  selected
                </span>
                <button
                  className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm font-semibold text-slate-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-60"
                  type="button"
                  data-testid="pointcloud-ablation-multi-view-select-all"
                  disabled={isExportingMultiView}
                  onClick={() =>
                    setSelectedMultiViewpointIds(
                      selectedMultiViewpoints.length === savedViewpoints.length
                        ? []
                        : null,
                    )
                  }
                >
                  {selectedMultiViewpoints.length === savedViewpoints.length
                    ? "Clear all"
                    : "Select all"}
                </button>
              </div>
              <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                {savedViewpoints.map((viewpoint) => (
                  <label
                    key={viewpoint.id}
                    className="flex cursor-pointer items-center gap-3 rounded-lg border border-white/10 bg-slate-950/65 px-4 py-3 text-sm text-slate-100"
                  >
                    <input
                      className="accent-amber-300"
                      type="checkbox"
                      data-testid={
                        "pointcloud-ablation-multi-view-viewpoint-" +
                        viewpoint.id
                      }
                      disabled={isExportingMultiView}
                      checked={effectiveMultiViewpointIds.includes(
                        viewpoint.id,
                      )}
                      onChange={(event) => {
                        const isChecked = event.currentTarget.checked;
                        setSelectedMultiViewpointIds((currentIds) => {
                          const concreteIds =
                            currentIds ??
                            savedViewpoints.map(
                              (savedViewpoint) => savedViewpoint.id,
                            );
                          return isChecked
                            ? Array.from(
                                new Set([...concreteIds, viewpoint.id]),
                              )
                            : concreteIds.filter((id) => id !== viewpoint.id);
                        });
                      }}
                    />
                    <span className="min-w-0 flex-1 truncate font-semibold">
                      {viewpoint.label}
                    </span>
                    {viewpoint.swapYZ ? (
                      <span className="text-xs font-semibold uppercase tracking-[0.15em] text-amber-200">
                        Y/Z
                      </span>
                    ) : null}
                  </label>
                ))}
              </div>
            </>
          ) : (
            <p
              className="mt-4 rounded-lg border border-amber-300/20 bg-amber-300/10 px-4 py-3 text-sm text-amber-100"
              data-testid="pointcloud-ablation-multi-view-empty"
            >
              Save at least one viewpoint on the Preview tab before exporting.
            </p>
          )}

          <div className="mt-4 flex justify-end">
            <button
              className="inline-flex items-center justify-center rounded-lg bg-amber-300 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-amber-200 disabled:cursor-not-allowed disabled:opacity-60"
              type="button"
              data-testid="pointcloud-ablation-multi-view-export-button"
              disabled={!canExportMultiView}
              onClick={() => {
                void exportAblationViewpoints();
              }}
            >
              {isExportingMultiView
                ? "Exporting viewpoints..."
                : "Export viewpoint sheet PNG"}
            </button>
          </div>

          {selectedMultiViewpoints.length === 0 &&
          savedViewpoints.length > 0 ? (
            <p
              className="mt-3 text-sm text-amber-100"
              data-testid="pointcloud-ablation-multi-view-selection-empty"
            >
              Select at least one saved viewpoint to export.
            </p>
          ) : null}
          {multiViewExportError !== null ? (
            <p
              className="mt-3 rounded-lg border border-rose-300/15 bg-rose-400/10 px-4 py-3 text-sm text-rose-100"
              data-testid="pointcloud-ablation-multi-view-export-error"
            >
              {multiViewExportError}
            </p>
          ) : null}
        </section>
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
