import {
  ablationDimensionDefinitions,
  buildAblationMatrixCellKey,
  compareAblationDimensionValues,
  getAblationDimensionValue,
  type AblationDimensionKey,
  type AblationDimensionSummary,
  type AblationDimensionValue,
  type AblationExperimentFile,
  type AblationGridSelection,
  type AblationMatrixCell,
} from "./experimentFilenames";

export type AblationAxisSelection = {
  readonly xAxis: AblationDimensionKey;
  readonly yAxis: AblationDimensionKey;
};

export type AblationMatrixCellFor<TFile extends AblationExperimentFile> = Omit<
  AblationMatrixCell,
  "files"
> & {
  readonly files: readonly TFile[];
};

export type AblationMatrixRow<TFile extends AblationExperimentFile> = {
  readonly yValue: AblationDimensionValue;
  readonly yValueKey: string;
  readonly yLabel: string;
  readonly cells: readonly AblationMatrixCellFor<TFile>[];
};

export type AblationMatrixModel<TFile extends AblationExperimentFile> = {
  readonly xSummary: AblationDimensionSummary;
  readonly ySummary: AblationDimensionSummary;
  readonly fixedSummaries: readonly AblationDimensionSummary[];
  readonly rows: readonly AblationMatrixRow<TFile>[];
};

const dimensionOrder = new Map(
  ablationDimensionDefinitions.map((definition, index) => [
    definition.key,
    index,
  ]),
);

const summaryForKey = (
  summaries: readonly AblationDimensionSummary[],
  key: AblationDimensionKey,
): AblationDimensionSummary | null =>
  summaries.find((summary) => summary.definition.key === key) ?? null;

const firstSummaryExcept = (
  summaries: readonly AblationDimensionSummary[],
  excludedKey: AblationDimensionKey,
): AblationDimensionSummary | null =>
  summaries.find((summary) => summary.definition.key !== excludedKey) ?? null;

export const chooseDefaultAblationAxes = (
  summaries: readonly AblationDimensionSummary[],
): AblationAxisSelection | null => {
  const varyingSummaries = summaries.filter(
    (summary) => summary.values.length > 1,
  );
  const contentSamplesSummary = varyingSummaries.find(
    (summary) => summary.definition.key === "contentSamplesPerFace",
  );
  const distanceSummary = varyingSummaries.find(
    (summary) => summary.definition.key === "distanceMeasure",
  );

  let xSummary: AblationDimensionSummary | null = contentSamplesSummary ?? null;
  let ySummary: AblationDimensionSummary | null = distanceSummary ?? null;

  if (xSummary === null && ySummary !== null) {
    xSummary =
      varyingSummaries.find(
        (summary) => summary.definition.key !== ySummary?.definition.key,
      ) ?? firstSummaryExcept(summaries, ySummary.definition.key);
  }

  if (xSummary === null) {
    xSummary = varyingSummaries[0] ?? summaries[0] ?? null;
  }

  if (xSummary === null) {
    return null;
  }

  if (
    ySummary === null ||
    ySummary.definition.key === xSummary.definition.key
  ) {
    ySummary =
      varyingSummaries.find(
        (summary) => summary.definition.key !== xSummary?.definition.key,
      ) ?? firstSummaryExcept(summaries, xSummary.definition.key);
  }

  if (ySummary === null) {
    return null;
  }

  return {
    xAxis: xSummary.definition.key,
    yAxis: ySummary.definition.key,
  };
};

export const sortAblationSummariesByDefinitionOrder = (
  summaries: readonly AblationDimensionSummary[],
): readonly AblationDimensionSummary[] =>
  [...summaries].sort((left, right) => {
    const leftOrder = dimensionOrder.get(left.definition.key) ?? 0;
    const rightOrder = dimensionOrder.get(right.definition.key) ?? 0;
    return leftOrder - rightOrder;
  });

export const getDefaultFixedAblationValue = (
  summary: AblationDimensionSummary,
): AblationDimensionValue | null => {
  if (summary.values.length === 0) {
    return null;
  }

  if (summary.definition.key !== "outputStep") {
    return summary.values[0].value;
  }

  return summary.values.reduce((highest, value) =>
    compareAblationDimensionValues(
      summary.definition,
      highest.value,
      value.value,
    ) < 0
      ? value
      : highest,
  ).value;
};

export const buildAblationMatrix = <TFile extends AblationExperimentFile>(
  files: readonly TFile[],
  summaries: readonly AblationDimensionSummary[],
  selection: AblationGridSelection,
): AblationMatrixModel<TFile> | null => {
  const xSummary = summaryForKey(summaries, selection.xAxis);
  const ySummary = summaryForKey(summaries, selection.yAxis);
  if (xSummary === null || ySummary === null) {
    return null;
  }

  const fixedSummaries = summaries.filter(
    (summary) =>
      summary.definition.key !== selection.xAxis &&
      summary.definition.key !== selection.yAxis,
  );

  return {
    xSummary,
    ySummary,
    fixedSummaries,
    rows: ySummary.values.map((yValue) => ({
      yValue: yValue.value,
      yValueKey: yValue.key,
      yLabel: yValue.label,
      cells: xSummary.values.map((xValue) => {
        const matchingFiles = files.filter((file) => {
          const xFileValue = getAblationDimensionValue(
            file.config,
            selection.xAxis,
          );
          const yFileValue = getAblationDimensionValue(
            file.config,
            selection.yAxis,
          );
          if (xFileValue !== xValue.value || yFileValue !== yValue.value) {
            return false;
          }

          for (const [dimensionKey, fixedValues] of selection.fixedValues) {
            const dimensionValue = getAblationDimensionValue(
              file.config,
              dimensionKey,
            );
            if (
              dimensionValue === null ||
              !fixedValues.includes(dimensionValue)
            ) {
              return false;
            }
          }

          return true;
        });

        return {
          xValue: xValue.value,
          yValue: yValue.value,
          key: buildAblationMatrixCellKey(
            xSummary.definition,
            xValue.value,
            ySummary.definition,
            yValue.value,
          ),
          status:
            matchingFiles.length === 0
              ? "missing"
              : matchingFiles.length === 1
                ? "available"
                : "ambiguous",
          files: matchingFiles,
        };
      }),
    })),
  };
};
