export type AblationTvMode = "L1" | "L2";
export type AblationFrameMode = "PCP" | "SPL-S" | "BST-S" | "SIMPLE_AXIS";
export type AblationInputColorMode = "RAW_COLORS" | "LOGIT";
export type AblationPoolMode = "MAX" | "AVG";
export type AblationDistanceMeasure = "EUCLIDEAN" | "SPECTRAL";
export type AblationStyleWeight = number | readonly number[];

export type AblationDimensionValue = number | string;
export type AblationDimensionValueType = "number" | "string";

export type AblationExperimentConfig = {
  readonly styleWeight: AblationStyleWeight;
  readonly contentWeight: number;
  readonly tvWeight: number;
  readonly tvMode: AblationTvMode;
  readonly contentSamplesPerFace: number;
  readonly styleResolution: string | null;
  readonly styleSamplesPerFace: number | null;
  readonly frameMode: AblationFrameMode;
  readonly colorMode: AblationInputColorMode;
  readonly poolMode: AblationPoolMode;
  readonly distanceMeasure: AblationDistanceMeasure;
  readonly knn: number;
  readonly radiusFactor: number;
  readonly geodesicRatio: number;
  readonly maxTriangles: number | null;
  readonly attenuationLabel: string;
  readonly optimizationSteps: number;
  readonly outputStep: number | null;
};

export type AblationExperimentFile = {
  readonly filename: string;
  readonly config: AblationExperimentConfig;
};

export type AblationParseFailure = {
  readonly filename: string;
  readonly reason: string;
};

export type AblationFilenameParseResult =
  | {
      readonly ok: true;
      readonly file: AblationExperimentFile;
    }
  | {
      readonly ok: false;
      readonly failure: AblationParseFailure;
    };

export type AblationBatchParseResult = {
  readonly files: readonly AblationExperimentFile[];
  readonly failures: readonly AblationParseFailure[];
};

export type AblationDimensionKey =
  | "styleWeight"
  | "contentWeight"
  | "tvWeight"
  | "tvMode"
  | "contentSamplesPerFace"
  | "styleResolution"
  | "styleSamplesPerFace"
  | "frameMode"
  | "colorMode"
  | "poolMode"
  | "distanceMeasure"
  | "knn"
  | "radiusFactor"
  | "geodesicRatio"
  | "maxTriangles"
  | "attenuationLabel"
  | "optimizationSteps"
  | "outputStep";

export type AblationDimensionDefinition = {
  readonly key: AblationDimensionKey;
  readonly label: string;
  readonly valueType: AblationDimensionValueType;
  readonly order?: readonly AblationDimensionValue[];
  readonly formatValue?: (value: AblationDimensionValue) => string;
};

export type AblationDimensionSummaryValue = {
  readonly value: AblationDimensionValue;
  readonly label: string;
  readonly key: string;
  readonly count: number;
};

export type AblationDimensionSummary = {
  readonly definition: AblationDimensionDefinition;
  readonly values: readonly AblationDimensionSummaryValue[];
};

export type AblationGridSelection = {
  readonly xAxis: AblationDimensionKey;
  readonly yAxis: AblationDimensionKey;
  readonly fixedValues: ReadonlyMap<
    AblationDimensionKey,
    readonly AblationDimensionValue[]
  >;
};

export type AblationMatrixCellStatus = "available" | "missing" | "ambiguous";

export type AblationMatrixCell = {
  readonly xValue: AblationDimensionValue;
  readonly yValue: AblationDimensionValue;
  readonly key: string;
  readonly status: AblationMatrixCellStatus;
  readonly files: readonly AblationExperimentFile[];
};

const numberPattern = String.raw`-?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)`;
const integerPattern = String.raw`[0-9]+`;
const styleWeightPattern = String.raw`${numberPattern}(?:-${numberPattern})*`;
const styleSourcePattern = String.raw`(?:(?:${numberPattern})x(?:${numberPattern})s-img|(?:${numberPattern})s-spf)`;

const frameModeTokens = ["PCP", "SPL-S", "BST-S", "SIMPLE_AXIS"] as const;
const inputColorModeTokens = ["RAW_COLORS", "LOGIT"] as const;
const poolModeTokens = ["MAX", "AVG"] as const;
const distanceMeasureTokens = ["EUCLIDEAN", "SPECTRAL"] as const;
const legacyDistanceMeasureTokens = ["BIHARMONIC"] as const;

const tokenPattern = (tokens: readonly string[]): string =>
  tokens.map((token) => token.replaceAll("-", String.raw`\-`)).join("|");

const frameModePattern = tokenPattern(frameModeTokens);
const inputColorModePattern = tokenPattern(inputColorModeTokens);
const poolModePattern = tokenPattern(poolModeTokens);
const distanceMeasurePattern = tokenPattern([
  ...distanceMeasureTokens,
  ...legacyDistanceMeasureTokens,
]);

const experimentFilenamePattern = new RegExp(
  `^(${styleWeightPattern})sw_` +
    `(${numberPattern})cw_` +
    `(${numberPattern})tv_` +
    `(L1|L2)_` +
    `(${numberPattern})c-spf_` +
    `(?:(${styleSourcePattern})_)?` +
    `(${frameModePattern})_` +
    `(${inputColorModePattern})_` +
    `(?:(${poolModePattern})_)?` +
    `(${distanceMeasurePattern})_` +
    `(${integerPattern})knn_` +
    `(${numberPattern})rf_` +
    `(${numberPattern})gr_` +
    `(?:(${integerPattern})tris_)?` +
    `([^_]+)-attn_` +
    `(${integerPattern})steps` +
    `(?:_step(${integerPattern}))?` +
    String.raw`\.json$`,
);

const tvModes = ["L1", "L2"] as const satisfies readonly AblationTvMode[];
const frameModes = frameModeTokens satisfies readonly AblationFrameMode[];
const inputColorModes =
  inputColorModeTokens satisfies readonly AblationInputColorMode[];
const poolModes = poolModeTokens satisfies readonly AblationPoolMode[];
const distanceMeasures =
  distanceMeasureTokens satisfies readonly AblationDistanceMeasure[];

export const ablationDimensionDefinitions: readonly AblationDimensionDefinition[] =
  [
    { key: "styleWeight", label: "Style weight", valueType: "string" },
    { key: "contentWeight", label: "Content weight", valueType: "number" },
    { key: "tvWeight", label: "TV weight", valueType: "number" },
    { key: "tvMode", label: "TV mode", valueType: "string", order: tvModes },
    {
      key: "contentSamplesPerFace",
      label: "Content samples/face",
      valueType: "number",
    },
    {
      key: "styleResolution",
      label: "Style resolution",
      valueType: "string",
    },
    {
      key: "styleSamplesPerFace",
      label: "Style samples/face",
      valueType: "number",
    },
    {
      key: "frameMode",
      label: "Frame mode",
      valueType: "string",
      order: frameModes,
    },
    {
      key: "colorMode",
      label: "Color mode",
      valueType: "string",
      order: inputColorModes,
    },
    {
      key: "poolMode",
      label: "Pool mode",
      valueType: "string",
      order: poolModes,
    },
    {
      key: "distanceMeasure",
      label: "Distance",
      valueType: "string",
      order: distanceMeasures,
    },
    { key: "knn", label: "KNN", valueType: "number" },
    { key: "radiusFactor", label: "Radius factor", valueType: "number" },
    { key: "geodesicRatio", label: "Geodesic ratio", valueType: "number" },
    { key: "maxTriangles", label: "Max triangles", valueType: "number" },
    { key: "attenuationLabel", label: "Attenuation", valueType: "string" },
    {
      key: "optimizationSteps",
      label: "Optimization steps",
      valueType: "number",
    },
    {
      key: "outputStep",
      label: "Output step",
      valueType: "number",
      formatValue: (value) => `step${value}`,
    },
  ];

export const ablationDimensionDefinitionByKey: ReadonlyMap<
  AblationDimensionKey,
  AblationDimensionDefinition
> = new Map(
  ablationDimensionDefinitions.map((definition) => [
    definition.key,
    definition,
  ]),
);

const parseNumber = (value: string, label: string): number => {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${label} must be a finite number.`);
  }
  return parsed;
};

const parseInteger = (value: string, label: string): number => {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) {
    throw new Error(`${label} must be a safe integer.`);
  }
  return parsed;
};

const parseStyleWeight = (value: string): AblationStyleWeight => {
  const values = value
    .split("-")
    .map((entry) => parseNumber(entry, "Style weight"));
  return values.length === 1 ? values[0] : values;
};

const formatStyleWeight = (value: AblationStyleWeight): string =>
  Array.isArray(value) ? value.map(String).join("-") : String(value);

const parseStyleSource = (
  value: string | undefined,
): Pick<
  AblationExperimentConfig,
  "styleResolution" | "styleSamplesPerFace"
> => {
  if (value === undefined) {
    return {
      styleResolution: null,
      styleSamplesPerFace: null,
    };
  }

  const resolutionMatch = new RegExp(
    `^(${numberPattern})x(${numberPattern})s-img$`,
  ).exec(value);
  if (resolutionMatch !== null) {
    const width = parseNumber(resolutionMatch[1], "Style image width");
    const height = parseNumber(resolutionMatch[2], "Style image height");
    return {
      styleResolution: `${width}x${height}`,
      styleSamplesPerFace: null,
    };
  }

  const samplesMatch = new RegExp(`^(${numberPattern})s-spf$`).exec(value);
  if (samplesMatch !== null) {
    return {
      styleResolution: null,
      styleSamplesPerFace: parseNumber(
        samplesMatch[1],
        "Style samples per face",
      ),
    };
  }

  throw new Error(`Unknown style source token: ${value}.`);
};

const normalizeDistanceMeasure = (value: string): AblationDistanceMeasure =>
  value === "BIHARMONIC" ? "SPECTRAL" : (value as AblationDistanceMeasure);

export const parseAblationExperimentFilename = (
  filename: string,
): AblationFilenameParseResult => {
  const basename = filename.split(/[\\/]/).at(-1) ?? filename;
  const match = experimentFilenamePattern.exec(basename);
  if (match === null) {
    return {
      ok: false,
      failure: {
        filename,
        reason:
          "Filename does not match the expected point-cloud ablation experiment pattern.",
      },
    };
  }

  try {
    const maxTriangles = match[14] ?? null;
    const outputStep = match[17] ?? null;
    return {
      ok: true,
      file: {
        filename,
        config: {
          styleWeight: parseStyleWeight(match[1]),
          contentWeight: parseNumber(match[2], "Content weight"),
          tvWeight: parseNumber(match[3], "TV weight"),
          tvMode: match[4] as AblationTvMode,
          contentSamplesPerFace: parseNumber(
            match[5],
            "Content samples per face",
          ),
          ...parseStyleSource(match[6]),
          frameMode: match[7] as AblationFrameMode,
          colorMode: match[8] as AblationInputColorMode,
          poolMode: (match[9] ?? "MAX") as AblationPoolMode,
          distanceMeasure: normalizeDistanceMeasure(match[10]),
          knn: parseInteger(match[11], "KNN"),
          radiusFactor: parseNumber(match[12], "Radius factor"),
          geodesicRatio: parseNumber(match[13], "Geodesic ratio"),
          maxTriangles:
            maxTriangles === null
              ? null
              : parseInteger(maxTriangles, "Maximum triangles"),
          attenuationLabel: match[15],
          optimizationSteps: parseInteger(match[16], "Optimization steps"),
          outputStep:
            outputStep === null
              ? null
              : parseInteger(outputStep, "Output step"),
        },
      },
    };
  } catch (error) {
    return {
      ok: false,
      failure: {
        filename,
        reason:
          error instanceof Error
            ? error.message
            : "Filename matched but could not be converted to typed values.",
      },
    };
  }
};

export const parseAblationExperimentFilenames = (
  filenames: readonly string[],
): AblationBatchParseResult => {
  const files: AblationExperimentFile[] = [];
  const failures: AblationParseFailure[] = [];

  filenames.forEach((filename) => {
    const result = parseAblationExperimentFilename(filename);
    if (result.ok) {
      files.push(result.file);
    } else {
      failures.push(result.failure);
    }
  });

  return { files, failures };
};

export const getAblationDimensionValue = (
  config: AblationExperimentConfig,
  key: AblationDimensionKey,
): AblationDimensionValue | null => {
  switch (key) {
    case "styleWeight":
      return formatStyleWeight(config.styleWeight);
    case "contentWeight":
      return config.contentWeight;
    case "tvWeight":
      return config.tvWeight;
    case "tvMode":
      return config.tvMode;
    case "contentSamplesPerFace":
      return config.contentSamplesPerFace;
    case "styleResolution":
      return config.styleResolution;
    case "styleSamplesPerFace":
      return config.styleSamplesPerFace;
    case "frameMode":
      return config.frameMode;
    case "colorMode":
      return config.colorMode;
    case "poolMode":
      return config.poolMode;
    case "distanceMeasure":
      return config.distanceMeasure;
    case "knn":
      return config.knn;
    case "radiusFactor":
      return config.radiusFactor;
    case "geodesicRatio":
      return config.geodesicRatio;
    case "maxTriangles":
      return config.maxTriangles;
    case "attenuationLabel":
      return config.attenuationLabel;
    case "optimizationSteps":
      return config.optimizationSteps;
    case "outputStep":
      return config.outputStep;
    default: {
      const exhaustive: never = key;
      return exhaustive;
    }
  }
};

export const formatAblationDimensionValue = (
  definition: AblationDimensionDefinition,
  value: AblationDimensionValue,
): string => definition.formatValue?.(value) ?? String(value);

export const getAblationDimensionValueKey = (
  definition: AblationDimensionDefinition,
  value: AblationDimensionValue,
): string => `${definition.valueType}:${encodeURIComponent(String(value))}`;

export const compareAblationDimensionValues = (
  definition: AblationDimensionDefinition,
  left: AblationDimensionValue,
  right: AblationDimensionValue,
): number => {
  if (definition.order !== undefined) {
    const leftIndex = definition.order.indexOf(left);
    const rightIndex = definition.order.indexOf(right);
    if (leftIndex !== -1 || rightIndex !== -1) {
      if (leftIndex === -1) {
        return 1;
      }
      if (rightIndex === -1) {
        return -1;
      }
      return leftIndex - rightIndex;
    }
  }

  if (
    definition.valueType === "number" &&
    typeof left === "number" &&
    typeof right === "number"
  ) {
    return left - right;
  }

  return String(left).localeCompare(String(right), "en", {
    numeric: true,
    sensitivity: "base",
  });
};

export const buildAblationMatrixCellKey = (
  xDefinition: AblationDimensionDefinition,
  xValue: AblationDimensionValue,
  yDefinition: AblationDimensionDefinition,
  yValue: AblationDimensionValue,
): string =>
  `${xDefinition.key}=${getAblationDimensionValueKey(
    xDefinition,
    xValue,
  )}|${yDefinition.key}=${getAblationDimensionValueKey(yDefinition, yValue)}`;

export const buildAblationExperimentKey = (
  file: AblationExperimentFile,
  dimensions: readonly AblationDimensionDefinition[] = ablationDimensionDefinitions,
): string =>
  dimensions
    .map((definition) => {
      const value = getAblationDimensionValue(file.config, definition.key);
      return value === null
        ? `${definition.key}=null`
        : `${definition.key}=${getAblationDimensionValueKey(
            definition,
            value,
          )}`;
    })
    .join("|");

export const summarizeAblationDimensions = (
  files: readonly AblationExperimentFile[],
  definitions: readonly AblationDimensionDefinition[] = ablationDimensionDefinitions,
): readonly AblationDimensionSummary[] =>
  definitions
    .map((definition) => {
      const counts = new Map<AblationDimensionValue, number>();
      files.forEach((file) => {
        const value = getAblationDimensionValue(file.config, definition.key);
        if (value !== null) {
          counts.set(value, (counts.get(value) ?? 0) + 1);
        }
      });

      return {
        definition,
        values: Array.from(counts.entries())
          .map(([value, count]) => ({
            value,
            label: formatAblationDimensionValue(definition, value),
            key: getAblationDimensionValueKey(definition, value),
            count,
          }))
          .sort((left, right) =>
            compareAblationDimensionValues(definition, left.value, right.value),
          ),
      };
    })
    .filter((summary) => summary.values.length > 0);
