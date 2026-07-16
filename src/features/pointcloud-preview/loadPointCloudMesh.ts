import { buildMeshVertexColors } from "./math/interpolation";
import { buildKdTree3d } from "../../ml/geometry/kdTree3d";
import { buildSpatialHashGrid3d } from "./math/spatialHash3d";
import type {
  Bounds3D,
  ConvolutionKernelLevelData,
  ConvolutionKernelPath,
  ConvolutionKernelPathGroup,
  ConvolutionKernelPathPoint,
  PointCloudMeshBaseJson,
  PointCloudMeshData,
  PointCloudMeshJson,
} from "./types";
import { convolutionKernelDirectionIndices } from "./types";

const dimensionsPerEntry = 3;
const pathsPerKernelGroup = convolutionKernelDirectionIndices.length;
const kernelLevelKeyPattern = /^level_(\d+)_paths$/;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

const clampColor = (value: number): number => Math.min(1, Math.max(0, value));

const parseVectorTriples = (
  value: unknown,
  fieldName: keyof PointCloudMeshBaseJson,
  options?: {
    readonly clampValues?: boolean;
    readonly integerValues?: boolean;
  },
): number[] => {
  if (!Array.isArray(value)) {
    throw new Error(`${fieldName} must be an array of 3-number entries.`);
  }

  const output: number[] = [];
  value.forEach((entry, entryIndex) => {
    if (!Array.isArray(entry) || entry.length !== dimensionsPerEntry) {
      throw new Error(
        `${fieldName}[${entryIndex}] must contain exactly 3 numeric values.`,
      );
    }
    entry.forEach((coordinate, coordinateIndex) => {
      if (!isFiniteNumber(coordinate)) {
        throw new Error(
          `${fieldName}[${entryIndex}][${coordinateIndex}] must be a finite number.`,
        );
      }
      const normalizedValue = options?.integerValues
        ? Math.trunc(coordinate)
        : coordinate;
      if (options?.integerValues && normalizedValue !== coordinate) {
        throw new Error(
          `${fieldName}[${entryIndex}][${coordinateIndex}] must be an integer index.`,
        );
      }
      output.push(
        options?.clampValues === true
          ? clampColor(normalizedValue)
          : normalizedValue,
      );
    });
  });

  return output;
};

const computeBounds = (meshPositions: Float32Array): Bounds3D => {
  if (meshPositions.length === 0) {
    return {
      min: [0, 0, 0],
      max: [0, 0, 0],
      center: [0, 0, 0],
      radius: 1,
    };
  }

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < meshPositions.length; index += 3) {
    const x = meshPositions[index];
    const y = meshPositions[index + 1];
    const z = meshPositions[index + 2];
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }

  const center = [
    (minX + maxX) / 2,
    (minY + maxY) / 2,
    (minZ + maxZ) / 2,
  ] as const;

  let radius = 0;
  for (let index = 0; index < meshPositions.length; index += 3) {
    const dx = meshPositions[index] - center[0];
    const dy = meshPositions[index + 1] - center[1];
    const dz = meshPositions[index + 2] - center[2];
    radius = Math.max(radius, Math.sqrt(dx * dx + dy * dy + dz * dz));
  }

  return {
    min: [minX, minY, minZ],
    max: [maxX, maxY, maxZ],
    center,
    radius,
  };
};

const parseKernelPathPoint = (
  value: unknown,
  fieldPath: string,
): ConvolutionKernelPathPoint => {
  if (!Array.isArray(value) || value.length !== dimensionsPerEntry) {
    throw new Error(`${fieldPath} must contain exactly 3 numeric values.`);
  }

  const coordinates = value.map((coordinate, coordinateIndex) => {
    if (!isFiniteNumber(coordinate)) {
      throw new Error(
        `${fieldPath}[${coordinateIndex}] must be a finite number.`,
      );
    }
    return coordinate;
  });

  return [coordinates[0], coordinates[1], coordinates[2]];
};

const parseConvolutionKernelLevel = (
  value: unknown,
  levelKey: string,
  levelIndex: number,
): ConvolutionKernelLevelData => {
  if (!Array.isArray(value)) {
    throw new Error(`${levelKey} must be an array of 8-path groups.`);
  }

  const groups: ConvolutionKernelPathGroup[] = value.map(
    (rawGroup, groupIndex): ConvolutionKernelPathGroup => {
      const groupPath = `${levelKey}[${groupIndex}]`;
      if (!Array.isArray(rawGroup) || rawGroup.length !== pathsPerKernelGroup) {
        throw new Error(
          `${groupPath} must contain exactly ${pathsPerKernelGroup} geodesic paths.`,
        );
      }

      return rawGroup.map((rawPath, pathIndex): ConvolutionKernelPath => {
        const pathPath = `${groupPath}[${pathIndex}]`;
        if (!Array.isArray(rawPath) || rawPath.length === 0) {
          throw new Error(
            `${pathPath} must contain at least one 3-number coordinate.`,
          );
        }
        return rawPath.map((rawPoint, pointIndex) =>
          parseKernelPathPoint(rawPoint, `${pathPath}[${pointIndex}]`),
        );
      });
    },
  );

  const anchorPositions = new Float32Array(groups.length * dimensionsPerEntry);
  groups.forEach((group, groupIndex) => {
    const anchor = group[0]?.[0];
    if (anchor === undefined) {
      throw new Error(
        `${levelKey}[${groupIndex}] must contain at least one anchor coordinate.`,
      );
    }
    const targetIndex = groupIndex * dimensionsPerEntry;
    anchorPositions[targetIndex] = anchor[0];
    anchorPositions[targetIndex + 1] = anchor[1];
    anchorPositions[targetIndex + 2] = anchor[2];
  });

  return {
    levelIndex,
    label: `Level ${levelIndex}`,
    groups,
    anchorPositions,
    groupCount: groups.length,
  };
};

const parseConvolutionKernelLevels = (
  value: Record<string, unknown>,
): readonly ConvolutionKernelLevelData[] =>
  Object.keys(value)
    .map((key) => {
      const match = kernelLevelKeyPattern.exec(key);
      if (match === null) {
        return null;
      }
      return {
        key,
        levelIndex: Number(match[1]),
      };
    })
    .filter(
      (
        entry,
      ): entry is {
        readonly key: string;
        readonly levelIndex: number;
      } => entry !== null,
    )
    .sort((left, right) => left.levelIndex - right.levelIndex)
    .map(({ key, levelIndex }) =>
      parseConvolutionKernelLevel(value[key], key, levelIndex),
    );

const validateFaceIndices = (
  meshIndices: Uint32Array,
  meshVertexCount: number,
): void => {
  for (let index = 0; index < meshIndices.length; index += 1) {
    if (meshIndices[index] >= meshVertexCount) {
      throw new Error(
        `m_faces[${Math.floor(index / 3)}] references vertex ${meshIndices[index]}, which is outside m_verts.`,
      );
    }
  }
};

export const parsePointCloudMeshJson = (value: unknown): PointCloudMeshJson => {
  if (!isRecord(value)) {
    throw new Error("Point-cloud mesh JSON must be an object.");
  }

  const meshVertices = value.m_verts;
  const meshFaces = value.m_faces;
  const pointPositions = value.pc_xyz;
  const pointColors = value.pc_rgb;
  if (
    meshVertices === undefined ||
    meshFaces === undefined ||
    pointPositions === undefined ||
    pointColors === undefined
  ) {
    throw new Error(
      "Point-cloud mesh JSON must contain m_verts, m_faces, pc_xyz, and pc_rgb.",
    );
  }

  return value as PointCloudMeshJson;
};

export const buildPointCloudMeshData = (
  json: PointCloudMeshJson,
): PointCloudMeshData => {
  const meshPositions = new Float32Array(
    parseVectorTriples(json.m_verts, "m_verts"),
  );
  const meshIndices = new Uint32Array(
    parseVectorTriples(json.m_faces, "m_faces", { integerValues: true }),
  );
  const pointPositions = new Float32Array(
    parseVectorTriples(json.pc_xyz, "pc_xyz"),
  );
  const pointColors = new Float32Array(
    parseVectorTriples(json.pc_rgb, "pc_rgb", { clampValues: true }),
  );

  const meshVertexCount = meshPositions.length / dimensionsPerEntry;
  const meshFaceCount = meshIndices.length / dimensionsPerEntry;
  const pointCount = pointPositions.length / dimensionsPerEntry;
  if (pointCount === 0) {
    throw new Error(
      "pc_xyz and pc_rgb must contain at least one point sample.",
    );
  }
  if (pointCount !== pointColors.length / dimensionsPerEntry) {
    throw new Error(
      "pc_xyz and pc_rgb must contain the same number of entries.",
    );
  }

  validateFaceIndices(meshIndices, meshVertexCount);

  const kdTree = buildKdTree3d(pointPositions);
  const spatialHash = buildSpatialHashGrid3d(pointPositions, pointColors);
  const convolutionKernelLevels = parseConvolutionKernelLevels(
    json as Record<string, unknown>,
  );
  return {
    sourceKind:
      convolutionKernelLevels.length > 0
        ? "convolution-kernels"
        : "point-cloud-mesh",
    meshPositions,
    meshIndices,
    meshVertexColors: buildMeshVertexColors(
      meshPositions,
      pointPositions,
      pointColors,
      kdTree,
    ),
    pointPositions,
    pointColors,
    spatialHash,
    meshVertexCount,
    meshFaceCount,
    pointCount,
    bounds: computeBounds(meshPositions),
    kdTree,
    convolutionKernelLevels,
  };
};

export const parsePointCloudMeshText = (text: string): PointCloudMeshData => {
  let parsedJson: unknown;
  try {
    parsedJson = JSON.parse(text) as unknown;
  } catch (error) {
    throw new Error(
      `Point-cloud mesh JSON could not be parsed: ${
        error instanceof Error ? error.message : "Unknown JSON parse error."
      }`,
      {
        cause: error,
      },
    );
  }

  return buildPointCloudMeshData(parsePointCloudMeshJson(parsedJson));
};
