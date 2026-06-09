import type { SpatialHashGrid3D } from "../types";

const dimensionsPerPoint = 3;
const axisActivityThreshold = 1e-3;
const targetPointsPerCell = 8;

const computeExtents = (
  points: Float32Array,
): {
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
  readonly extents: readonly [number, number, number];
} => {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < points.length; index += dimensionsPerPoint) {
    const x = points[index];
    const y = points[index + 1];
    const z = points[index + 2];
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }

  return {
    min: [minX, minY, minZ],
    max: [maxX, maxY, maxZ],
    extents: [maxX - minX, maxY - minY, maxZ - minZ],
  };
};

const determineGridDimensions = (
  extents: readonly [number, number, number],
  pointCount: number,
): readonly [number, number, number] => {
  const maxExtent = Math.max(...extents);
  if (!Number.isFinite(maxExtent) || maxExtent <= 0) {
    return [1, 1, 1];
  }

  const activeAxes = extents.map(
    (extent) => extent > maxExtent * axisActivityThreshold,
  ) as [boolean, boolean, boolean];
  const activeAxisCount = Math.max(
    activeAxes.filter((isActive) => isActive).length,
    1,
  );
  const approximateCellCount = Math.max(
    1,
    Math.ceil(pointCount / targetPointsPerCell),
  );
  const baseResolution = Math.pow(approximateCellCount, 1 / activeAxisCount);
  const maxActiveExtent = Math.max(
    ...extents.filter((extent, axis) => activeAxes[axis] && extent > 0),
  );

  return extents.map((extent, axis) => {
    if (!activeAxes[axis] || extent <= 0) {
      return 1;
    }
    return Math.max(1, Math.ceil((extent / maxActiveExtent) * baseResolution));
  }) as [number, number, number];
};

const flattenCellIndex = (
  x: number,
  y: number,
  z: number,
  dimensions: readonly [number, number, number],
): number => x + y * dimensions[0] + z * dimensions[0] * dimensions[1];

const coordinateForAxis = (
  value: number,
  min: number,
  extent: number,
  dimension: number,
): number => {
  if (dimension <= 1 || extent <= 0) {
    return 0;
  }
  const normalized = (value - min) / extent;
  return Math.min(
    dimension - 1,
    Math.max(0, Math.floor(normalized * dimension)),
  );
};

export const buildSpatialHashGrid3d = (
  pointPositions: Float32Array,
  pointColors: Float32Array,
): SpatialHashGrid3D => {
  const pointCount = Math.floor(pointPositions.length / dimensionsPerPoint);
  const { min, max, extents } = computeExtents(pointPositions);
  const dimensions = determineGridDimensions(extents, pointCount);
  const cellCount = dimensions[0] * dimensions[1] * dimensions[2];
  const cellCounts = new Uint32Array(cellCount);

  for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
    const baseIndex = pointIndex * dimensionsPerPoint;
    const cellX = coordinateForAxis(
      pointPositions[baseIndex],
      min[0],
      extents[0],
      dimensions[0],
    );
    const cellY = coordinateForAxis(
      pointPositions[baseIndex + 1],
      min[1],
      extents[1],
      dimensions[1],
    );
    const cellZ = coordinateForAxis(
      pointPositions[baseIndex + 2],
      min[2],
      extents[2],
      dimensions[2],
    );
    cellCounts[flattenCellIndex(cellX, cellY, cellZ, dimensions)] += 1;
  }

  const cellOffsets = new Uint32Array(cellCount + 1);
  let maxPointsPerCell = 0;
  for (let cellIndex = 0; cellIndex < cellCount; cellIndex += 1) {
    cellOffsets[cellIndex + 1] = cellOffsets[cellIndex] + cellCounts[cellIndex];
    maxPointsPerCell = Math.max(maxPointsPerCell, cellCounts[cellIndex] ?? 0);
  }

  const nextCellOffsets = cellOffsets.slice();
  const sortedPointPositions = new Float32Array(pointPositions.length);
  const sortedPointColors = new Float32Array(pointColors.length);

  for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
    const baseIndex = pointIndex * dimensionsPerPoint;
    const cellX = coordinateForAxis(
      pointPositions[baseIndex],
      min[0],
      extents[0],
      dimensions[0],
    );
    const cellY = coordinateForAxis(
      pointPositions[baseIndex + 1],
      min[1],
      extents[1],
      dimensions[1],
    );
    const cellZ = coordinateForAxis(
      pointPositions[baseIndex + 2],
      min[2],
      extents[2],
      dimensions[2],
    );
    const cellIndex = flattenCellIndex(cellX, cellY, cellZ, dimensions);
    const sortedPointIndex = nextCellOffsets[cellIndex] * dimensionsPerPoint;

    sortedPointPositions[sortedPointIndex] = pointPositions[baseIndex];
    sortedPointPositions[sortedPointIndex + 1] = pointPositions[baseIndex + 1];
    sortedPointPositions[sortedPointIndex + 2] = pointPositions[baseIndex + 2];
    sortedPointColors[sortedPointIndex] = pointColors[baseIndex];
    sortedPointColors[sortedPointIndex + 1] = pointColors[baseIndex + 1];
    sortedPointColors[sortedPointIndex + 2] = pointColors[baseIndex + 2];
    nextCellOffsets[cellIndex] += 1;
  }

  return {
    min,
    max,
    cellSize: [
      extents[0] > 0 ? extents[0] / dimensions[0] : 1,
      extents[1] > 0 ? extents[1] / dimensions[1] : 1,
      extents[2] > 0 ? extents[2] / dimensions[2] : 1,
    ],
    dimensions,
    cellCount,
    cellOffsets,
    sortedPointPositions,
    sortedPointColors,
    maxPointsPerCell,
  };
};
