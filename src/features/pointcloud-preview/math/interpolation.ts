import {
  findKNearestNeighbors,
  type KdTree3DNode,
  type NearestNeighbor,
} from "../../../ml/geometry/kdTree3d";

const dimensionsPerPoint = 3;
const minimumSquaredDistance = 1e-16;

const colorAt = (
  colors: Float32Array,
  pointIndex: number,
): readonly [number, number, number] => {
  const baseIndex = pointIndex * dimensionsPerPoint;
  return [
    colors[baseIndex],
    colors[baseIndex + 1],
    colors[baseIndex + 2],
  ] as const;
};

const positionAt = (
  positions: Float32Array,
  pointIndex: number,
): readonly [number, number, number] => {
  const baseIndex = pointIndex * dimensionsPerPoint;
  return [
    positions[baseIndex],
    positions[baseIndex + 1],
    positions[baseIndex + 2],
  ] as const;
};

export const interpolateColorFromNeighbors = (
  pointColors: Float32Array,
  neighbors: readonly NearestNeighbor[],
): readonly [number, number, number] => {
  if (neighbors.length === 0) {
    return [0, 0, 0];
  }

  const exactNeighbor = neighbors.find(
    (neighbor) => neighbor.squaredDistance <= minimumSquaredDistance,
  );
  if (exactNeighbor !== undefined) {
    return colorAt(pointColors, exactNeighbor.index);
  }

  let totalWeight = 0;
  let red = 0;
  let green = 0;
  let blue = 0;
  for (const neighbor of neighbors) {
    const weight =
      1 / Math.max(neighbor.squaredDistance, minimumSquaredDistance);
    const [neighborRed, neighborGreen, neighborBlue] = colorAt(
      pointColors,
      neighbor.index,
    );
    totalWeight += weight;
    red += neighborRed * weight;
    green += neighborGreen * weight;
    blue += neighborBlue * weight;
  }

  if (totalWeight === 0) {
    return [0, 0, 0];
  }

  return [red / totalWeight, green / totalWeight, blue / totalWeight];
};

export const sampleInterpolatedColor = (
  kdTree: KdTree3DNode | null,
  pointPositions: Float32Array,
  pointColors: Float32Array,
  target: readonly [number, number, number],
  neighborCount = 3,
): {
  readonly color: readonly [number, number, number];
  readonly neighbors: readonly NearestNeighbor[];
} => {
  const neighbors = findKNearestNeighbors(
    kdTree,
    pointPositions,
    target,
    neighborCount,
  );
  return {
    color: interpolateColorFromNeighbors(pointColors, neighbors),
    neighbors,
  };
};

export const buildMeshVertexColors = (
  meshPositions: Float32Array,
  pointPositions: Float32Array,
  pointColors: Float32Array,
  kdTree: KdTree3DNode | null,
): Float32Array => {
  const meshVertexColors = new Float32Array(meshPositions.length);
  for (let pointIndex = 0; pointIndex < meshPositions.length; pointIndex += 3) {
    const target = [
      meshPositions[pointIndex],
      meshPositions[pointIndex + 1],
      meshPositions[pointIndex + 2],
    ] as const;
    const { color } = sampleInterpolatedColor(
      kdTree,
      pointPositions,
      pointColors,
      target,
    );
    meshVertexColors[pointIndex] = color[0];
    meshVertexColors[pointIndex + 1] = color[1];
    meshVertexColors[pointIndex + 2] = color[2];
  }
  return meshVertexColors;
};

export const pointPositionAt = positionAt;
export const pointColorAt = colorAt;
