import type { KdTree3DNode, NearestNeighbor } from "../types";

const dimensionsPerPoint = 3;

const coordinateAt = (
  points: Float32Array,
  pointIndex: number,
  axis: 0 | 1 | 2,
): number => points[pointIndex * dimensionsPerPoint + axis];

const squaredDistanceToTarget = (
  points: Float32Array,
  pointIndex: number,
  target: readonly [number, number, number],
): number => {
  const baseIndex = pointIndex * dimensionsPerPoint;
  const dx = points[baseIndex] - target[0];
  const dy = points[baseIndex + 1] - target[1];
  const dz = points[baseIndex + 2] - target[2];
  return dx * dx + dy * dy + dz * dz;
};

const insertNeighbor = (
  neighbors: NearestNeighbor[],
  candidate: NearestNeighbor,
  limit: number,
): void => {
  let insertAt = neighbors.findIndex(
    (neighbor) => candidate.squaredDistance < neighbor.squaredDistance,
  );
  if (insertAt === -1) {
    insertAt = neighbors.length;
  }
  neighbors.splice(insertAt, 0, candidate);
  if (neighbors.length > limit) {
    neighbors.pop();
  }
};

const buildNode = (
  points: Float32Array,
  indices: number[],
  depth: number,
): KdTree3DNode | null => {
  if (indices.length === 0) {
    return null;
  }

  const axis = (depth % dimensionsPerPoint) as 0 | 1 | 2;
  indices.sort(
    (left, right) =>
      coordinateAt(points, left, axis) - coordinateAt(points, right, axis),
  );

  const middleIndex = Math.floor(indices.length / 2);
  return {
    axis,
    index: indices[middleIndex],
    left: buildNode(points, indices.slice(0, middleIndex), depth + 1),
    right: buildNode(points, indices.slice(middleIndex + 1), depth + 1),
  };
};

export const buildKdTree3d = (points: Float32Array): KdTree3DNode | null => {
  const pointCount = Math.floor(points.length / dimensionsPerPoint);
  const indices = Array.from({ length: pointCount }, (_value, index) => index);
  return buildNode(points, indices, 0);
};

export const findKNearestNeighbors = (
  tree: KdTree3DNode | null,
  points: Float32Array,
  target: readonly [number, number, number],
  limit: number,
): readonly NearestNeighbor[] => {
  if (tree === null || limit <= 0) {
    return [];
  }

  const neighbors: NearestNeighbor[] = [];

  const visit = (node: KdTree3DNode | null): void => {
    if (node === null) {
      return;
    }

    const axis = node.axis;
    const axisDistance = target[axis] - coordinateAt(points, node.index, axis);
    const primaryBranch = axisDistance <= 0 ? node.left : node.right;
    const secondaryBranch = axisDistance <= 0 ? node.right : node.left;

    visit(primaryBranch);

    insertNeighbor(
      neighbors,
      {
        index: node.index,
        squaredDistance: squaredDistanceToTarget(points, node.index, target),
      },
      limit,
    );

    const furthestDistance =
      neighbors.length < limit
        ? Number.POSITIVE_INFINITY
        : neighbors[neighbors.length - 1].squaredDistance;
    if (axisDistance * axisDistance <= furthestDistance) {
      visit(secondaryBranch);
    }
  };

  visit(tree);
  return neighbors;
};
