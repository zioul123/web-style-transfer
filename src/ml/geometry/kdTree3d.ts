import { dimensionsPerVec3, squaredDistanceVec3, vec3At } from "./vector3";
import type { Vec3 } from "./vector3";

export type NearestNeighbor = {
  readonly index: number;
  readonly squaredDistance: number;
};

export type KdTree3DNode = {
  readonly axis: 0 | 1 | 2;
  readonly index: number;
  readonly left: KdTree3DNode | null;
  readonly right: KdTree3DNode | null;
};

export type KdTree3DQueryResult = {
  readonly distances: Float64Array;
  readonly indices: Int32Array;
  readonly queryCount: number;
  readonly neighborCount: number;
};

const coordinateAt = (
  points: ArrayLike<number>,
  pointIndex: number,
  axis: 0 | 1 | 2,
): number => points[pointIndex * dimensionsPerVec3 + axis];

const squaredDistanceToTarget = (
  points: ArrayLike<number>,
  pointIndex: number,
  target: Vec3,
): number => squaredDistanceVec3(vec3At(points, pointIndex), target);

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
  points: ArrayLike<number>,
  indices: number[],
  depth: number,
): KdTree3DNode | null => {
  if (indices.length === 0) {
    return null;
  }

  const axis = (depth % dimensionsPerVec3) as 0 | 1 | 2;
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

const normalizePoints = (points: ArrayLike<number>): Float32Array => {
  if (points.length % dimensionsPerVec3 !== 0) {
    throw new Error("KD-tree points length must be divisible by 3.");
  }

  const normalized = new Float32Array(points.length);
  for (let index = 0; index < points.length; index += 1) {
    const value = points[index];
    if (!Number.isFinite(value)) {
      throw new Error(`KD-tree coordinate at ${index} must be finite.`);
    }
    normalized[index] = value;
  }
  return normalized;
};

const normalizeQueryTargets = (
  targets: ArrayLike<number> | Vec3,
): Float32Array => {
  if (
    targets.length !== dimensionsPerVec3 &&
    targets.length % dimensionsPerVec3 !== 0
  ) {
    throw new Error(
      "KD-tree query targets length must be 3 or divisible by 3.",
    );
  }
  return normalizePoints(targets);
};

export const buildKdTree3d = (
  points: ArrayLike<number>,
): KdTree3DNode | null => {
  const pointCount = Math.floor(points.length / dimensionsPerVec3);
  const indices = Array.from({ length: pointCount }, (_value, index) => index);
  return buildNode(points, indices, 0);
};

export const findKNearestNeighbors = (
  tree: KdTree3DNode | null,
  points: ArrayLike<number>,
  target: Vec3,
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

export const findNeighborsWithinRadius = (
  tree: KdTree3DNode | null,
  points: ArrayLike<number>,
  target: Vec3,
  radius: number,
): readonly NearestNeighbor[] => {
  if (tree === null || radius < 0 || !Number.isFinite(radius)) {
    return [];
  }

  const radiusSquared = radius * radius;
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

    const squaredDistance = squaredDistanceToTarget(points, node.index, target);
    if (squaredDistance <= radiusSquared) {
      neighbors.push({ index: node.index, squaredDistance });
    }

    if (axisDistance * axisDistance <= radiusSquared) {
      visit(secondaryBranch);
    }
  };

  visit(tree);
  neighbors.sort((left, right) => {
    const distanceDelta = left.squaredDistance - right.squaredDistance;
    return distanceDelta === 0 ? left.index - right.index : distanceDelta;
  });
  return neighbors;
};

export class KdTree3D {
  readonly points: Float32Array;
  readonly pointCount: number;
  readonly tree: KdTree3DNode | null;

  constructor(points: ArrayLike<number>) {
    this.points = normalizePoints(points);
    this.pointCount = this.points.length / dimensionsPerVec3;
    this.tree = buildKdTree3d(this.points);
  }

  query(targets: ArrayLike<number> | Vec3, k = 1): KdTree3DQueryResult {
    if (!Number.isInteger(k) || k <= 0) {
      throw new Error("KD-tree query k must be a positive integer.");
    }

    const normalizedTargets = normalizeQueryTargets(targets);
    const queryCount = normalizedTargets.length / dimensionsPerVec3;
    const distances = new Float64Array(queryCount * k);
    const indices = new Int32Array(queryCount * k);
    distances.fill(Number.POSITIVE_INFINITY);
    indices.fill(this.pointCount);

    for (let queryIndex = 0; queryIndex < queryCount; queryIndex += 1) {
      const target = vec3At(normalizedTargets, queryIndex);
      const neighbors = findKNearestNeighbors(
        this.tree,
        this.points,
        target,
        k,
      );
      const outputBaseIndex = queryIndex * k;
      for (
        let neighborIndex = 0;
        neighborIndex < neighbors.length;
        neighborIndex += 1
      ) {
        const neighbor = neighbors[neighborIndex];
        distances[outputBaseIndex + neighborIndex] = Math.sqrt(
          neighbor.squaredDistance,
        );
        indices[outputBaseIndex + neighborIndex] = neighbor.index;
      }
    }

    return {
      distances,
      indices,
      queryCount,
      neighborCount: k,
    };
  }

  queryPairs(radius: number): Uint32Array {
    if (radius < 0 || !Number.isFinite(radius)) {
      throw new Error(
        "KD-tree queryPairs radius must be a finite non-negative number.",
      );
    }

    const pairs: number[] = [];
    for (let pointIndex = 0; pointIndex < this.pointCount; pointIndex += 1) {
      const target = vec3At(this.points, pointIndex);
      const neighbors = findNeighborsWithinRadius(
        this.tree,
        this.points,
        target,
        radius,
      );
      for (const neighbor of neighbors) {
        if (neighbor.index > pointIndex) {
          pairs.push(pointIndex, neighbor.index);
        }
      }
    }
    return new Uint32Array(pairs);
  }
}

export const createKdTree3d = (points: ArrayLike<number>): KdTree3D =>
  new KdTree3D(points);
