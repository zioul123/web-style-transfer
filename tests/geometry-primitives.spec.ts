import { expect, test } from "@playwright/test";
import {
  KdTree3D,
  analyzeMeshGeometry,
  buildKdTree3d,
  findKNearestNeighbors,
} from "../src/ml/geometry";

test("mesh analysis computes areas, normals, edges, adjacency, and vertex faces", () => {
  const mesh = analyzeMeshGeometry({
    vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]),
    faces: new Uint32Array([0, 1, 2, 1, 3, 2]),
  });

  expect(mesh.vertexCount).toBe(4);
  expect(mesh.faceCount).toBe(2);
  expect(mesh.area).toBeCloseTo(1);
  expect(Array.from(mesh.faceAreas)).toEqual([0.5, 0.5]);
  expect(Array.from(mesh.faceNormals)).toEqual([0, 0, 1, 0, 0, 1]);
  expect(Array.from(mesh.vertexNormals)).toEqual([
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
  ]);
  expect(Array.from(mesh.uniqueEdges)).toEqual([0, 1, 1, 2, 0, 2, 1, 3, 2, 3]);
  expect(mesh.uniqueEdgeLengths).toHaveLength(5);
  expect(mesh.uniqueEdgeLengths[1]).toBeCloseTo(Math.SQRT2);
  expect(mesh.meanEdgeLength).toBeCloseTo((4 + Math.SQRT2) / 5);
  expect(Array.from(mesh.faceAdjacency)).toEqual([-1, 1, -1, -1, -1, 0]);
  expect(Array.from(mesh.faceAdjacencyPairs)).toEqual([0, 1]);
  expect(Array.from(mesh.faceAdjacencyEdges)).toEqual([1, 2]);
  expect(Array.from(mesh.vertexFaceCounts)).toEqual([1, 2, 2, 1]);
  expect(mesh.maxFacesPerVertex).toBe(2);
  expect(Array.from(mesh.vertexFaces)).toEqual([0, -1, 0, 1, 0, 1, 1, -1]);
});

test("KD-tree query returns scipy-like shaped nearest-neighbor distances and indices", () => {
  const tree = new KdTree3D(
    new Float32Array([0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3]),
  );

  const result = tree.query(new Float32Array([0.25, 0, 0, 0, 0, 2.5]), 2);

  expect(result.queryCount).toBe(2);
  expect(result.neighborCount).toBe(2);
  expect(Array.from(result.indices)).toEqual([0, 1, 3, 0]);
  expect(result.distances[0]).toBeCloseTo(0.25);
  expect(result.distances[1]).toBeCloseTo(1.75);
  expect(result.distances[2]).toBeCloseTo(0.5);
  expect(result.distances[3]).toBeCloseTo(2.5);
});

test("KD-tree queryPairs returns lexicographic radius pairs", () => {
  const tree = new KdTree3D(
    new Float32Array([0, 0, 0, 0.5, 0, 0, 2, 0, 0, 2.25, 0, 0, 4, 0, 0]),
  );

  expect(Array.from(tree.queryPairs(0.6))).toEqual([0, 1, 2, 3]);
});

test("preview-compatible KD-tree nearest search stays ordered", () => {
  const points = new Float32Array([0, 0, 0, 3, 0, 0, 1, 0, 0, 2, 0, 0]);
  const tree = buildKdTree3d(points);

  const neighbors = findKNearestNeighbors(tree, points, [0.9, 0, 0], 3);

  expect(neighbors.map((neighbor) => neighbor.index)).toEqual([2, 0, 3]);
  expect(neighbors[0].squaredDistance).toBeLessThanOrEqual(
    neighbors[1].squaredDistance,
  );
  expect(neighbors[1].squaredDistance).toBeLessThanOrEqual(
    neighbors[2].squaredDistance,
  );
});
