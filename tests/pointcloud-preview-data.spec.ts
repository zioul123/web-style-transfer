import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import {
  buildPointCloudMeshData,
  parsePointCloudMeshText,
} from "../src/features/pointcloud-preview/loadPointCloudMesh";
import { sampleInterpolatedColor } from "../src/features/pointcloud-preview/math/interpolation";
import {
  buildKdTree3d,
  findKNearestNeighbors,
} from "../src/features/pointcloud-preview/math/kdTree3d";
import type { PointCloudMeshJson } from "../src/features/pointcloud-preview/types";

const tinyFixture = JSON.parse(
  readFileSync(
    new URL(
      "../python-reference/pointcloud-style-transfer/pc-and-mesh-tiny-example.json",
      import.meta.url,
    ),
    "utf8",
  ),
) as PointCloudMeshJson;

test("point-cloud preview parser converts the tiny fixture into typed arrays", () => {
  const data = buildPointCloudMeshData(tinyFixture);

  expect(data.meshVertexCount).toBe(4);
  expect(data.meshFaceCount).toBe(2);
  expect(data.pointCount).toBe(3);
  expect(data.meshPositions).toBeInstanceOf(Float32Array);
  expect(data.meshIndices).toBeInstanceOf(Uint32Array);
  expect(data.meshVertexColors).toHaveLength(12);
  expect(data.bounds.min).toEqual([-1, 0, -1]);
  expect(data.bounds.max).toEqual([1, 0, 1]);
});

test("point-cloud preview parser rejects mismatched point colour counts", () => {
  const invalidText = JSON.stringify({
    ...tinyFixture,
    pc_rgb: tinyFixture.pc_rgb.slice(0, 2),
  });

  expect(() => parsePointCloudMeshText(invalidText)).toThrow(
    /pc_xyz and pc_rgb must contain the same number of entries/i,
  );
});

test("point-cloud preview parser rejects out-of-range mesh faces", () => {
  const invalidText = JSON.stringify({
    ...tinyFixture,
    m_faces: [[0, 1, 9]],
  });

  expect(() => parsePointCloudMeshText(invalidText)).toThrow(
    /references vertex 9/i,
  );
});

test("3-NN interpolation returns the exact point colour on an exact hit", () => {
  const pointPositions = new Float32Array([0, 0, 0, 2, 0, 0, 0, 2, 0]);
  const pointColors = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const kdTree = buildKdTree3d(pointPositions);

  const sample = sampleInterpolatedColor(
    kdTree,
    pointPositions,
    pointColors,
    [0, 0, 0],
  );

  expect(sample.color[0]).toBeCloseTo(1);
  expect(sample.color[1]).toBeCloseTo(0);
  expect(sample.color[2]).toBeCloseTo(0);
  expect(sample.neighbors[0].index).toBe(0);
});

test("3-NN interpolation follows inverse-squared distance weighting", () => {
  const pointPositions = new Float32Array([1, 0, 0, 2, 0, 0, 4, 0, 0]);
  const pointColors = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const kdTree = buildKdTree3d(pointPositions);

  const sample = sampleInterpolatedColor(
    kdTree,
    pointPositions,
    pointColors,
    [0, 0, 0],
  );

  const weights = [1, 1 / 4, 1 / 16];
  const totalWeight = weights.reduce((sum, value) => sum + value, 0);
  expect(sample.color[0]).toBeCloseTo(weights[0] / totalWeight);
  expect(sample.color[1]).toBeCloseTo(weights[1] / totalWeight);
  expect(sample.color[2]).toBeCloseTo(weights[2] / totalWeight);
});

test("k-d tree nearest-3 query stays ordered on the tiny fixture", () => {
  const data = buildPointCloudMeshData(tinyFixture);

  const neighbors = findKNearestNeighbors(
    data.kdTree,
    data.pointPositions,
    [0.86, 0, -0.18],
    3,
  );

  expect(neighbors).toHaveLength(3);
  expect(neighbors[0].index).toBe(0);
  expect(neighbors[0].squaredDistance).toBeLessThanOrEqual(
    neighbors[1].squaredDistance,
  );
  expect(neighbors[1].squaredDistance).toBeLessThanOrEqual(
    neighbors[2].squaredDistance,
  );
});
