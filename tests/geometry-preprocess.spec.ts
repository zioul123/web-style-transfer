import { expect, test } from "@playwright/test";
import {
  analyzeMeshGeometry,
  buildPoolingMap,
  computeKnnInterpolationData,
  prepareMeshPointCloudData,
} from "../src/ml/geometry";
import type { TextureData } from "../src/ml/geometry";

const tinyTexture: TextureData = {
  width: 2,
  height: 2,
  data: new Uint8ClampedArray([
    255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
  ]),
  channels: 4,
};

const expectFiniteArray = (values: ArrayLike<number>): void => {
  for (let index = 0; index < values.length; index += 1) {
    expect(Number.isFinite(values[index])).toBe(true);
  }
};

test("pooling maps query fine points against coarse parents", () => {
  const pooling = buildPoolingMap(
    new Float32Array([0, 0, 0, 0.9, 0, 0, 2.1, 0, 0]),
    new Float32Array([0, 0, 0, 2, 0, 0]),
  );

  expect(pooling.fineCount).toBe(3);
  expect(pooling.coarseCount).toBe(2);
  expect(pooling.k).toBe(1);
  expect(Array.from(pooling.indices)).toEqual([0, 0, 1]);
});

test("KNN interpolation data normalizes Gaussian weights for center and endpoints", () => {
  const knn = computeKnnInterpolationData(
    new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    new Float32Array([0, 0, 0]),
    new Float32Array([
      1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]),
    2,
    1,
  );

  expect(knn.sampleCount).toBe(1);
  expect(knn.kernelPointCount).toBe(9);
  expect(knn.indices).toHaveLength(18);
  expect(knn.weights).toHaveLength(18);
  for (
    let kernelIndex = 0;
    kernelIndex < knn.kernelPointCount;
    kernelIndex += 1
  ) {
    const sum =
      knn.weights[kernelIndex * knn.k] + knn.weights[kernelIndex * knn.k + 1];
    expect(sum).toBeCloseTo(1);
  }
});

test("prepareMeshPointCloudData returns finite multi-resolution samples and geodesic data", () => {
  const mesh = analyzeMeshGeometry({
    vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]),
    faces: new Uint32Array([0, 1, 2, 1, 3, 2]),
    uvs: new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]),
  });

  const prepared = prepareMeshPointCloudData(mesh, tinyTexture, {
    radiusFactor: 0.01,
    samplesPerFace: [2, 1],
    geodesicDistRatio: 0.5,
    convAttenuation: 0.6827,
    knnK: [2, 2],
    seed: 7,
  });

  expect(prepared.mesh.area).toBeCloseTo(1);
  expect(prepared.pointClouds).toHaveLength(2);
  expect(prepared.geodesicLayers).toHaveLength(2);
  expect(prepared.poolingMaps).toHaveLength(1);
  expect(prepared.minSampleDistances[0]).toBeGreaterThan(0);
  expect(prepared.minSampleDistances[1]).toBeGreaterThan(0);

  const fine = prepared.pointClouds[0];
  const coarse = prepared.pointClouds[1];
  expect(fine.count).toBe(4);
  expect(coarse.count).toBe(2);
  expect(prepared.poolingMaps[0].indices).toHaveLength(fine.count);

  for (
    let layerIndex = 0;
    layerIndex < prepared.pointClouds.length;
    layerIndex += 1
  ) {
    const pointCloud = prepared.pointClouds[layerIndex];
    const geodesic = prepared.geodesicLayers[layerIndex];
    expect(pointCloud.points).toHaveLength(pointCloud.count * 3);
    expect(pointCloud.colors).toHaveLength(pointCloud.count * 3);
    expect(geodesic.starts.count).toBe(pointCloud.count);
    expect(geodesic.endpoints.xyz).toHaveLength(pointCloud.count * 8 * 3);
    expect(geodesic.endpoints.faces).toHaveLength(pointCloud.count * 8);
    expect(geodesic.endpoints.uvs).toHaveLength(pointCloud.count * 8 * 2);
    expect(geodesic.knn.indices).toHaveLength(pointCloud.count * 9 * 2);
    expect(geodesic.knn.weights).toHaveLength(pointCloud.count * 9 * 2);
    expectFiniteArray(pointCloud.points);
    expectFiniteArray(pointCloud.colors);
    expectFiniteArray(geodesic.endpoints.xyz);
    expectFiniteArray(geodesic.knn.weights);
  }
});

test("prepareMeshPointCloudData keeps fractional positive budgets non-empty", () => {
  const mesh = analyzeMeshGeometry({
    vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    faces: new Uint32Array([0, 1, 2]),
    uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
  });

  const prepared = prepareMeshPointCloudData(mesh, tinyTexture, {
    radiusFactor: 0.01,
    samplesPerFace: [0.125],
    geodesicDistRatio: 0.5,
    convAttenuation: 0.6827,
    knnK: [1],
    seed: 7,
  });

  expect(prepared.pointClouds).toHaveLength(1);
  expect(prepared.pointClouds[0].samplesPerFace).toBe(0.125);
  expect(prepared.pointClouds[0].count).toBe(1);
  expect(prepared.geodesicLayers[0].knn.indices).toHaveLength(9);
});
