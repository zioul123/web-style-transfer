import { expect, test } from "@playwright/test";
import {
  analyzeMeshGeometry,
  createMeshPointBatch,
  generateSurfaceAwareDirections,
  interpolateMeshPointBatch,
  meshPointBatchBarycentric,
  meshPointBatchFromBarycentric,
  reshapeGeodesicEndpoints,
  traceGeodesics,
} from "../src/ml/geometry";

const expectArrayCloseTo = (
  received: ArrayLike<number>,
  expected: readonly number[],
): void => {
  expect(received.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(received[index]).toBeCloseTo(expected[index], 5);
  }
};

const singleTriangle = () =>
  analyzeMeshGeometry({
    vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    faces: new Uint32Array([0, 1, 2]),
  });

const twoTriangles = () =>
  analyzeMeshGeometry({
    vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]),
    faces: new Uint32Array([0, 1, 2, 1, 3, 2]),
  });

test("MeshPointBatch converts barycentric UVs and interpolates positions", () => {
  const mesh = singleTriangle();
  const batch = meshPointBatchFromBarycentric(
    new Int32Array([0]),
    new Float32Array([0.5, 0.2, 0.3]),
  );

  expect(Array.from(batch.faces)).toEqual([0]);
  expectArrayCloseTo(batch.uvs, [0.2, 0.3]);
  expectArrayCloseTo(meshPointBatchBarycentric(batch), [0.5, 0.2, 0.3]);
  expectArrayCloseTo(interpolateMeshPointBatch(mesh, batch), [0.2, 0.3, 0]);
});

test("surface-aware direction generation emits the 8 square directions per sample", () => {
  const mesh = singleTriangle();
  const batch = createMeshPointBatch(
    new Int32Array([0]),
    new Float32Array([0.2, 0.3]),
  );
  const generated = generateSurfaceAwareDirections(mesh, batch, 2);

  expect(generated.angleCount).toBe(8);
  expect(generated.starts.count).toBe(8);
  expect(Array.from(generated.starts.faces)).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
  expectArrayCloseTo(
    generated.directions,
    [
      2, 0, 0, 2, 2, 0, 0, 2, 0, -2, 2, 0, -2, 0, 0, -2, -2, 0, 0, -2, 0, 2, -2,
      0,
    ],
  );
});

test("CPU geodesic tracing stays inside a single triangle when the segment fits", () => {
  const mesh = singleTriangle();
  const starts = createMeshPointBatch(
    new Int32Array([0]),
    new Float32Array([0.25, 0.25]),
  );
  const endpoints = traceGeodesics(
    mesh,
    starts,
    new Float32Array([0.2, 0, 0]),
    { gradient: "none" },
  );

  expect(Array.from(endpoints.faces)).toEqual([0]);
  expectArrayCloseTo(endpoints.uvs, [0.45, 0.25]);
  expectArrayCloseTo(
    interpolateMeshPointBatch(mesh, endpoints),
    [0.45, 0.25, 0],
  );
});

test("CPU geodesic tracing stops on a boundary edge with no adjacent face", () => {
  const mesh = singleTriangle();
  const starts = createMeshPointBatch(
    new Int32Array([0]),
    new Float32Array([0.25, 0.25]),
  );
  const endpoints = traceGeodesics(mesh, starts, new Float32Array([1, 0, 0]), {
    gradient: "none",
  });

  expect(Array.from(endpoints.faces)).toEqual([0]);
  expectArrayCloseTo(endpoints.uvs, [0.75, 0.25]);
  expectArrayCloseTo(
    interpolateMeshPointBatch(mesh, endpoints),
    [0.75, 0.25, 0],
  );
});

test("CPU geodesic tracing crosses a shared edge onto the adjacent triangle", () => {
  const mesh = twoTriangles();
  const starts = createMeshPointBatch(
    new Int32Array([0]),
    new Float32Array([0.5, 0.25]),
  );
  const endpoints = traceGeodesics(
    mesh,
    starts,
    new Float32Array([0.4, 0, 0]),
    { gradient: "none" },
  );

  expect(Array.from(endpoints.faces)).toEqual([1]);
  expectArrayCloseTo(endpoints.uvs, [0.15, 0.1]);
  expectArrayCloseTo(
    interpolateMeshPointBatch(mesh, endpoints),
    [0.9, 0.25, 0],
  );
});

test("endpoint reshaping exposes endpoint xyz, face, and uv arrays", () => {
  const mesh = twoTriangles();
  const endpoints = createMeshPointBatch(
    new Int32Array([0, 1]),
    new Float32Array([0.2, 0.3, 0.15, 0.1]),
  );
  const reshaped = reshapeGeodesicEndpoints(mesh, endpoints, 1, 2);

  expect(reshaped.sampleCount).toBe(1);
  expect(reshaped.angleCount).toBe(2);
  expect(Array.from(reshaped.faces)).toEqual([0, 1]);
  expectArrayCloseTo(reshaped.uvs, [0.2, 0.3, 0.15, 0.1]);
  expectArrayCloseTo(reshaped.xyz, [0.2, 0.3, 0, 0.9, 0.25, 0]);
});
