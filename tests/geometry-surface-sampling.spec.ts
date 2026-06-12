import { Buffer } from "node:buffer";
import { expect, test } from "@playwright/test";
import {
  analyzeMeshGeometry,
  interpolateFaceVertexUvs,
  parseGltfMeshGeometry,
  removeClosePointsByDegree,
  sampleSurface,
  sampleSurfaceEven,
  sampleTextureBilinear,
  sampleTexturedSurfacePointCloud,
} from "../src/ml/geometry";
import type { RandomSource, TextureData } from "../src/ml/geometry";

const tinyTexture: TextureData = {
  width: 2,
  height: 2,
  data: new Uint8ClampedArray([
    255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
  ]),
  channels: 4,
};

const sequenceRandom = (values: readonly number[]): RandomSource => {
  let index = 0;
  return () => {
    const value = values[index % values.length];
    index += 1;
    return value;
  };
};

const expectArrayCloseTo = (
  received: ArrayLike<number>,
  expected: readonly number[],
): void => {
  expect(received.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(received[index]).toBeCloseTo(expected[index]);
  }
};

const installProgressEventPolyfill = (): void => {
  if (typeof globalThis.ProgressEvent !== "undefined") {
    return;
  }

  class NodeProgressEvent extends Event {
    readonly lengthComputable: boolean;
    readonly loaded: number;
    readonly total: number;

    constructor(type: string, eventInitDict?: ProgressEventInit) {
      super(type, eventInitDict);
      this.lengthComputable = eventInitDict?.lengthComputable ?? false;
      this.loaded = eventInitDict?.loaded ?? 0;
      this.total = eventInitDict?.total ?? 0;
    }
  }

  Object.defineProperty(globalThis, "ProgressEvent", {
    configurable: true,
    writable: true,
    value: NodeProgressEvent,
  });
};

const makeTinyGltf = (): string => {
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const indices = new Uint16Array([0, 1, 2]);
  const uvs = new Float32Array([0, 0, 1, 0, 0, 1]);
  const positionBytes = Buffer.from(positions.buffer);
  const indexBytes = Buffer.from(indices.buffer);
  const uvBytes = Buffer.from(uvs.buffer);
  const buffer = Buffer.concat([positionBytes, indexBytes, uvBytes]);
  const indexOffset = positionBytes.byteLength;
  const uvOffset = indexOffset + indexBytes.byteLength;

  return JSON.stringify({
    asset: { version: "2.0" },
    buffers: [
      {
        uri: `data:application/octet-stream;base64,${buffer.toString("base64")}`,
        byteLength: buffer.byteLength,
      },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: positionBytes.byteLength },
      { buffer: 0, byteOffset: indexOffset, byteLength: indexBytes.byteLength },
      { buffer: 0, byteOffset: uvOffset, byteLength: uvBytes.byteLength },
    ],
    accessors: [
      {
        bufferView: 0,
        componentType: 5126,
        count: 3,
        type: "VEC3",
        min: [0, 0, 0],
        max: [1, 1, 0],
      },
      {
        bufferView: 1,
        componentType: 5123,
        count: 3,
        type: "SCALAR",
      },
      {
        bufferView: 2,
        componentType: 5126,
        count: 3,
        type: "VEC2",
      },
    ],
    meshes: [
      {
        primitives: [
          {
            attributes: { POSITION: 0, TEXCOORD_0: 2 },
            indices: 1,
            mode: 4,
          },
        ],
      },
    ],
    nodes: [{ mesh: 0 }],
    scenes: [{ nodes: [0] }],
    scene: 0,
  });
};

test("surface sampling uses weighted faces and triangle point picking", () => {
  const mesh = analyzeMeshGeometry({
    vertices: new Float32Array([
      0, 0, 0, 1, 0, 0, 0, 1, 0, 10, 0, 0, 12, 0, 0, 10, 2, 0,
    ]),
    faces: new Uint32Array([0, 1, 2, 3, 4, 5]),
  });

  const samples = sampleSurface(mesh, 2, {
    faceWeight: new Float32Array([1, 3]),
    random: sequenceRandom([0.2, 0.2, 0.3, 0.6, 0.8, 0.7]),
  });

  expect(Array.from(samples.faceIndices)).toEqual([0, 1]);
  expectArrayCloseTo(samples.barycentric, [0.5, 0.2, 0.3, 0.5, 0.2, 0.3]);
  expectArrayCloseTo(samples.points, [0.2, 0.3, 0, 10.4, 0.6, 0]);
});

test("degree-based close-point removal culls the highest-degree sample in close pairs", () => {
  const removal = removeClosePointsByDegree(
    new Float32Array([0, 0, 0, 0.1, 0, 0, 0.2, 0, 0, 2, 0, 0]),
    0.15,
  );

  expect(Array.from(removal.mask)).toEqual([1, 0, 1, 1]);
  expectArrayCloseTo(removal.points, [0, 0, 0, 0.2, 0, 0, 2, 0, 0]);
});

test("sampleSurfaceEven oversamples, removes close points, and caps to requested count", () => {
  const mesh = analyzeMeshGeometry({
    vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    faces: new Uint32Array([0, 1, 2]),
  });

  const samples = sampleSurfaceEven(mesh, 2, {
    radius: 0,
    random: sequenceRandom([
      0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.3, 0.3, 0.1, 0.4, 0.4, 0.1, 0.5, 0.2, 0.2,
      0.6, 0.1, 0.2,
    ]),
  });

  expect(samples.count).toBe(2);
  expect(Array.from(samples.faceIndices)).toEqual([0, 0]);
});

test("UV interpolation wraps or clips before bilinear texture lookup", () => {
  const mesh = analyzeMeshGeometry({
    vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    faces: new Uint32Array([0, 1, 2]),
    uvs: new Float32Array([0, 0, 2, 0, 0, -1]),
  });

  const wrapped = interpolateFaceVertexUvs(
    mesh,
    new Uint32Array([0]),
    new Float32Array([0, 0.75, 0.25]),
    "wrap",
  );
  const clipped = interpolateFaceVertexUvs(
    mesh,
    new Uint32Array([0]),
    new Float32Array([0, 0.75, 0.25]),
    "clip",
  );
  const centerColor = sampleTextureBilinear(
    tinyTexture,
    new Float32Array([0.5, 0.5]),
  );

  expect(Array.from(wrapped)).toEqual([0.5, 0.75]);
  expect(Array.from(clipped)).toEqual([1, 0]);
  expect(Array.from(sampleTextureBilinear(tinyTexture, wrapped))).toEqual([
    128, 128, 64, 255,
  ]);
  expect(
    Array.from(
      sampleTextureBilinear(tinyTexture, clipped, { wrapMode: "clip" }),
    ),
  ).toEqual([255, 255, 255, 255]);
  expect(Array.from(centerColor)).toEqual([128, 128, 128, 255]);
});

test("textured surface point cloud returns sampled positions, barycentric UVs, and normalized RGB", () => {
  const mesh = analyzeMeshGeometry({
    vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    faces: new Uint32Array([0, 1, 2]),
    uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
  });

  const pointCloud = sampleTexturedSurfacePointCloud(mesh, tinyTexture, 0, 1, {
    random: sequenceRandom([0.1, 0, 0, 0.2, 0.5, 0, 0.3, 0, 0.5]),
  });

  expect(pointCloud.count).toBe(1);
  expect(pointCloud.points).toHaveLength(3);
  expect(pointCloud.barycentric).toHaveLength(3);
  expect(pointCloud.uvs).toHaveLength(2);
  expect(pointCloud.colors).toHaveLength(3);
  expect(pointCloud.colors[0]).toBeGreaterThanOrEqual(0);
  expect(pointCloud.colors[0]).toBeLessThanOrEqual(1);
});

test("textured surface point cloud rejects non-positive point budgets", () => {
  const mesh = analyzeMeshGeometry({
    vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    faces: new Uint32Array([0, 1, 2]),
    uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
  });

  expect(() =>
    sampleTexturedSurfacePointCloud(mesh, tinyTexture, 0, 0),
  ).toThrow(/samplesPerFace must be a positive finite number/i);
  expect(() =>
    sampleTexturedSurfacePointCloud(mesh, tinyTexture, 0, -1),
  ).toThrow(/samplesPerFace must be a positive finite number/i);
  expect(() =>
    sampleTexturedSurfacePointCloud(
      mesh,
      tinyTexture,
      0,
      Number.POSITIVE_INFINITY,
    ),
  ).toThrow(/samplesPerFace must be a positive finite number/i);
});

test("GLTF extraction uses GLTFLoader geometry and flips V coordinates for texture sampling", async () => {
  installProgressEventPolyfill();

  const mesh = await parseGltfMeshGeometry(makeTinyGltf());

  expect(Array.from(mesh.vertices)).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  expect(Array.from(mesh.faces)).toEqual([0, 1, 2]);
  expect(Array.from(mesh.uvs ?? [])).toEqual([0, 1, 1, 1, 0, 0]);
});
