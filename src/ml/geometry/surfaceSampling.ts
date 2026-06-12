import { KdTree3D } from "./kdTree3d";
import {
  ensureMeshGeometryAnalysis,
  interpolateMeshFacePosition,
} from "./mesh";
import {
  dimensionsPerVec2,
  dimensionsPerVec3,
  vec2At,
  vec3At,
  writeVec3At,
} from "./vector3";
import type { MeshGeometryAnalysis, MeshGeometryInput } from "./mesh";
import type { Vec2 } from "./vector3";

export type UvWrapMode = "wrap" | "clip";

export type TextureData = {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8ClampedArray | Uint8Array;
  readonly channels?: 3 | 4;
};

export type SurfaceSampleResult = {
  readonly points: Float32Array;
  readonly faceIndices: Uint32Array;
  readonly barycentric: Float32Array;
  readonly count: number;
};

export type ClosePointRemovalResult = {
  readonly points: Float32Array;
  readonly mask: Uint8Array;
  readonly count: number;
};

export type TexturedSurfacePointCloud = SurfaceSampleResult & {
  readonly uvs: Float32Array;
  readonly colors: Float32Array;
  readonly colorBytes: Uint8ClampedArray;
};

export type RandomSource = () => number;

type SampleSurfaceOptions = {
  readonly faceWeight?: ArrayLike<number>;
  readonly seed?: number;
  readonly random?: RandomSource;
};

type SampleSurfaceEvenOptions = SampleSurfaceOptions & {
  readonly radius?: number;
};

type SampleTextureOptions = {
  readonly wrapMode?: UvWrapMode;
};

export const createSeededRandom = (seed: number): RandomSource => {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
};

const defaultRandom = (): number => Math.random();

const randomFromOptions = (options?: SampleSurfaceOptions): RandomSource => {
  if (options?.random !== undefined) {
    return options.random;
  }
  if (options?.seed !== undefined) {
    return createSeededRandom(options.seed);
  }
  return defaultRandom;
};

const searchSorted = (values: Float64Array, target: number): number => {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (values[middle] < target) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return Math.min(low, values.length - 1);
};

const cumulativeWeights = (
  faceAreas: Float32Array,
  faceWeight?: ArrayLike<number>,
): Float64Array => {
  const source = faceWeight ?? faceAreas;
  if (source.length !== faceAreas.length) {
    throw new Error("faceWeight length must match mesh face count.");
  }

  const weights = new Float64Array(source.length);
  let sum = 0;
  for (let index = 0; index < source.length; index += 1) {
    const weight = source[index];
    if (!Number.isFinite(weight) || weight < 0) {
      throw new Error(`faceWeight[${index}] must be finite and non-negative.`);
    }
    sum += weight;
    weights[index] = sum;
  }
  if (sum <= 0) {
    throw new Error("At least one triangle sampling weight must be positive.");
  }
  return weights;
};

const writeBarycentric = (
  output: Float32Array,
  index: number,
  w: number,
  u: number,
  v: number,
): void => {
  const baseIndex = index * 3;
  output[baseIndex] = w;
  output[baseIndex + 1] = u;
  output[baseIndex + 2] = v;
};

const wrapUnit = (value: number): number => ((value % 1) + 1) % 1;

const clampUnit = (value: number): number => Math.min(1, Math.max(0, value));

export const normalizeUv = (uv: Vec2, wrapMode: UvWrapMode = "wrap"): Vec2 =>
  wrapMode === "wrap"
    ? [wrapUnit(uv[0]), wrapUnit(uv[1])]
    : [clampUnit(uv[0]), clampUnit(uv[1])];

export const sampleSurface = (
  meshInput: MeshGeometryInput | MeshGeometryAnalysis,
  count: number,
  options?: SampleSurfaceOptions,
): SurfaceSampleResult => {
  if (!Number.isInteger(count) || count < 0) {
    throw new Error("sampleSurface count must be a non-negative integer.");
  }

  const mesh = ensureMeshGeometryAnalysis(meshInput);
  const random = randomFromOptions(options);
  const weights = cumulativeWeights(mesh.faceAreas, options?.faceWeight);
  const totalWeight = weights[weights.length - 1];
  const points = new Float32Array(count * dimensionsPerVec3);
  const faceIndices = new Uint32Array(count);
  const barycentric = new Float32Array(count * 3);

  for (let sampleIndex = 0; sampleIndex < count; sampleIndex += 1) {
    const facePick = random() * totalWeight;
    const faceIndex = searchSorted(weights, facePick);
    faceIndices[sampleIndex] = faceIndex;

    let u = random();
    let v = random();
    if (u + v > 1) {
      u = Math.abs(u - 1);
      v = Math.abs(v - 1);
    }
    const w = 1 - u - v;
    writeBarycentric(barycentric, sampleIndex, w, u, v);

    writeVec3At(
      points,
      sampleIndex,
      interpolateMeshFacePosition(mesh, faceIndex, [w, u, v]),
    );
  }

  return {
    points,
    faceIndices,
    barycentric,
    count,
  };
};

export const removeClosePointsByDegree = (
  points: ArrayLike<number>,
  radius: number,
): ClosePointRemovalResult => {
  if (points.length % dimensionsPerVec3 !== 0) {
    throw new Error(
      "removeClosePointsByDegree points length must be divisible by 3.",
    );
  }
  if (radius < 0 || !Number.isFinite(radius)) {
    throw new Error(
      "removeClosePointsByDegree radius must be finite and non-negative.",
    );
  }

  const tree = new KdTree3D(points);
  const pairs = tree.queryPairs(radius);
  const pointCount = points.length / dimensionsPerVec3;
  const counts = new Uint32Array(pointCount);
  for (let index = 0; index < pairs.length; index += 1) {
    counts[pairs[index]] += 1;
  }

  const mask = new Uint8Array(pointCount);
  mask.fill(1);
  for (let pairIndex = 0; pairIndex < pairs.length; pairIndex += 2) {
    const left = pairs[pairIndex];
    const right = pairs[pairIndex + 1];
    const highest = counts[left] >= counts[right] ? left : right;
    mask[highest] = 0;
  }

  let keptCount = 0;
  for (let index = 0; index < mask.length; index += 1) {
    keptCount += mask[index];
  }

  const filtered = new Float32Array(keptCount * dimensionsPerVec3);
  let outputIndex = 0;
  for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
    if (mask[pointIndex] === 0) {
      continue;
    }
    writeVec3At(filtered, outputIndex, vec3At(points, pointIndex));
    outputIndex += 1;
  }

  return {
    points: filtered,
    mask,
    count: keptCount,
  };
};

export const sampleSurfaceEven = (
  meshInput: MeshGeometryInput | MeshGeometryAnalysis,
  count: number,
  options?: SampleSurfaceEvenOptions,
): SurfaceSampleResult => {
  const mesh = ensureMeshGeometryAnalysis(meshInput);
  const radius =
    options?.radius ?? Math.sqrt(mesh.area / Math.max(1, 3 * count));
  const oversampled = sampleSurface(mesh, count * 3, options);
  const removal = removeClosePointsByDegree(oversampled.points, radius);
  const outputCount = Math.min(count, removal.count);
  const points = new Float32Array(outputCount * dimensionsPerVec3);
  const faceIndices = new Uint32Array(outputCount);
  const barycentric = new Float32Array(outputCount * 3);
  let outputIndex = 0;

  for (
    let sampleIndex = 0;
    sampleIndex < oversampled.count && outputIndex < outputCount;
    sampleIndex += 1
  ) {
    if (removal.mask[sampleIndex] === 0) {
      continue;
    }

    writeVec3At(points, outputIndex, vec3At(oversampled.points, sampleIndex));
    faceIndices[outputIndex] = oversampled.faceIndices[sampleIndex];
    const inputBaryBase = sampleIndex * 3;
    writeBarycentric(
      barycentric,
      outputIndex,
      oversampled.barycentric[inputBaryBase],
      oversampled.barycentric[inputBaryBase + 1],
      oversampled.barycentric[inputBaryBase + 2],
    );
    outputIndex += 1;
  }

  return {
    points,
    faceIndices,
    barycentric,
    count: outputCount,
  };
};

export const interpolateFaceVertexUvs = (
  meshInput: MeshGeometryInput | MeshGeometryAnalysis,
  faceIndices: ArrayLike<number>,
  barycentric: ArrayLike<number>,
  wrapMode: UvWrapMode = "wrap",
): Float32Array => {
  const mesh = ensureMeshGeometryAnalysis(meshInput);
  if (mesh.uvs === undefined) {
    throw new Error("Mesh UVs are required for UV interpolation.");
  }
  if (barycentric.length !== faceIndices.length * 3) {
    throw new Error("Barycentric length must be exactly face count * 3.");
  }

  const output = new Float32Array(faceIndices.length * dimensionsPerVec2);
  for (
    let sampleIndex = 0;
    sampleIndex < faceIndices.length;
    sampleIndex += 1
  ) {
    const faceIndex = faceIndices[sampleIndex];
    if (
      !Number.isInteger(faceIndex) ||
      faceIndex < 0 ||
      faceIndex >= mesh.faceCount
    ) {
      throw new Error(
        `faceIndices[${sampleIndex}] references an invalid face.`,
      );
    }
    const baryBase = sampleIndex * 3;
    const faceBase = faceIndex * 3;
    const uv0 = vec2At(mesh.uvs, mesh.faces[faceBase]);
    const uv1 = vec2At(mesh.uvs, mesh.faces[faceBase + 1]);
    const uv2 = vec2At(mesh.uvs, mesh.faces[faceBase + 2]);
    const uv = normalizeUv(
      [
        uv0[0] * barycentric[baryBase] +
          uv1[0] * barycentric[baryBase + 1] +
          uv2[0] * barycentric[baryBase + 2],
        uv0[1] * barycentric[baryBase] +
          uv1[1] * barycentric[baryBase + 1] +
          uv2[1] * barycentric[baryBase + 2],
      ],
      wrapMode,
    );
    const outputBase = sampleIndex * dimensionsPerVec2;
    output[outputBase] = uv[0];
    output[outputBase + 1] = uv[1];
  }
  return output;
};

const textureChannelAt = (
  texture: TextureData,
  x: number,
  y: number,
  channel: 0 | 1 | 2 | 3,
): number => {
  const channels = texture.channels ?? 4;
  if (channel === 3 && channels === 3) {
    return 255;
  }
  return texture.data[(y * texture.width + x) * channels + channel];
};

const wrappedPixelIndex = (value: number, size: number): number => {
  const rounded = Math.trunc(value);
  return ((rounded % size) + size) % size;
};

export const sampleTextureBilinear = (
  texture: TextureData,
  uvs: ArrayLike<number>,
  options?: SampleTextureOptions,
): Uint8ClampedArray => {
  if (texture.width <= 0 || texture.height <= 0) {
    throw new Error("Texture width and height must be positive.");
  }
  if (uvs.length % dimensionsPerVec2 !== 0) {
    throw new Error("Texture UV length must be divisible by 2.");
  }
  const channels = texture.channels ?? 4;
  if (texture.data.length !== texture.width * texture.height * channels) {
    throw new Error(
      "Texture data length does not match dimensions and channels.",
    );
  }

  const wrapMode = options?.wrapMode ?? "wrap";
  const sampleCount = uvs.length / dimensionsPerVec2;
  const output = new Uint8ClampedArray(sampleCount * 4);
  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
    const uv = normalizeUv(vec2At(uvs, sampleIndex), wrapMode);
    const x = uv[0] * (texture.width - 1);
    const y = (1 - uv[1]) * (texture.height - 1);
    const xFloor = wrappedPixelIndex(Math.floor(x), texture.width);
    const yFloor = wrappedPixelIndex(Math.floor(y), texture.height);
    const xCeil = wrappedPixelIndex(Math.ceil(x), texture.width);
    const yCeil = wrappedPixelIndex(Math.ceil(y), texture.height);
    const dx = x - Math.floor(x);
    const dy = y - Math.floor(y);
    const weights = [
      (1 - dx) * (1 - dy),
      dx * (1 - dy),
      (1 - dx) * dy,
      dx * dy,
    ] as const;
    const outputBase = sampleIndex * 4;

    for (let channel = 0; channel < 4; channel += 1) {
      const typedChannel = channel as 0 | 1 | 2 | 3;
      const value =
        textureChannelAt(texture, xFloor, yFloor, typedChannel) * weights[0] +
        textureChannelAt(texture, xCeil, yFloor, typedChannel) * weights[1] +
        textureChannelAt(texture, xFloor, yCeil, typedChannel) * weights[2] +
        textureChannelAt(texture, xCeil, yCeil, typedChannel) * weights[3];
      output[outputBase + channel] = Math.round(value);
    }
  }
  return output;
};

export const sampleTexturedSurfacePointCloud = (
  meshInput: MeshGeometryInput | MeshGeometryAnalysis,
  texture: TextureData,
  minSampleDistance: number,
  samplesPerFace: number,
  options?: SampleSurfaceOptions & SampleTextureOptions,
): TexturedSurfacePointCloud => {
  const mesh = ensureMeshGeometryAnalysis(meshInput);
  if (!Number.isFinite(samplesPerFace) || samplesPerFace <= 0) {
    throw new Error("samplesPerFace must be a positive finite number.");
  }
  const sampleCount = Math.max(1, Math.floor(mesh.faceCount * samplesPerFace));
  const samples = sampleSurfaceEven(mesh, sampleCount, {
    ...options,
    radius: minSampleDistance,
  });
  const uvs = interpolateFaceVertexUvs(
    mesh,
    samples.faceIndices,
    samples.barycentric,
    options?.wrapMode,
  );
  const colorBytes = sampleTextureBilinear(texture, uvs, {
    wrapMode: options?.wrapMode,
  });
  const colors = new Float32Array(samples.count * dimensionsPerVec3);
  for (let sampleIndex = 0; sampleIndex < samples.count; sampleIndex += 1) {
    const colorBase = sampleIndex * 4;
    const outputBase = sampleIndex * dimensionsPerVec3;
    colors[outputBase] = colorBytes[colorBase] / 255;
    colors[outputBase + 1] = colorBytes[colorBase + 1] / 255;
    colors[outputBase + 2] = colorBytes[colorBase + 2] / 255;
  }

  return {
    ...samples,
    uvs,
    colors,
    colorBytes,
  };
};
