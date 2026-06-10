import {
  interpolateMeshPointBatch,
  meshPointBatchFromBarycentric,
} from "./meshPoints";
import {
  generateSurfaceAwareDirections,
  reshapeGeodesicEndpoints,
  traceGeodesics,
} from "./surfaceGeodesics";
import { KdTree3D } from "./kdTree3d";
import { ensureMeshGeometryAnalysis } from "./mesh";
import { sampleTexturedSurfacePointCloud } from "./surfaceSampling";
import { dimensionsPerVec3, vec3At } from "./vector3";
import type { MeshPointBatch } from "./meshPoints";
import type {
  EndpointReshapeResult,
  SurfaceAwareDirectionBatch,
} from "./surfaceGeodesics";
import type { MeshGeometryAnalysis, MeshGeometryInput } from "./mesh";
import type {
  TexturedSurfacePointCloud,
  TextureData,
  UvWrapMode,
} from "./surfaceSampling";
import type { Vec3 } from "./vector3";

export type PreparedPointCloudLayer = TexturedSurfacePointCloud & {
  readonly minSampleDistance: number;
  readonly samplesPerFace: number;
};

export type PoolingMap = {
  readonly indices: Int32Array;
  readonly fineCount: number;
  readonly coarseCount: number;
  readonly k: number;
};

export type KnnInterpolationData = {
  readonly indices: Int32Array;
  readonly weights: Float32Array;
  readonly sampleCount: number;
  readonly kernelPointCount: number;
  readonly k: number;
  readonly tau: number;
};

export type PreparedGeodesicLayer = {
  readonly starts: MeshPointBatch;
  readonly directions: SurfaceAwareDirectionBatch;
  readonly endpoints: EndpointReshapeResult;
  readonly knn: KnnInterpolationData;
};

export type PreparedMeshPointCloudData = {
  readonly mesh: MeshGeometryAnalysis;
  readonly minSampleDistances: Float32Array;
  readonly taus: Float32Array;
  readonly pointClouds: readonly PreparedPointCloudLayer[];
  readonly poolingMaps: readonly PoolingMap[];
  readonly geodesicLayers: readonly PreparedGeodesicLayer[];
};

export type PrepareMeshPointCloudOptions = {
  readonly radiusFactor?: number;
  readonly samplesPerFace?: readonly number[];
  readonly geodesicDistRatio?: number;
  readonly convAttenuation?: number;
  readonly knnK?: number | readonly number[];
  readonly seed?: number;
  readonly wrapMode?: UvWrapMode;
};

const defaultSamplesPerFace = [128, 32, 8, 2, 0.5] as const;
const defaultRadiusFactor = 0.6;
const defaultGeodesicDistRatio = 2;
const defaultConvAttenuation = 0.6827;
const defaultKnnK = 8;

const ensurePositiveFinite = (value: number, name: string): void => {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive finite number.`);
  }
};

const knnForLayer = (
  knnK: number | readonly number[],
  layerIndex: number,
): number => {
  const value = Array.isArray(knnK) ? knnK[layerIndex] : knnK;
  if (value === undefined) {
    throw new Error(`Missing knnK value for layer ${layerIndex}.`);
  }
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error("knnK values must be positive integers.");
  }
  return value;
};

const computeMinSampleDistances = (
  mesh: MeshGeometryAnalysis,
  radiusFactor: number,
  samplesPerFace: readonly number[],
): Float32Array => {
  const radiusRatio = radiusFactor * Math.sqrt(mesh.area);
  const distances = new Float32Array(samplesPerFace.length);
  for (let index = 0; index < samplesPerFace.length; index += 1) {
    ensurePositiveFinite(samplesPerFace[index], `samplesPerFace[${index}]`);
    distances[index] =
      radiusRatio / Math.sqrt(samplesPerFace[index] * mesh.faceCount);
  }
  return distances;
};

const computeTaus = (
  minSampleDistances: Float32Array,
  geodesicDistRatio: number,
  convAttenuation: number,
): Float32Array => {
  if (convAttenuation <= 0 || convAttenuation >= 1) {
    throw new Error("convAttenuation must be between 0 and 1.");
  }

  const taus = new Float32Array(minSampleDistances.length);
  for (let index = 0; index < minSampleDistances.length; index += 1) {
    const halfGeodesic = minSampleDistances[index] * geodesicDistRatio * 0.5;
    taus[index] =
      -(halfGeodesic * halfGeodesic) / Math.log(1 - convAttenuation);
  }
  return taus;
};

const writeKernelPoint = (
  output: Float32Array,
  outputIndex: number,
  value: Vec3,
): void => {
  const base = outputIndex * dimensionsPerVec3;
  output[base] = value[0];
  output[base + 1] = value[1];
  output[base + 2] = value[2];
};

export const buildPoolingMap = (
  finePoints: Float32Array,
  coarsePoints: Float32Array,
  k = 1,
): PoolingMap => {
  const tree = new KdTree3D(coarsePoints);
  const result = tree.query(finePoints, k);
  return {
    indices: result.indices,
    fineCount: result.queryCount,
    coarseCount: tree.pointCount,
    k,
  };
};

export const computeKnnInterpolationData = (
  pointPositions: Float32Array,
  startXyz: Float32Array,
  endpointXyz: Float32Array,
  k: number,
  tau: number,
): KnnInterpolationData => {
  ensurePositiveFinite(tau, "tau");
  if (!Number.isInteger(k) || k <= 0) {
    throw new Error("KNN interpolation k must be a positive integer.");
  }
  if (startXyz.length % dimensionsPerVec3 !== 0) {
    throw new Error("startXyz length must be divisible by 3.");
  }
  const sampleCount = startXyz.length / dimensionsPerVec3;
  if (endpointXyz.length !== sampleCount * 8 * dimensionsPerVec3) {
    throw new Error("endpointXyz must have shape [sampleCount, 8, 3].");
  }

  const kernelPointCount = 9;
  const kernelPoints = new Float32Array(
    sampleCount * kernelPointCount * dimensionsPerVec3,
  );
  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
    writeKernelPoint(
      kernelPoints,
      sampleIndex * kernelPointCount,
      vec3At(startXyz, sampleIndex),
    );
    for (let angleIndex = 0; angleIndex < 8; angleIndex += 1) {
      writeKernelPoint(
        kernelPoints,
        sampleIndex * kernelPointCount + 1 + angleIndex,
        vec3At(endpointXyz, sampleIndex * 8 + angleIndex),
      );
    }
  }

  const tree = new KdTree3D(pointPositions);
  const nearest = tree.query(kernelPoints, k);
  const weights = new Float32Array(nearest.indices.length);
  for (
    let kernelIndex = 0;
    kernelIndex < nearest.queryCount;
    kernelIndex += 1
  ) {
    let total = 0;
    for (let neighborIndex = 0; neighborIndex < k; neighborIndex += 1) {
      const outputIndex = kernelIndex * k + neighborIndex;
      const pointIndex = nearest.indices[outputIndex];
      if (pointIndex >= tree.pointCount) {
        weights[outputIndex] = 0;
        continue;
      }
      const distance = nearest.distances[outputIndex];
      const weight = Math.exp(-(distance * distance) / tau);
      weights[outputIndex] = weight;
      total += weight;
    }
    if (total > 0) {
      for (let neighborIndex = 0; neighborIndex < k; neighborIndex += 1) {
        const outputIndex = kernelIndex * k + neighborIndex;
        weights[outputIndex] /= total;
      }
    }
  }

  return {
    indices: nearest.indices,
    weights,
    sampleCount,
    kernelPointCount,
    k,
    tau,
  };
};

export const prepareMeshPointCloudData = (
  meshInput: MeshGeometryInput | MeshGeometryAnalysis,
  texture: TextureData,
  options?: PrepareMeshPointCloudOptions,
): PreparedMeshPointCloudData => {
  const mesh = ensureMeshGeometryAnalysis(meshInput);
  const radiusFactor = options?.radiusFactor ?? defaultRadiusFactor;
  const samplesPerFace = options?.samplesPerFace ?? defaultSamplesPerFace;
  const geodesicDistRatio =
    options?.geodesicDistRatio ?? defaultGeodesicDistRatio;
  const convAttenuation = options?.convAttenuation ?? defaultConvAttenuation;
  const knnK = options?.knnK ?? defaultKnnK;
  const seed = options?.seed ?? 1000;
  const wrapMode = options?.wrapMode ?? "wrap";

  ensurePositiveFinite(radiusFactor, "radiusFactor");
  ensurePositiveFinite(mesh.area, "mesh.area");
  ensurePositiveFinite(geodesicDistRatio, "geodesicDistRatio");

  const minSampleDistances = computeMinSampleDistances(
    mesh,
    radiusFactor,
    samplesPerFace,
  );
  const taus = computeTaus(
    minSampleDistances,
    geodesicDistRatio,
    convAttenuation,
  );
  const pointClouds: PreparedPointCloudLayer[] = [];
  const poolingMaps: PoolingMap[] = [];
  const geodesicLayers: PreparedGeodesicLayer[] = [];

  for (
    let layerIndex = 0;
    layerIndex < samplesPerFace.length;
    layerIndex += 1
  ) {
    const pointCloud = sampleTexturedSurfacePointCloud(
      mesh,
      texture,
      minSampleDistances[layerIndex],
      samplesPerFace[layerIndex],
      {
        seed: seed + layerIndex,
        wrapMode,
      },
    );
    const layer: PreparedPointCloudLayer = {
      ...pointCloud,
      minSampleDistance: minSampleDistances[layerIndex],
      samplesPerFace: samplesPerFace[layerIndex],
    };
    pointClouds.push(layer);

    if (layerIndex > 0) {
      poolingMaps.push(
        buildPoolingMap(
          pointClouds[layerIndex - 1].points,
          pointClouds[layerIndex].points,
        ),
      );
    }

    const starts = meshPointBatchFromBarycentric(
      layer.faceIndices,
      layer.barycentric,
    );
    const directions = generateSurfaceAwareDirections(
      mesh,
      starts,
      minSampleDistances[0] * geodesicDistRatio * 2 ** layerIndex,
    );
    const traced = traceGeodesics(
      mesh,
      directions.starts,
      directions.directions,
      {
        gradient: "none",
      },
    );
    const endpoints = reshapeGeodesicEndpoints(
      mesh,
      traced,
      layer.count,
      directions.angleCount,
    );
    const startXyz = interpolateMeshPointBatch(mesh, starts);
    const knn = computeKnnInterpolationData(
      layer.points,
      startXyz,
      endpoints.xyz,
      knnForLayer(knnK, layerIndex),
      taus[layerIndex],
    );

    geodesicLayers.push({
      starts,
      directions,
      endpoints,
      knn,
    });
  }

  return {
    mesh,
    minSampleDistances,
    taus,
    pointClouds,
    poolingMaps,
    geodesicLayers,
  };
};
