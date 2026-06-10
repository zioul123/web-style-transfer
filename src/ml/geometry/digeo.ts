import { analyzeMeshGeometry } from "./mesh";
import {
  addVec3,
  crossVec3,
  dimensionsPerVec2,
  dimensionsPerVec3,
  dotVec3,
  lengthVec3,
  normalizeVec3,
  scaleVec3,
  subtractVec3,
  vec3At,
} from "./vector3";
import type { MeshGeometryAnalysis, MeshGeometryInput } from "./mesh";
import type { Vec3 } from "./vector3";

export const surfaceAwareAngleCount = 8;

export type MeshPointBatch = {
  readonly faces: Int32Array;
  readonly uvs: Float32Array;
  readonly count: number;
};

export type SurfaceAwareDirectionBatch = {
  readonly directions: Float32Array;
  readonly starts: MeshPointBatch;
  readonly angleCount: number;
};

export type EndpointReshapeResult = {
  readonly xyz: Float32Array;
  readonly faces: Int32Array;
  readonly uvs: Float32Array;
  readonly sampleCount: number;
  readonly angleCount: number;
};

export type TraceGeodesicOptions = {
  readonly gradient?: "none";
  readonly maxSteps?: number;
  readonly eps?: number;
};

type Barycentric = [number, number, number];

const primaryDirection: Vec3 = [0.9998477, 0, -0.01745241];
const secondaryDirection: Vec3 = [0, 0.9998477, -0.01745241];
const squareDirectionScales = [
  1,
  Math.SQRT2,
  1,
  Math.SQRT2,
  1,
  Math.SQRT2,
  1,
  Math.SQRT2,
] as const;

const ensureMeshAnalysis = (
  mesh: MeshGeometryInput | MeshGeometryAnalysis,
): MeshGeometryAnalysis =>
  "faceAreas" in mesh ? mesh : analyzeMeshGeometry(mesh);

const writeVec3 = (output: Float32Array, index: number, value: Vec3): void => {
  const baseIndex = index * dimensionsPerVec3;
  output[baseIndex] = value[0];
  output[baseIndex + 1] = value[1];
  output[baseIndex + 2] = value[2];
};

const faceVertexIndex = (
  mesh: MeshGeometryAnalysis,
  faceIndex: number,
  localIndex: 0 | 1 | 2,
): number => mesh.faces[faceIndex * 3 + localIndex];

const faceVertex = (
  mesh: MeshGeometryAnalysis,
  faceIndex: number,
  localIndex: 0 | 1 | 2,
): Vec3 => vec3At(mesh.vertices, faceVertexIndex(mesh, faceIndex, localIndex));

const faceNormal = (mesh: MeshGeometryAnalysis, faceIndex: number): Vec3 =>
  vec3At(mesh.faceNormals, faceIndex);

const faceBarycentricToPosition = (
  mesh: MeshGeometryAnalysis,
  faceIndex: number,
  barycentric: Barycentric,
): Vec3 => {
  const a = faceVertex(mesh, faceIndex, 0);
  const b = faceVertex(mesh, faceIndex, 1);
  const c = faceVertex(mesh, faceIndex, 2);
  return addVec3(
    addVec3(scaleVec3(a, barycentric[0]), scaleVec3(b, barycentric[1])),
    scaleVec3(c, barycentric[2]),
  );
};

const projectOntoPlane = (value: Vec3, normal: Vec3): Vec3 =>
  subtractVec3(value, scaleVec3(normal, dotVec3(value, normal)));

const rotateVec3 = (value: Vec3, angle: number, normal: Vec3): Vec3 => {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return addVec3(
    addVec3(scaleVec3(value, cos), scaleVec3(crossVec3(normal, value), sin)),
    scaleVec3(normal, dotVec3(normal, value) * (1 - cos)),
  );
};

const signedAngle = (left: Vec3, right: Vec3, normal: Vec3): number => {
  const n = normalizeVec3(normal);
  const projectedLeft = normalizeVec3(projectOntoPlane(left, n));
  const projectedRight = normalizeVec3(projectOntoPlane(right, n));
  if (lengthVec3(projectedLeft) === 0 || lengthVec3(projectedRight) === 0) {
    return 0;
  }
  return Math.atan2(
    dotVec3(n, crossVec3(projectedLeft, projectedRight)),
    dotVec3(projectedLeft, projectedRight),
  );
};

const triangleNormal = (a: Vec3, b: Vec3, c: Vec3): Vec3 =>
  normalizeVec3(crossVec3(subtractVec3(b, a), subtractVec3(c, a)));

const triEdgeCoords = (
  mesh: MeshGeometryAnalysis,
  faceIndex: number,
  direction: Vec3,
): Barycentric => {
  const a = faceVertex(mesh, faceIndex, 0);
  const b = faceVertex(mesh, faceIndex, 1);
  const c = faceVertex(mesh, faceIndex, 2);
  const e1 = subtractVec3(b, a);
  const e2 = subtractVec3(c, a);
  const d00 = dotVec3(e1, e1);
  const d01 = dotVec3(e1, e2);
  const d11 = dotVec3(e2, e2);
  const d20 = dotVec3(direction, e1);
  const d21 = dotVec3(direction, e2);
  const denom = d00 * d11 - d01 * d01;
  if (Math.abs(denom) === 0) {
    return [0, 0, 0];
  }
  const u = (d11 * d20 - d01 * d21) / denom;
  const v = (d00 * d21 - d01 * d20) / denom;
  return [-u - v, u, v];
};

const cleanBarycentric = (value: Barycentric, eps: number): Barycentric => {
  const cleaned = value.map((entry) => {
    if (Math.abs(entry) < eps) {
      return 0;
    }
    if (Math.abs(entry - 1) < eps) {
      return 1;
    }
    return entry;
  }) as Barycentric;
  const sum = cleaned[0] + cleaned[1] + cleaned[2];
  if (sum !== 0 && Math.abs(sum - 1) > eps) {
    return [cleaned[0] / sum, cleaned[1] / sum, cleaned[2] / sum];
  }
  return cleaned;
};

const edgeIndexFromBarycentric = (
  barycentric: Barycentric,
  eps: number,
): 0 | 1 | 2 | -1 => {
  if (barycentric[0] < eps) {
    return 1;
  }
  if (barycentric[1] < eps) {
    return 2;
  }
  if (barycentric[2] < eps) {
    return 0;
  }
  return -1;
};

const commonVertices = (
  mesh: MeshGeometryAnalysis,
  leftFace: number,
  rightFace: number,
): {
  readonly common: readonly number[];
  readonly leftOther?: number;
  readonly rightOther?: number;
} => {
  const left = [
    faceVertexIndex(mesh, leftFace, 0),
    faceVertexIndex(mesh, leftFace, 1),
    faceVertexIndex(mesh, leftFace, 2),
  ];
  const right = [
    faceVertexIndex(mesh, rightFace, 0),
    faceVertexIndex(mesh, rightFace, 1),
    faceVertexIndex(mesh, rightFace, 2),
  ];
  const common = left.filter((vertex) => right.includes(vertex));
  return {
    common,
    leftOther: left.find((vertex) => !common.includes(vertex)),
    rightOther: right.find((vertex) => !common.includes(vertex)),
  };
};

const nextBarycentricAcrossEdge = (
  mesh: MeshGeometryAnalysis,
  currentFace: number,
  nextFace: number,
  currentBarycentric: Barycentric,
): Barycentric => {
  if (currentFace === nextFace) {
    return currentBarycentric;
  }

  const next: Barycentric = [0, 0, 0];
  for (let currentLocal = 0; currentLocal < 3; currentLocal += 1) {
    const vertex = faceVertexIndex(
      mesh,
      currentFace,
      currentLocal as 0 | 1 | 2,
    );
    for (let nextLocal = 0; nextLocal < 3; nextLocal += 1) {
      if (faceVertexIndex(mesh, nextFace, nextLocal as 0 | 1 | 2) === vertex) {
        next[nextLocal] = currentBarycentric[currentLocal];
      }
    }
  }
  return next;
};

const transportAcrossEdge = (
  mesh: MeshGeometryAnalysis,
  currentFace: number,
  nextFace: number,
  direction: Vec3,
  currentNormal: Vec3,
): {
  readonly direction: Vec3;
  readonly normal: Vec3;
} => {
  if (currentFace === nextFace) {
    return { direction, normal: currentNormal };
  }

  const { common, leftOther, rightOther } = commonVertices(
    mesh,
    currentFace,
    nextFace,
  );
  if (
    common.length !== 2 ||
    leftOther === undefined ||
    rightOther === undefined
  ) {
    return { direction, normal: currentNormal };
  }

  const p0 = vec3At(mesh.vertices, common[0]);
  const p1 = vec3At(mesh.vertices, common[1]);
  const p2Current = vec3At(mesh.vertices, leftOther);
  const p2Next = vec3At(mesh.vertices, rightOther);
  const n1 = triangleNormal(p0, p1, p2Current);
  const n2 = triangleNormal(p1, p0, p2Next);
  const axis = normalizeVec3(subtractVec3(p1, p0));
  const angle = signedAngle(n1, n2, axis);
  const transported = rotateVec3(direction, angle, axis);
  const normalSign = dotVec3(n1, currentNormal) >= 0 ? 1 : -1;
  return {
    direction: transported,
    normal: scaleVec3(n2, normalSign),
  };
};

const traceSingleGeodesic = (
  mesh: MeshGeometryAnalysis,
  startFace: number,
  startUv: Vec3,
  direction: Vec3,
  options: Required<TraceGeodesicOptions>,
): {
  readonly face: number;
  readonly uv: readonly [number, number];
} => {
  let currentFace = startFace;
  let currentBarycentric = cleanBarycentric(
    [1 - startUv[0] - startUv[1], startUv[0], startUv[1]],
    options.eps,
  );
  let currentNormal = faceNormal(mesh, currentFace);
  let projectedDirection = projectOntoPlane(direction, currentNormal);
  const pathLength = lengthVec3(projectedDirection);
  if (pathLength < options.eps) {
    return {
      face: currentFace,
      uv: [currentBarycentric[1], currentBarycentric[2]],
    };
  }

  let unitDirection = normalizeVec3(projectedDirection);
  let travelled = 0;

  for (let step = 0; step < options.maxSteps; step += 1) {
    const remaining = pathLength - travelled;
    if (remaining <= options.eps) {
      break;
    }

    const dirBarycentric = triEdgeCoords(mesh, currentFace, unitDirection);
    let bestDistance = Number.POSITIVE_INFINITY;
    for (let index = 0; index < 3; index += 1) {
      if (dirBarycentric[index] < -options.eps) {
        const distance = -currentBarycentric[index] / dirBarycentric[index];
        if (distance > options.eps && distance < bestDistance) {
          bestDistance = distance;
        }
      }
    }

    if (bestDistance > remaining || !Number.isFinite(bestDistance)) {
      currentBarycentric = cleanBarycentric(
        [
          currentBarycentric[0] + remaining * dirBarycentric[0],
          currentBarycentric[1] + remaining * dirBarycentric[1],
          currentBarycentric[2] + remaining * dirBarycentric[2],
        ],
        options.eps,
      );
      break;
    }

    currentBarycentric = cleanBarycentric(
      [
        currentBarycentric[0] + bestDistance * dirBarycentric[0],
        currentBarycentric[1] + bestDistance * dirBarycentric[1],
        currentBarycentric[2] + bestDistance * dirBarycentric[2],
      ],
      options.eps,
    );
    travelled += bestDistance;

    const edgeIndex = edgeIndexFromBarycentric(currentBarycentric, options.eps);
    if (edgeIndex === -1) {
      break;
    }

    const nextFace = mesh.faceAdjacency[currentFace * 3 + edgeIndex];
    if (nextFace < 0) {
      break;
    }

    const transported = transportAcrossEdge(
      mesh,
      currentFace,
      nextFace,
      unitDirection,
      currentNormal,
    );
    currentBarycentric = cleanBarycentric(
      nextBarycentricAcrossEdge(
        mesh,
        currentFace,
        nextFace,
        currentBarycentric,
      ),
      options.eps,
    );
    currentFace = nextFace;
    currentNormal = transported.normal;
    projectedDirection = projectOntoPlane(transported.direction, currentNormal);
    unitDirection = normalizeVec3(projectedDirection);
    if (lengthVec3(unitDirection) < options.eps) {
      break;
    }
  }

  return {
    face: currentFace,
    uv: [currentBarycentric[1], currentBarycentric[2]],
  };
};

export const createMeshPointBatch = (
  faces: ArrayLike<number>,
  uvs: ArrayLike<number>,
): MeshPointBatch => {
  if (uvs.length !== faces.length * dimensionsPerVec2) {
    throw new Error("MeshPointBatch uvs must have shape [N, 2].");
  }

  const normalizedFaces = new Int32Array(faces.length);
  for (let index = 0; index < faces.length; index += 1) {
    const face = faces[index];
    if (!Number.isInteger(face) || face < 0) {
      throw new Error(
        `MeshPointBatch face ${index} must be a non-negative integer.`,
      );
    }
    normalizedFaces[index] = face;
  }

  const normalizedUvs = new Float32Array(uvs.length);
  for (let index = 0; index < uvs.length; index += 1) {
    const uv = uvs[index];
    if (!Number.isFinite(uv)) {
      throw new Error(`MeshPointBatch uv ${index} must be finite.`);
    }
    normalizedUvs[index] = uv;
  }

  return {
    faces: normalizedFaces,
    uvs: normalizedUvs,
    count: normalizedFaces.length,
  };
};

export const meshPointBatchFromBarycentric = (
  faces: ArrayLike<number>,
  barycentric: ArrayLike<number>,
): MeshPointBatch => {
  if (barycentric.length !== faces.length * 3) {
    throw new Error("Barycentric coordinates must have shape [N, 3].");
  }
  const uvs = new Float32Array(faces.length * dimensionsPerVec2);
  for (let index = 0; index < faces.length; index += 1) {
    uvs[index * dimensionsPerVec2] = barycentric[index * 3 + 1];
    uvs[index * dimensionsPerVec2 + 1] = barycentric[index * 3 + 2];
  }
  return createMeshPointBatch(faces, uvs);
};

export const meshPointBatchBarycentric = (
  batch: MeshPointBatch,
): Float32Array => {
  const output = new Float32Array(batch.count * 3);
  for (let index = 0; index < batch.count; index += 1) {
    const uvBase = index * dimensionsPerVec2;
    const baryBase = index * 3;
    const u = batch.uvs[uvBase];
    const v = batch.uvs[uvBase + 1];
    output[baryBase] = 1 - u - v;
    output[baryBase + 1] = u;
    output[baryBase + 2] = v;
  }
  return output;
};

export const interpolateMeshPointBatch = (
  meshInput: MeshGeometryInput | MeshGeometryAnalysis,
  batch: MeshPointBatch,
): Float32Array => {
  const mesh = ensureMeshAnalysis(meshInput);
  const output = new Float32Array(batch.count * dimensionsPerVec3);
  for (let index = 0; index < batch.count; index += 1) {
    const face = batch.faces[index];
    if (face < 0 || face >= mesh.faceCount) {
      throw new Error(
        `MeshPointBatch face ${index} references an invalid face.`,
      );
    }
    const uvBase = index * dimensionsPerVec2;
    writeVec3(
      output,
      index,
      faceBarycentricToPosition(mesh, face, [
        1 - batch.uvs[uvBase] - batch.uvs[uvBase + 1],
        batch.uvs[uvBase],
        batch.uvs[uvBase + 1],
      ]),
    );
  }
  return output;
};

export const generateSurfaceAwareDirections = (
  meshInput: MeshGeometryInput | MeshGeometryAnalysis,
  samples: MeshPointBatch,
  distanceScale: number,
  options?: {
    readonly orientationTolerance?: number;
    readonly unitScale?: number;
  },
): SurfaceAwareDirectionBatch => {
  const mesh = ensureMeshAnalysis(meshInput);
  const orientationTolerance = options?.orientationTolerance ?? 0.95;
  const unitScale = options?.unitScale ?? 1;
  const directions = new Float32Array(
    samples.count * surfaceAwareAngleCount * dimensionsPerVec3,
  );
  const repeatedFaces = new Int32Array(samples.count * surfaceAwareAngleCount);
  const repeatedUvs = new Float32Array(
    samples.count * surfaceAwareAngleCount * dimensionsPerVec2,
  );

  for (let sampleIndex = 0; sampleIndex < samples.count; sampleIndex += 1) {
    const face = samples.faces[sampleIndex];
    const normal = faceNormal(mesh, face);
    const initial =
      normal[0] < orientationTolerance ? primaryDirection : secondaryDirection;
    const tangent = scaleVec3(
      normalizeVec3(projectOntoPlane(initial, normal)),
      distanceScale * unitScale,
    );

    for (
      let angleIndex = 0;
      angleIndex < surfaceAwareAngleCount;
      angleIndex += 1
    ) {
      const theta = (2 * Math.PI * angleIndex) / surfaceAwareAngleCount;
      const scaled = scaleVec3(tangent, squareDirectionScales[angleIndex]);
      const rotated = addVec3(
        scaleVec3(scaled, Math.cos(theta)),
        scaleVec3(crossVec3(normal, scaled), Math.sin(theta)),
      );
      const outputIndex = sampleIndex * surfaceAwareAngleCount + angleIndex;
      writeVec3(directions, outputIndex, rotated);
      repeatedFaces[outputIndex] = face;
      const sourceUvBase = sampleIndex * dimensionsPerVec2;
      const outputUvBase = outputIndex * dimensionsPerVec2;
      repeatedUvs[outputUvBase] = samples.uvs[sourceUvBase];
      repeatedUvs[outputUvBase + 1] = samples.uvs[sourceUvBase + 1];
    }
  }

  return {
    directions,
    starts: createMeshPointBatch(repeatedFaces, repeatedUvs),
    angleCount: surfaceAwareAngleCount,
  };
};

export const traceGeodesics = (
  meshInput: MeshGeometryInput | MeshGeometryAnalysis,
  starts: MeshPointBatch,
  directions: ArrayLike<number>,
  options?: TraceGeodesicOptions,
): MeshPointBatch => {
  if (options?.gradient !== undefined && options.gradient !== "none") {
    throw new Error('CPU geodesic tracing only supports gradient: "none".');
  }
  if (directions.length !== starts.count * dimensionsPerVec3) {
    throw new Error("Geodesic directions must have shape [N, 3].");
  }

  const mesh = ensureMeshAnalysis(meshInput);
  const resolvedOptions: Required<TraceGeodesicOptions> = {
    gradient: "none",
    maxSteps: options?.maxSteps ?? 2000,
    eps: options?.eps ?? 1e-4,
  };
  const faces = new Int32Array(starts.count);
  const uvs = new Float32Array(starts.count * dimensionsPerVec2);

  for (let index = 0; index < starts.count; index += 1) {
    const face = starts.faces[index];
    const uvBase = index * dimensionsPerVec2;
    const direction = vec3At(directions, index);
    const endpoint = traceSingleGeodesic(
      mesh,
      face,
      [starts.uvs[uvBase], starts.uvs[uvBase + 1], 0],
      direction,
      resolvedOptions,
    );
    faces[index] = endpoint.face;
    uvs[uvBase] = endpoint.uv[0];
    uvs[uvBase + 1] = endpoint.uv[1];
  }

  return createMeshPointBatch(faces, uvs);
};

export const reshapeGeodesicEndpoints = (
  meshInput: MeshGeometryInput | MeshGeometryAnalysis,
  endpoints: MeshPointBatch,
  sampleCount: number,
  angleCount = surfaceAwareAngleCount,
): EndpointReshapeResult => {
  if (endpoints.count !== sampleCount * angleCount) {
    throw new Error("Endpoint count must equal sampleCount * angleCount.");
  }

  return {
    xyz: interpolateMeshPointBatch(meshInput, endpoints),
    faces: endpoints.faces,
    uvs: endpoints.uvs,
    sampleCount,
    angleCount,
  };
};
