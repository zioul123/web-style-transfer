import {
  crossVec3,
  dimensionsPerVec2,
  dimensionsPerVec3,
  distanceVec3,
  normalizeVec3,
  subtractVec3,
  vec3At,
} from "./vector3";
import type { Vec3 } from "./vector3";

export type MeshGeometryInput = {
  readonly vertices: ArrayLike<number>;
  readonly faces: ArrayLike<number>;
  readonly uvs?: ArrayLike<number>;
};

export type MeshGeometry = {
  readonly vertices: Float32Array;
  readonly faces: Uint32Array;
  readonly uvs?: Float32Array;
  readonly vertexCount: number;
  readonly faceCount: number;
};

export type MeshGeometryAnalysis = MeshGeometry & {
  readonly area: number;
  readonly faceAreas: Float32Array;
  readonly faceNormals: Float32Array;
  readonly vertexNormals: Float32Array;
  readonly uniqueEdges: Uint32Array;
  readonly uniqueEdgeLengths: Float32Array;
  readonly meanEdgeLength: number;
  readonly faceAdjacency: Int32Array;
  readonly faceAdjacencyPairs: Uint32Array;
  readonly faceAdjacencyEdges: Uint32Array;
  readonly vertexFaceCounts: Uint32Array;
  readonly vertexFaces: Int32Array;
  readonly maxFacesPerVertex: number;
};

type EdgeRecord = {
  readonly index: number;
  readonly firstFace: number;
  readonly firstLocalEdge: number;
};

const localEdgeVertexIndices = [
  [0, 1],
  [1, 2],
  [2, 0],
] as const;

const toFiniteFloat32Array = (
  values: ArrayLike<number>,
  fieldName: string,
): Float32Array => {
  const output = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!Number.isFinite(value)) {
      throw new Error(`${fieldName}[${index}] must be a finite number.`);
    }
    output[index] = value;
  }
  return output;
};

const toFaceArray = (
  values: ArrayLike<number>,
  vertexCount: number,
): Uint32Array => {
  if (values.length % 3 !== 0) {
    throw new Error("Mesh faces length must be divisible by 3.");
  }

  const output = new Uint32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!Number.isInteger(value) || value < 0) {
      throw new Error(`faces[${index}] must be a non-negative integer.`);
    }
    if (value >= vertexCount) {
      throw new Error(
        `faces[${index}] references vertex ${value}, which is outside vertices.`,
      );
    }
    output[index] = value;
  }
  return output;
};

export const isMeshGeometryAnalysis = (
  mesh: MeshGeometryInput | MeshGeometryAnalysis,
): mesh is MeshGeometryAnalysis => "faceAreas" in mesh;

export const ensureMeshGeometryAnalysis = (
  mesh: MeshGeometryInput | MeshGeometryAnalysis,
): MeshGeometryAnalysis =>
  isMeshGeometryAnalysis(mesh) ? mesh : analyzeMeshGeometry(mesh);

export const meshFaceVertexIndex = (
  mesh: Pick<MeshGeometry, "faces">,
  faceIndex: number,
  localVertexIndex: 0 | 1 | 2,
): number => mesh.faces[faceIndex * 3 + localVertexIndex];

export const meshFaceVertex = (
  mesh: Pick<MeshGeometry, "vertices" | "faces">,
  faceIndex: number,
  localVertexIndex: 0 | 1 | 2,
): Vec3 =>
  vec3At(mesh.vertices, meshFaceVertexIndex(mesh, faceIndex, localVertexIndex));

export const meshFaceNormal = (
  mesh: Pick<MeshGeometryAnalysis, "faceNormals">,
  faceIndex: number,
): Vec3 => vec3At(mesh.faceNormals, faceIndex);

export const interpolateMeshFacePosition = (
  mesh: Pick<MeshGeometry, "vertices" | "faces">,
  faceIndex: number,
  barycentric: readonly [number, number, number],
): Vec3 => {
  const a = meshFaceVertex(mesh, faceIndex, 0);
  const b = meshFaceVertex(mesh, faceIndex, 1);
  const c = meshFaceVertex(mesh, faceIndex, 2);
  return [
    a[0] * barycentric[0] + b[0] * barycentric[1] + c[0] * barycentric[2],
    a[1] * barycentric[0] + b[1] * barycentric[1] + c[1] * barycentric[2],
    a[2] * barycentric[0] + b[2] * barycentric[1] + c[2] * barycentric[2],
  ];
};

const edgeKey = (left: number, right: number): string =>
  left < right ? `${left}:${right}` : `${right}:${left}`;

export const createMeshGeometry = (input: MeshGeometryInput): MeshGeometry => {
  if (input.vertices.length % dimensionsPerVec3 !== 0) {
    throw new Error("Mesh vertices length must be divisible by 3.");
  }
  const vertexCount = input.vertices.length / dimensionsPerVec3;
  const vertices = toFiniteFloat32Array(input.vertices, "vertices");
  const faces = toFaceArray(input.faces, vertexCount);
  const faceCount = faces.length / 3;

  let uvs: Float32Array | undefined;
  if (input.uvs !== undefined) {
    if (input.uvs.length !== vertexCount * dimensionsPerVec2) {
      throw new Error("Mesh UV length must be exactly vertexCount * 2.");
    }
    uvs = toFiniteFloat32Array(input.uvs, "uvs");
  }

  return {
    vertices,
    faces,
    uvs,
    vertexCount,
    faceCount,
  };
};

export const analyzeMeshGeometry = (
  input: MeshGeometryInput,
): MeshGeometryAnalysis => {
  const geometry = createMeshGeometry(input);
  const { vertices, vertexCount, faceCount } = geometry;
  const faceAreas = new Float32Array(faceCount);
  const faceNormals = new Float32Array(faceCount * dimensionsPerVec3);
  const vertexNormalSums = new Float64Array(vertexCount * dimensionsPerVec3);
  let area = 0;

  for (let faceIndex = 0; faceIndex < faceCount; faceIndex += 1) {
    const a = meshFaceVertex(geometry, faceIndex, 0);
    const b = meshFaceVertex(geometry, faceIndex, 1);
    const c = meshFaceVertex(geometry, faceIndex, 2);
    const ab = subtractVec3(b, a);
    const ac = subtractVec3(c, a);
    const cross = crossVec3(ab, ac);
    const doubleArea = Math.hypot(cross[0], cross[1], cross[2]);
    const faceArea = doubleArea * 0.5;
    const normal = normalizeVec3(cross);
    const normalBaseIndex = faceIndex * dimensionsPerVec3;
    faceAreas[faceIndex] = faceArea;
    faceNormals[normalBaseIndex] = normal[0];
    faceNormals[normalBaseIndex + 1] = normal[1];
    faceNormals[normalBaseIndex + 2] = normal[2];
    area += faceArea;

    for (
      let localVertexIndex = 0;
      localVertexIndex < 3;
      localVertexIndex += 1
    ) {
      const vertexIndex = meshFaceVertexIndex(
        geometry,
        faceIndex,
        localVertexIndex as 0 | 1 | 2,
      );
      const vertexBaseIndex = vertexIndex * dimensionsPerVec3;
      vertexNormalSums[vertexBaseIndex] += normal[0];
      vertexNormalSums[vertexBaseIndex + 1] += normal[1];
      vertexNormalSums[vertexBaseIndex + 2] += normal[2];
    }
  }

  const vertexNormals = new Float32Array(vertexCount * dimensionsPerVec3);
  for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex += 1) {
    const baseIndex = vertexIndex * dimensionsPerVec3;
    const normal = normalizeVec3([
      vertexNormalSums[baseIndex],
      vertexNormalSums[baseIndex + 1],
      vertexNormalSums[baseIndex + 2],
    ]);
    vertexNormals[baseIndex] = normal[0];
    vertexNormals[baseIndex + 1] = normal[1];
    vertexNormals[baseIndex + 2] = normal[2];
  }

  const uniqueEdges: number[] = [];
  const uniqueEdgeLengths: number[] = [];
  const faceAdjacency = new Int32Array(faceCount * 3);
  faceAdjacency.fill(-1);
  const faceAdjacencyPairs: number[] = [];
  const faceAdjacencyEdges: number[] = [];
  const edgeToRecord = new Map<string, EdgeRecord>();

  for (let faceIndex = 0; faceIndex < faceCount; faceIndex += 1) {
    for (let localEdgeIndex = 0; localEdgeIndex < 3; localEdgeIndex += 1) {
      const [leftLocalIndex, rightLocalIndex] =
        localEdgeVertexIndices[localEdgeIndex];
      const left = meshFaceVertexIndex(geometry, faceIndex, leftLocalIndex);
      const right = meshFaceVertexIndex(geometry, faceIndex, rightLocalIndex);
      const minVertex = Math.min(left, right);
      const maxVertex = Math.max(left, right);
      const key = edgeKey(left, right);
      const existing = edgeToRecord.get(key);
      if (existing === undefined) {
        const edgeIndex = uniqueEdges.length / 2;
        edgeToRecord.set(key, {
          index: edgeIndex,
          firstFace: faceIndex,
          firstLocalEdge: localEdgeIndex,
        });
        uniqueEdges.push(minVertex, maxVertex);
        uniqueEdgeLengths.push(
          distanceVec3(
            vec3At(vertices, minVertex),
            vec3At(vertices, maxVertex),
          ),
        );
        continue;
      }

      faceAdjacency[faceIndex * 3 + localEdgeIndex] = existing.firstFace;
      faceAdjacency[existing.firstFace * 3 + existing.firstLocalEdge] =
        faceIndex;
      faceAdjacencyPairs.push(
        Math.min(existing.firstFace, faceIndex),
        Math.max(existing.firstFace, faceIndex),
      );
      faceAdjacencyEdges.push(minVertex, maxVertex);
    }
  }

  const vertexFaceLists = Array.from(
    { length: vertexCount },
    () => [] as number[],
  );
  let maxFacesPerVertex = 0;
  for (let faceIndex = 0; faceIndex < faceCount; faceIndex += 1) {
    for (
      let localVertexIndex = 0;
      localVertexIndex < 3;
      localVertexIndex += 1
    ) {
      const vertexIndex = meshFaceVertexIndex(
        geometry,
        faceIndex,
        localVertexIndex as 0 | 1 | 2,
      );
      vertexFaceLists[vertexIndex].push(faceIndex);
      maxFacesPerVertex = Math.max(
        maxFacesPerVertex,
        vertexFaceLists[vertexIndex].length,
      );
    }
  }

  const vertexFaceCounts = new Uint32Array(vertexCount);
  const vertexFaces = new Int32Array(vertexCount * maxFacesPerVertex);
  vertexFaces.fill(-1);
  for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex += 1) {
    const incidentFaces = vertexFaceLists[vertexIndex];
    vertexFaceCounts[vertexIndex] = incidentFaces.length;
    for (
      let entryIndex = 0;
      entryIndex < incidentFaces.length;
      entryIndex += 1
    ) {
      vertexFaces[vertexIndex * maxFacesPerVertex + entryIndex] =
        incidentFaces[entryIndex];
    }
  }

  const meanEdgeLength =
    uniqueEdgeLengths.length === 0
      ? 0
      : uniqueEdgeLengths.reduce((sum, value) => sum + value, 0) /
        uniqueEdgeLengths.length;

  return {
    ...geometry,
    area,
    faceAreas,
    faceNormals,
    vertexNormals,
    uniqueEdges: new Uint32Array(uniqueEdges),
    uniqueEdgeLengths: new Float32Array(uniqueEdgeLengths),
    meanEdgeLength,
    faceAdjacency,
    faceAdjacencyPairs: new Uint32Array(faceAdjacencyPairs),
    faceAdjacencyEdges: new Uint32Array(faceAdjacencyEdges),
    vertexFaceCounts,
    vertexFaces,
    maxFacesPerVertex,
  };
};
