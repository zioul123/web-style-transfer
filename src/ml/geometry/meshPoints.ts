import {
  ensureMeshGeometryAnalysis,
  interpolateMeshFacePosition,
} from "./mesh";
import { dimensionsPerVec2, dimensionsPerVec3, writeVec3At } from "./vector3";
import type { MeshGeometryAnalysis, MeshGeometryInput } from "./mesh";

export type MeshPointBatch = {
  readonly faces: Int32Array;
  readonly uvs: Float32Array;
  readonly count: number;
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
  const mesh = ensureMeshGeometryAnalysis(meshInput);
  const output = new Float32Array(batch.count * dimensionsPerVec3);
  for (let index = 0; index < batch.count; index += 1) {
    const face = batch.faces[index];
    if (face < 0 || face >= mesh.faceCount) {
      throw new Error(
        `MeshPointBatch face ${index} references an invalid face.`,
      );
    }
    const uvBase = index * dimensionsPerVec2;
    writeVec3At(
      output,
      index,
      interpolateMeshFacePosition(mesh, face, [
        1 - batch.uvs[uvBase] - batch.uvs[uvBase + 1],
        batch.uvs[uvBase],
        batch.uvs[uvBase + 1],
      ]),
    );
  }
  return output;
};
