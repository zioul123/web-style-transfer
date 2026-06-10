import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { createMeshGeometry } from "./mesh";
import type { MeshGeometry } from "./mesh";
import type { TextureData } from "./surfaceSampling";

export type ExtractedGltfMesh = MeshGeometry & {
  readonly texture?: TextureData;
};

type ExtractOptions = {
  readonly flipV?: boolean;
};

type TextureLikeImage = {
  readonly width: number;
  readonly height: number;
  readonly data?: ArrayLike<number>;
};

const isTextureLikeImage = (value: unknown): value is TextureLikeImage =>
  typeof value === "object" &&
  value !== null &&
  "width" in value &&
  "height" in value &&
  typeof (value as TextureLikeImage).width === "number" &&
  typeof (value as TextureLikeImage).height === "number";

const materialTexture = (
  material: THREE.Material,
): THREE.Texture | undefined => {
  const maybeMaterial = material as THREE.Material & {
    readonly map?: THREE.Texture | null;
  };
  return maybeMaterial.map ?? undefined;
};

const rgbaFromChannels = (
  source: ArrayLike<number>,
  width: number,
  height: number,
): Uint8ClampedArray => {
  const pixels = width * height;
  const channels = source.length / pixels;
  if (channels !== 3 && channels !== 4) {
    throw new Error("Three texture data must have 3 or 4 channels.");
  }

  const output = new Uint8ClampedArray(pixels * 4);
  for (let pixel = 0; pixel < pixels; pixel += 1) {
    const sourceBase = pixel * channels;
    const outputBase = pixel * 4;
    output[outputBase] = source[sourceBase];
    output[outputBase + 1] = source[sourceBase + 1];
    output[outputBase + 2] = source[sourceBase + 2];
    output[outputBase + 3] = channels === 4 ? source[sourceBase + 3] : 255;
  }
  return output;
};

export const textureDataFromThreeTexture = (
  texture: THREE.Texture,
): TextureData => {
  const image = texture.image as unknown;
  if (!isTextureLikeImage(image)) {
    throw new Error("Three texture does not expose image dimensions.");
  }

  if (image.data !== undefined) {
    return {
      width: image.width,
      height: image.height,
      data: rgbaFromChannels(image.data, image.width, image.height),
      channels: 4,
    };
  }

  if (typeof document === "undefined") {
    throw new Error("Canvas is required to extract non-data Three textures.");
  }

  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const context = canvas.getContext("2d");
  if (context === null) {
    throw new Error(
      "Unable to create a canvas context for texture extraction.",
    );
  }
  context.drawImage(image as CanvasImageSource, 0, 0);
  return {
    width: image.width,
    height: image.height,
    data: context.getImageData(0, 0, image.width, image.height).data,
    channels: 4,
  };
};

const pushGeometry = (
  mesh: THREE.Mesh,
  vertices: number[],
  faces: number[],
  uvs: number[],
  textures: TextureData[],
  options: Required<ExtractOptions>,
): void => {
  const geometry = mesh.geometry;
  const positions = geometry.getAttribute("position");
  if (positions === undefined) {
    return;
  }

  mesh.updateWorldMatrix(true, false);
  const vertexOffset = vertices.length / 3;
  const position = new THREE.Vector3();
  for (let index = 0; index < positions.count; index += 1) {
    position
      .fromBufferAttribute(positions, index)
      .applyMatrix4(mesh.matrixWorld);
    vertices.push(position.x, position.y, position.z);
  }

  const uvAttribute = geometry.getAttribute("uv");
  if (uvAttribute !== undefined) {
    for (let index = 0; index < uvAttribute.count; index += 1) {
      const u = uvAttribute.getX(index);
      const v = uvAttribute.getY(index);
      uvs.push(u, options.flipV ? 1 - v : v);
    }
  }

  const index = geometry.getIndex();
  if (index !== null) {
    for (let entry = 0; entry < index.count; entry += 3) {
      faces.push(
        vertexOffset + index.getX(entry),
        vertexOffset + index.getX(entry + 1),
        vertexOffset + index.getX(entry + 2),
      );
    }
  } else {
    for (let entry = 0; entry < positions.count; entry += 3) {
      faces.push(
        vertexOffset + entry,
        vertexOffset + entry + 1,
        vertexOffset + entry + 2,
      );
    }
  }

  const materials = Array.isArray(mesh.material)
    ? mesh.material
    : [mesh.material];
  for (const material of materials) {
    const texture = materialTexture(material);
    if (texture !== undefined) {
      textures.push(textureDataFromThreeTexture(texture));
      break;
    }
  }
};

export const extractMeshGeometryFromObject3D = (
  root: THREE.Object3D,
  options?: ExtractOptions,
): ExtractedGltfMesh => {
  const resolvedOptions = {
    flipV: options?.flipV ?? false,
  };
  const vertices: number[] = [];
  const faces: number[] = [];
  const uvs: number[] = [];
  const textures: TextureData[] = [];

  root.updateWorldMatrix(true, true);
  root.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      pushGeometry(child, vertices, faces, uvs, textures, resolvedOptions);
    }
  });

  if (vertices.length === 0 || faces.length === 0) {
    throw new Error("No triangle mesh geometry was found in the GLTF scene.");
  }

  const geometry = createMeshGeometry({
    vertices,
    faces,
    uvs: uvs.length === (vertices.length / 3) * 2 ? uvs : undefined,
  });
  return {
    ...geometry,
    texture: textures[0],
  };
};

export const parseGltfMeshGeometry = async (
  data: string | ArrayBuffer,
): Promise<ExtractedGltfMesh> => {
  const loader = new GLTFLoader();
  const gltf = await new Promise<Awaited<ReturnType<GLTFLoader["parseAsync"]>>>(
    (resolve, reject) => {
      loader.parse(
        data,
        "",
        (loaded) => resolve(loaded),
        (error) => reject(error),
      );
    },
  );
  return extractMeshGeometryFromObject3D(gltf.scene, { flipV: true });
};

export const loadGltfMeshGeometry = async (
  url: string,
): Promise<ExtractedGltfMesh> => {
  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync(url);
  return extractMeshGeometryFromObject3D(gltf.scene, { flipV: true });
};
