import type { KdTree3DNode } from "../../ml/geometry/kdTree3d";

export type ConvolutionKernelPathPoint = readonly [number, number, number];

export type ConvolutionKernelPath = readonly ConvolutionKernelPathPoint[];

export type ConvolutionKernelPathGroup = readonly ConvolutionKernelPath[];

export type ConvolutionKernelLevelJson = readonly ConvolutionKernelPathGroup[];

export type PointCloudMeshBaseJson = {
  readonly m_verts: readonly (readonly [number, number, number])[];
  readonly m_faces: readonly (readonly [number, number, number])[];
  readonly pc_xyz: readonly (readonly [number, number, number])[];
  readonly pc_rgb: readonly (readonly [number, number, number])[];
};

export type PointCloudMeshJson = PointCloudMeshBaseJson &
  Partial<Record<`level_${number}_paths`, ConvolutionKernelLevelJson>>;

export type PointCloudMeshSourceKind =
  | "point-cloud-mesh"
  | "convolution-kernels";

export type Bounds3D = {
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
  readonly center: readonly [number, number, number];
  readonly radius: number;
};

export type SpatialHashGrid3D = {
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
  readonly cellSize: readonly [number, number, number];
  readonly dimensions: readonly [number, number, number];
  readonly cellCount: number;
  readonly cellOffsets: Uint32Array;
  readonly sortedPointPositions: Float32Array;
  readonly sortedPointColors: Float32Array;
  readonly maxPointsPerCell: number;
};

export type ConvolutionKernelLevelData = {
  readonly levelIndex: number;
  readonly label: string;
  readonly groups: readonly ConvolutionKernelPathGroup[];
  readonly anchorPositions: Float32Array;
  readonly groupCount: number;
};

export type PointCloudMeshData = {
  readonly sourceKind: PointCloudMeshSourceKind;
  readonly meshPositions: Float32Array;
  readonly meshIndices: Uint32Array;
  readonly meshVertexColors: Float32Array;
  readonly pointPositions: Float32Array;
  readonly pointColors: Float32Array;
  readonly spatialHash: SpatialHashGrid3D;
  readonly meshVertexCount: number;
  readonly meshFaceCount: number;
  readonly pointCount: number;
  readonly bounds: Bounds3D;
  readonly kdTree: KdTree3DNode | null;
  readonly convolutionKernelLevels: readonly ConvolutionKernelLevelData[];
};

export type NeighborSample = {
  readonly index: number;
  readonly distance: number;
  readonly color: readonly [number, number, number];
  readonly position: readonly [number, number, number];
};

export type PointCloudHitSample = {
  readonly point: readonly [number, number, number];
  readonly color: readonly [number, number, number];
  readonly neighbors: readonly NeighborSample[];
};

export type ConvolutionKernelHitSample = {
  readonly levelIndex: number;
  readonly groupIndex: number;
  readonly point: readonly [number, number, number];
  readonly pathCount: number;
};

export type PreviewCameraState = {
  readonly position: readonly [number, number, number];
  readonly target: readonly [number, number, number];
};

export type MeshColorMode = "baked" | "fragment-knn";

export type PointCloudPreviewRenderMode = "surface" | "kernels";

export type PointCloudPreviewBackgroundColor = "default" | "black" | "white";

export type PointCloudPreviewViewAxis =
  | "pos-x"
  | "neg-x"
  | "pos-y"
  | "neg-y"
  | "pos-z"
  | "neg-z";

export type PointCloudPreviewCameraCommand =
  | {
      readonly id: number;
      readonly type: "frame";
    }
  | {
      readonly id: number;
      readonly type: "restore";
      readonly camera: PreviewCameraState;
    }
  | {
      readonly id: number;
      readonly type: "snap-axis";
      readonly axis: PointCloudPreviewViewAxis;
    };

export type PointCloudPreviewViewSettings = {
  readonly renderMode: PointCloudPreviewRenderMode;
  readonly showMesh: boolean;
  readonly showPoints: boolean;
  readonly showWireframe: boolean;
  readonly showGroundPlane: boolean;
  readonly meshColorMode: MeshColorMode;
  readonly kernelLevelIndex: number;
  readonly pointSize: number;
  readonly backgroundColor: PointCloudPreviewBackgroundColor;
  readonly disableGammaDecoding: boolean;
  readonly brightness: number;
  readonly swapYZ: boolean;
};
