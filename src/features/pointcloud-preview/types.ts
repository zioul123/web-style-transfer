export type PointCloudMeshJson = {
  readonly m_verts: readonly (readonly [number, number, number])[];
  readonly m_faces: readonly (readonly [number, number, number])[];
  readonly pc_xyz: readonly (readonly [number, number, number])[];
  readonly pc_rgb: readonly (readonly [number, number, number])[];
};

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

export type NearestNeighbor = {
  readonly index: number;
  readonly squaredDistance: number;
};

export type KdTree3DNode = {
  readonly axis: 0 | 1 | 2;
  readonly index: number;
  readonly left: KdTree3DNode | null;
  readonly right: KdTree3DNode | null;
};

export type PointCloudMeshData = {
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

export type PreviewCameraState = {
  readonly position: readonly [number, number, number];
  readonly target: readonly [number, number, number];
};
