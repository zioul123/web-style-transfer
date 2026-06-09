import {
  Canvas,
  useFrame,
  useThree,
  type ThreeEvent,
} from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import * as THREE from "three";
import {
  pointColorAt,
  pointPositionAt,
  sampleInterpolatedColor,
} from "./math/interpolation";
import type { PointCloudHitSample, PointCloudMeshData } from "./types";

export const maxFragmentShaderPointsPerCell = 256;

type MeshColorMode = "baked" | "fragment-knn";

type PointCloudPreviewSceneProps = {
  readonly data: PointCloudMeshData;
  readonly showMesh: boolean;
  readonly showPoints: boolean;
  readonly showWireframe: boolean;
  readonly meshColorMode: MeshColorMode;
  readonly showNeighborDebug: boolean;
  readonly pointSize: number;
  readonly selectedHit: PointCloudHitSample | null;
  readonly onHoverSampleChange: (sample: PointCloudHitSample | null) => void;
};

type OrbitCameraControlsProps = {
  readonly target: readonly [number, number, number];
};

function OrbitCameraControls({ target }: OrbitCameraControlsProps) {
  const { camera, gl } = useThree();
  const controls = useMemo(() => {
    const nextControls = new OrbitControls(camera, gl.domElement);
    nextControls.enableDamping = true;
    nextControls.dampingFactor = 0.08;
    nextControls.target.set(target[0], target[1], target[2]);
    nextControls.update();
    return nextControls;
  }, [camera, gl.domElement, target]);

  useEffect(() => {
    return () => {
      controls.dispose();
    };
  }, [controls]);

  useFrame(() => {
    controls.update();
  });

  return null;
}

type CameraFramerProps = {
  readonly data: PointCloudMeshData;
};

function CameraFramer({ data }: CameraFramerProps) {
  const setThreeState = useThree((state) => state.set);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const radius = Math.max(data.bounds.radius, 0.25);
  const distance = radius * 2.75;
  const position = useMemo(
    () =>
      [
        data.bounds.center[0] + distance,
        data.bounds.center[1] + distance * 0.55,
        data.bounds.center[2] + distance,
      ] as const,
    [data.bounds.center, distance],
  );

  useEffect(() => {
    const nextCamera = cameraRef.current;
    if (nextCamera !== null) {
      setThreeState({ camera: nextCamera });
    }
  }, [setThreeState]);

  return (
    <perspectiveCamera
      ref={cameraRef}
      fov={42}
      near={Math.max(radius / 200, 0.01)}
      far={Math.max(radius * 40, 50)}
      position={position}
      onUpdate={(camera) => {
        camera.lookAt(...data.bounds.center);
        camera.updateProjectionMatrix();
      }}
    />
  );
}

type SceneContentProps = Omit<PointCloudPreviewSceneProps, "selectedHit"> & {
  readonly highlightSample: PointCloudHitSample | null;
};

const minimumSquaredDistance = 1e-16;
const fragmentShaderNeighborCellRadius = 1;
const textureMaxWidth = 2048;

const fragmentKnnVertexShader = `
  varying vec3 vSamplePosition;

  void main() {
    vSamplePosition = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const buildFragmentKnnFragmentShader = (maxPointsPerCell: number): string => `
  precision highp float;

  uniform sampler2D pointPositionTexture;
  uniform sampler2D pointColorTexture;
  uniform sampler2D cellOffsetTexture;
  uniform float pointTextureWidth;
  uniform float pointTextureHeight;
  uniform float cellOffsetTextureWidth;
  uniform float cellOffsetTextureHeight;
  uniform vec3 gridMin;
  uniform vec3 gridCellSize;
  uniform vec3 gridDimensions;

  varying vec3 vSamplePosition;

  vec4 readTextureValue(
    sampler2D textureSampler,
    int index,
    float textureWidth,
    float textureHeight
  ) {
    float row = floor(float(index) / textureWidth);
    float column = float(index) - (row * textureWidth);
    vec2 uv = vec2(
      (column + 0.5) / textureWidth,
      (row + 0.5) / textureHeight
    );
    return texture2D(textureSampler, uv);
  }

  int clampCellCoordinate(float value, float minValue, float cellSizeValue, int dimension) {
    if (dimension <= 1 || cellSizeValue <= 0.0) {
      return 0;
    }
    int coordinate = int(floor((value - minValue) / cellSizeValue));
    return min(dimension - 1, max(0, coordinate));
  }

  int flattenCellIndex(ivec3 coordinate, ivec3 dimensions) {
    return coordinate.x + (coordinate.y * dimensions.x) + (coordinate.z * dimensions.x * dimensions.y);
  }

  void insertNeighbor(
    in float squaredDistance,
    in vec3 color,
    inout float distance0,
    inout float distance1,
    inout float distance2,
    inout vec3 color0,
    inout vec3 color1,
    inout vec3 color2
  ) {
    if (squaredDistance < distance0) {
      distance2 = distance1;
      color2 = color1;
      distance1 = distance0;
      color1 = color0;
      distance0 = squaredDistance;
      color0 = color;
      return;
    }
    if (squaredDistance < distance1) {
      distance2 = distance1;
      color2 = color1;
      distance1 = squaredDistance;
      color1 = color;
      return;
    }
    if (squaredDistance < distance2) {
      distance2 = squaredDistance;
      color2 = color;
    }
  }

  void main() {
    float distance0 = 1.0e20;
    float distance1 = 1.0e20;
    float distance2 = 1.0e20;
    vec3 color0 = vec3(0.0);
    vec3 color1 = vec3(0.0);
    vec3 color2 = vec3(0.0);

    ivec3 dimensions = ivec3(
      int(gridDimensions.x + 0.5),
      int(gridDimensions.y + 0.5),
      int(gridDimensions.z + 0.5)
    );
    ivec3 queryCell = ivec3(
      clampCellCoordinate(vSamplePosition.x, gridMin.x, gridCellSize.x, dimensions.x),
      clampCellCoordinate(vSamplePosition.y, gridMin.y, gridCellSize.y, dimensions.y),
      clampCellCoordinate(vSamplePosition.z, gridMin.z, gridCellSize.z, dimensions.z)
    );

    for (int cellOffsetZ = -${fragmentShaderNeighborCellRadius}; cellOffsetZ <= ${fragmentShaderNeighborCellRadius}; cellOffsetZ += 1) {
      for (int cellOffsetY = -${fragmentShaderNeighborCellRadius}; cellOffsetY <= ${fragmentShaderNeighborCellRadius}; cellOffsetY += 1) {
        for (int cellOffsetX = -${fragmentShaderNeighborCellRadius}; cellOffsetX <= ${fragmentShaderNeighborCellRadius}; cellOffsetX += 1) {
          ivec3 cellCoordinate = queryCell + ivec3(cellOffsetX, cellOffsetY, cellOffsetZ);
          if (
            cellCoordinate.x < 0 || cellCoordinate.x >= dimensions.x ||
            cellCoordinate.y < 0 || cellCoordinate.y >= dimensions.y ||
            cellCoordinate.z < 0 || cellCoordinate.z >= dimensions.z
          ) {
            continue;
          }

          int cellIndex = flattenCellIndex(cellCoordinate, dimensions);
          int rangeStart = int(
            readTextureValue(
              cellOffsetTexture,
              cellIndex,
              cellOffsetTextureWidth,
              cellOffsetTextureHeight
            ).x + 0.5
          );
          int rangeEnd = int(
            readTextureValue(
              cellOffsetTexture,
              cellIndex + 1,
              cellOffsetTextureWidth,
              cellOffsetTextureHeight
            ).x + 0.5
          );

          for (int pointOffset = 0; pointOffset < ${maxPointsPerCell}; pointOffset += 1) {
            int pointIndex = rangeStart + pointOffset;
            if (pointIndex >= rangeEnd) {
              break;
            }

            vec3 pointPosition = readTextureValue(
              pointPositionTexture,
              pointIndex,
              pointTextureWidth,
              pointTextureHeight
            ).xyz;
            vec3 pointColor = readTextureValue(
              pointColorTexture,
              pointIndex,
              pointTextureWidth,
              pointTextureHeight
            ).xyz;
            vec3 delta = pointPosition - vSamplePosition;
            float squaredDistance = dot(delta, delta);

            insertNeighbor(
              squaredDistance,
              pointColor,
              distance0,
              distance1,
              distance2,
              color0,
              color1,
              color2
            );
          }
        }
      }
    }

    vec3 color = vec3(0.0);
    if (distance0 <= ${minimumSquaredDistance.toExponential(1)}) {
      color = color0;
    } else {
      float weight0 = 1.0 / max(distance0, ${minimumSquaredDistance.toExponential(1)});
      float weight1 = 1.0 / max(distance1, ${minimumSquaredDistance.toExponential(1)});
      float weight2 = 1.0 / max(distance2, ${minimumSquaredDistance.toExponential(1)});
      float totalWeight = weight0 + weight1 + weight2;
      color = ((color0 * weight0) + (color1 * weight1) + (color2 * weight2)) / totalWeight;
    }

    gl_FragColor = vec4(color, 1.0);
  }
`;

type PackedTexture = {
  readonly texture: THREE.DataTexture;
  readonly width: number;
  readonly height: number;
};

const buildPointTexture = (values: Float32Array): PackedTexture => {
  const pointCount = values.length / 3;
  const width = Math.min(Math.max(pointCount, 1), textureMaxWidth);
  const height = Math.max(1, Math.ceil(pointCount / width));
  const textureData = new Float32Array(width * height * 4);
  for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
    const sourceIndex = pointIndex * 3;
    const targetIndex = pointIndex * 4;
    textureData[targetIndex] = values[sourceIndex];
    textureData[targetIndex + 1] = values[sourceIndex + 1];
    textureData[targetIndex + 2] = values[sourceIndex + 2];
    textureData[targetIndex + 3] = 1;
  }

  const texture = new THREE.DataTexture(
    textureData,
    width,
    height,
    THREE.RGBAFormat,
    THREE.FloatType,
  );
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return { texture, width, height };
};

const buildScalarTexture = (values: Uint32Array): PackedTexture => {
  const width = Math.min(Math.max(values.length, 1), textureMaxWidth);
  const height = Math.max(1, Math.ceil(values.length / width));
  const textureData = new Float32Array(width * height * 4);
  for (let valueIndex = 0; valueIndex < values.length; valueIndex += 1) {
    textureData[valueIndex * 4] = values[valueIndex] ?? 0;
  }

  const texture = new THREE.DataTexture(
    textureData,
    width,
    height,
    THREE.RGBAFormat,
    THREE.FloatType,
  );
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return { texture, width, height };
};

function SceneContent({
  data,
  showMesh,
  showPoints,
  showWireframe,
  meshColorMode,
  showNeighborDebug,
  pointSize,
  highlightSample,
  onHoverSampleChange,
}: SceneContentProps) {
  const meshGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(data.meshPositions, 3),
    );
    geometry.setAttribute(
      "color",
      new THREE.BufferAttribute(data.meshVertexColors, 3),
    );
    geometry.setIndex(new THREE.BufferAttribute(data.meshIndices, 1));
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    return geometry;
  }, [data.meshIndices, data.meshPositions, data.meshVertexColors]);

  const pointGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(data.pointPositions, 3),
    );
    geometry.setAttribute(
      "color",
      new THREE.BufferAttribute(data.pointColors, 3),
    );
    geometry.computeBoundingSphere();
    return geometry;
  }, [data.pointColors, data.pointPositions]);

  const debugLinePositions = useMemo(() => {
    if (!showNeighborDebug || highlightSample === null) {
      return new Float32Array();
    }

    const positions = new Float32Array(highlightSample.neighbors.length * 6);
    highlightSample.neighbors.forEach((neighbor, index) => {
      const baseIndex = index * 6;
      positions[baseIndex] = highlightSample.point[0];
      positions[baseIndex + 1] = highlightSample.point[1];
      positions[baseIndex + 2] = highlightSample.point[2];
      positions[baseIndex + 3] = neighbor.position[0];
      positions[baseIndex + 4] = neighbor.position[1];
      positions[baseIndex + 5] = neighbor.position[2];
    });
    return positions;
  }, [highlightSample, showNeighborDebug]);

  const debugLineGeometry = useMemo(() => {
    if (debugLinePositions.length === 0) {
      return null;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(debugLinePositions, 3),
    );
    return geometry;
  }, [debugLinePositions]);

  const fragmentShaderPointPositionTexture = useMemo(() => {
    if (meshColorMode !== "fragment-knn") {
      return null;
    }
    return buildPointTexture(data.spatialHash.sortedPointPositions);
  }, [data.spatialHash.sortedPointPositions, meshColorMode]);

  const fragmentShaderPointColorTexture = useMemo(() => {
    if (meshColorMode !== "fragment-knn") {
      return null;
    }
    return buildPointTexture(data.spatialHash.sortedPointColors);
  }, [data.spatialHash.sortedPointColors, meshColorMode]);

  const fragmentShaderCellOffsetTexture = useMemo(() => {
    if (meshColorMode !== "fragment-knn") {
      return null;
    }
    return buildScalarTexture(data.spatialHash.cellOffsets);
  }, [data.spatialHash.cellOffsets, meshColorMode]);

  const fragmentShaderMaterial = useMemo(() => {
    if (
      meshColorMode !== "fragment-knn" ||
      fragmentShaderPointPositionTexture === null ||
      fragmentShaderPointColorTexture === null ||
      fragmentShaderCellOffsetTexture === null
    ) {
      return null;
    }

    return new THREE.ShaderMaterial({
      uniforms: {
        pointPositionTexture: {
          value: fragmentShaderPointPositionTexture.texture,
        },
        pointColorTexture: { value: fragmentShaderPointColorTexture.texture },
        cellOffsetTexture: { value: fragmentShaderCellOffsetTexture.texture },
        pointTextureWidth: { value: fragmentShaderPointPositionTexture.width },
        pointTextureHeight: {
          value: fragmentShaderPointPositionTexture.height,
        },
        cellOffsetTextureWidth: {
          value: fragmentShaderCellOffsetTexture.width,
        },
        cellOffsetTextureHeight: {
          value: fragmentShaderCellOffsetTexture.height,
        },
        gridMin: { value: new THREE.Vector3(...data.spatialHash.min) },
        gridCellSize: {
          value: new THREE.Vector3(...data.spatialHash.cellSize),
        },
        gridDimensions: {
          value: new THREE.Vector3(...data.spatialHash.dimensions),
        },
      },
      vertexShader: fragmentKnnVertexShader,
      fragmentShader: buildFragmentKnnFragmentShader(
        data.spatialHash.maxPointsPerCell,
      ),
      side: THREE.DoubleSide,
      wireframe: showWireframe,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });
  }, [
    data.spatialHash.cellSize,
    data.spatialHash.dimensions,
    data.spatialHash.maxPointsPerCell,
    data.spatialHash.min,
    fragmentShaderCellOffsetTexture,
    fragmentShaderPointColorTexture,
    fragmentShaderPointPositionTexture,
    meshColorMode,
    showWireframe,
  ]);

  useEffect(
    () => () => {
      meshGeometry.dispose();
      pointGeometry.dispose();
      debugLineGeometry?.dispose();
      fragmentShaderPointPositionTexture?.texture.dispose();
      fragmentShaderPointColorTexture?.texture.dispose();
      fragmentShaderCellOffsetTexture?.texture.dispose();
      fragmentShaderMaterial?.dispose();
    },
    [
      debugLineGeometry,
      fragmentShaderCellOffsetTexture,
      fragmentShaderMaterial,
      fragmentShaderPointColorTexture,
      fragmentShaderPointPositionTexture,
      meshGeometry,
      pointGeometry,
    ],
  );

  const markerRadius = Math.max(data.bounds.radius * 0.03, 0.015);
  const effectivePointSize = Math.max(data.bounds.radius * pointSize, 0.008);

  const handleIntersection = (event: ThreeEvent<PointerEvent>): void => {
    event.stopPropagation();
    const hitPoint = [event.point.x, event.point.y, event.point.z] as const;
    const { color, neighbors } = sampleInterpolatedColor(
      data.kdTree,
      data.pointPositions,
      data.pointColors,
      hitPoint,
    );
    onHoverSampleChange({
      point: hitPoint,
      color,
      neighbors: neighbors.map((neighbor) => ({
        index: neighbor.index,
        distance: Math.sqrt(neighbor.squaredDistance),
        color: pointColorAt(data.pointColors, neighbor.index),
        position: pointPositionAt(data.pointPositions, neighbor.index),
      })),
    });
  };

  return (
    <>
      <color attach="background" args={["#09111f"]} />
      <fog attach="fog" args={["#09111f", 6, 24]} />
      <CameraFramer data={data} />
      <OrbitCameraControls target={data.bounds.center} />
      <gridHelper
        args={[data.bounds.radius * 6, 12, "#164e63", "#0f172a"]}
        position={[
          data.bounds.center[0],
          data.bounds.min[1],
          data.bounds.center[2],
        ]}
      />
      {showMesh ? (
        <mesh
          geometry={meshGeometry}
          onPointerMove={handleIntersection}
          onClick={handleIntersection}
          onPointerOut={() => onHoverSampleChange(null)}
        >
          {fragmentShaderMaterial !== null ? (
            <primitive object={fragmentShaderMaterial} attach="material" />
          ) : (
            <meshBasicMaterial
              side={THREE.DoubleSide}
              vertexColors
              wireframe={showWireframe}
              polygonOffset
              polygonOffsetFactor={1}
              polygonOffsetUnits={1}
            />
          )}
        </mesh>
      ) : null}
      {showPoints ? (
        <points geometry={pointGeometry} renderOrder={1}>
          <pointsMaterial
            size={effectivePointSize}
            sizeAttenuation
            vertexColors
            depthWrite={false}
          />
        </points>
      ) : null}
      {showNeighborDebug && highlightSample !== null ? (
        <>
          <mesh position={highlightSample.point}>
            <sphereGeometry args={[markerRadius, 18, 18]} />
            <meshBasicMaterial color={highlightSample.color} />
          </mesh>
          {highlightSample.neighbors.map((neighbor) => (
            <mesh key={neighbor.index} position={neighbor.position}>
              <sphereGeometry args={[markerRadius * 0.78, 16, 16]} />
              <meshBasicMaterial color={neighbor.color} />
            </mesh>
          ))}
          {debugLineGeometry !== null ? (
            <lineSegments geometry={debugLineGeometry}>
              <lineBasicMaterial color="#f8fafc" transparent opacity={0.75} />
            </lineSegments>
          ) : null}
        </>
      ) : null}
    </>
  );
}

export function PointCloudPreviewScene({
  data,
  showMesh,
  showPoints,
  showWireframe,
  meshColorMode,
  showNeighborDebug,
  pointSize,
  selectedHit,
  onHoverSampleChange,
}: PointCloudPreviewSceneProps) {
  return (
    <div
      data-testid="pointcloud-preview-canvas"
      className="h-[28rem] overflow-hidden rounded-[2rem] border border-white/10 bg-slate-950/60 shadow-2xl shadow-black/35 lg:h-[38rem]"
    >
      <Canvas
        camera={{ fov: 42, near: 0.01, far: 1000, position: [3, 2, 3] }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: false }}
      >
        <SceneContent
          data={data}
          showMesh={showMesh}
          showPoints={showPoints}
          showWireframe={showWireframe}
          meshColorMode={meshColorMode}
          showNeighborDebug={showNeighborDebug}
          pointSize={pointSize}
          highlightSample={selectedHit}
          onHoverSampleChange={onHoverSampleChange}
        />
      </Canvas>
    </div>
  );
}
