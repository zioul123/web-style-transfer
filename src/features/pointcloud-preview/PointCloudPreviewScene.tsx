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
import type {
  PointCloudHitSample,
  PointCloudMeshData,
  PreviewCameraState,
} from "./types";

export const maxFragmentShaderPointsPerCell = 256;

type MeshColorMode = "baked" | "fragment-knn";
type ViewAxis = "pos-x" | "neg-x" | "pos-y" | "neg-y" | "pos-z" | "neg-z";

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
      readonly axis: ViewAxis;
    };

type PointCloudPreviewSceneProps = {
  readonly data: PointCloudMeshData;
  readonly showMesh: boolean;
  readonly showPoints: boolean;
  readonly showWireframe: boolean;
  readonly meshColorMode: MeshColorMode;
  readonly showNeighborDebug: boolean;
  readonly pointSize: number;
  readonly pointGammaCorrection: boolean;
  readonly brightness: number;
  readonly swapYZ: boolean;
  readonly selectedHit: PointCloudHitSample | null;
  readonly cameraCommand: PointCloudPreviewCameraCommand;
  readonly onHoverSampleChange: (sample: PointCloudHitSample | null) => void;
  readonly onCameraStateChange: (state: PreviewCameraState) => void;
  readonly onFramesPerSecondChange: (fps: number) => void;
};

type OrbitCameraControlsProps = {
  readonly data: PointCloudMeshData;
  readonly command: PointCloudPreviewCameraCommand;
  readonly onCameraStateChange: (state: PreviewCameraState) => void;
};

type SceneContentProps = Omit<
  PointCloudPreviewSceneProps,
  "selectedHit" | "cameraCommand"
> & {
  readonly highlightSample: PointCloudHitSample | null;
  readonly transformMatrix: THREE.Matrix4;
  readonly inverseTransformMatrix: THREE.Matrix4;
};

const minimumSquaredDistance = 1e-16;
const fragmentShaderNeighborCellRadius = 1;
const textureMaxWidth = 2048;
const cameraStateEpsilon = 1e-5;

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
  uniform float brightness;
  uniform float gammaDecodeEnabled;
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

  vec3 encodeGamma(vec3 color) {
    vec3 lowRange = color * 12.92;
    vec3 highRange = (1.055 * pow(max(color, vec3(0.0)), vec3(1.0 / 2.4))) - 0.055;
    bvec3 selector = lessThanEqual(color, vec3(0.0031308));
    return vec3(
      selector.x ? lowRange.x : highRange.x,
      selector.y ? lowRange.y : highRange.y,
      selector.z ? lowRange.z : highRange.z
    );
  }

  vec3 adjustDisplayColor(vec3 color) {
    vec3 adjusted = gammaDecodeEnabled > 0.5 ? color : encodeGamma(color);
    return clamp(adjusted * brightness, 0.0, 1.0);
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
              adjustDisplayColor(pointColor),
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

    gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
  }
`;

type PackedTexture = {
  readonly texture: THREE.DataTexture;
  readonly width: number;
  readonly height: number;
};

const buildDefaultCameraState = (
  data: PointCloudMeshData,
): PreviewCameraState => {
  const radius = Math.max(data.bounds.radius, 0.25);
  const distance = radius * 2.75;
  return {
    position: [
      data.bounds.center[0] + distance,
      data.bounds.center[1] + distance * 0.55,
      data.bounds.center[2] + distance,
    ],
    target: data.bounds.center,
  };
};

const axisVectorByView: Record<ViewAxis, readonly [number, number, number]> = {
  "pos-x": [1, 0, 0],
  "neg-x": [-1, 0, 0],
  "pos-y": [0, 1, 0],
  "neg-y": [0, -1, 0],
  "pos-z": [0, 0, 1],
  "neg-z": [0, 0, -1],
};

const vectorTriplesClose = (
  left: readonly [number, number, number],
  right: readonly [number, number, number],
): boolean =>
  Math.abs(left[0] - right[0]) <= cameraStateEpsilon &&
  Math.abs(left[1] - right[1]) <= cameraStateEpsilon &&
  Math.abs(left[2] - right[2]) <= cameraStateEpsilon;

const srgbToLinearChannel = (value: number): number =>
  value <= 0.04045 ? value / 12.92 : Math.pow((value + 0.055) / 1.055, 2.4);

const maybeGammaCorrectColors = (
  colors: Float32Array,
  enabled: boolean,
): Float32Array => {
  if (!enabled) {
    return colors;
  }
  const corrected = new Float32Array(colors.length);
  for (let index = 0; index < colors.length; index += 1) {
    corrected[index] = srgbToLinearChannel(colors[index] ?? 0);
  }
  return corrected;
};

const applyBrightnessToColors = (
  colors: Float32Array,
  brightness: number,
): Float32Array => {
  if (Math.abs(brightness - 1) <= Number.EPSILON) {
    return colors;
  }
  const adjusted = new Float32Array(colors.length);
  for (let index = 0; index < colors.length; index += 1) {
    adjusted[index] = Math.min(
      1,
      Math.max(0, (colors[index] ?? 0) * brightness),
    );
  }
  return adjusted;
};

const adjustDisplayColors = (
  colors: Float32Array,
  gammaDecodingEnabled: boolean,
  brightness: number,
): Float32Array =>
  applyBrightnessToColors(
    maybeGammaCorrectColors(colors, gammaDecodingEnabled),
    brightness,
  );

const adjustDisplayTuple = (
  color: readonly [number, number, number],
  gammaDecodingEnabled: boolean,
  brightness: number,
): readonly [number, number, number] => {
  const source = new Float32Array(color);
  const adjusted = adjustDisplayColors(
    source,
    gammaDecodingEnabled,
    brightness,
  );
  return [adjusted[0], adjusted[1], adjusted[2]];
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

function OrbitCameraControls({
  data,
  command,
  onCameraStateChange,
}: OrbitCameraControlsProps) {
  const { camera, gl } = useThree();
  const controlsRef = useRef<OrbitControls | null>(null);
  const lastCommandIdRef = useRef<number>(-1);
  const lastCameraStateRef = useRef<PreviewCameraState | null>(null);
  const emissionAccumulatorRef = useRef<number>(0);

  useEffect(() => {
    const controls = new OrbitControls(camera, gl.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controlsRef.current = controls;
    return () => {
      controls.dispose();
      controlsRef.current = null;
    };
  }, [camera, gl.domElement]);

  useEffect(() => {
    const controls = controlsRef.current;
    if (controls === null || lastCommandIdRef.current === command.id) {
      return;
    }

    const nextState =
      command.type === "frame"
        ? buildDefaultCameraState(data)
        : command.type === "restore"
          ? command.camera
          : (() => {
              const currentTarget = [
                controls.target.x,
                controls.target.y,
                controls.target.z,
              ] as const;
              const delta = new THREE.Vector3()
                .copy(camera.position)
                .sub(controls.target);
              const distance = Math.max(
                delta.length(),
                Math.max(data.bounds.radius, 0.25) * 2.75,
              );
              const axis = axisVectorByView[command.axis];
              return {
                position: [
                  currentTarget[0] + axis[0] * distance,
                  currentTarget[1] + axis[1] * distance,
                  currentTarget[2] + axis[2] * distance,
                ] as const,
                target: currentTarget,
              };
            })();

    camera.position.set(...nextState.position);
    controls.target.set(...nextState.target);
    camera.lookAt(...nextState.target);
    camera.updateProjectionMatrix();
    controls.update();
    lastCameraStateRef.current = nextState;
    onCameraStateChange(nextState);
    lastCommandIdRef.current = command.id;
  }, [camera, command, data, onCameraStateChange]);

  useFrame((_, delta) => {
    const controls = controlsRef.current;
    if (controls === null) {
      return;
    }
    controls.update();
    emissionAccumulatorRef.current += delta;
    if (emissionAccumulatorRef.current < 0.1) {
      return;
    }

    emissionAccumulatorRef.current = 0;
    const nextState: PreviewCameraState = {
      position: [camera.position.x, camera.position.y, camera.position.z],
      target: [controls.target.x, controls.target.y, controls.target.z],
    };
    const previousState = lastCameraStateRef.current;
    if (
      previousState !== null &&
      vectorTriplesClose(previousState.position, nextState.position) &&
      vectorTriplesClose(previousState.target, nextState.target)
    ) {
      return;
    }
    lastCameraStateRef.current = nextState;
    onCameraStateChange(nextState);
  });

  return null;
}

function FpsSampler({
  onFramesPerSecondChange,
}: Pick<PointCloudPreviewSceneProps, "onFramesPerSecondChange">) {
  const elapsedRef = useRef<number>(0);
  const framesRef = useRef<number>(0);

  useFrame((_, delta) => {
    elapsedRef.current += delta;
    framesRef.current += 1;
    if (elapsedRef.current < 0.4) {
      return;
    }
    onFramesPerSecondChange(framesRef.current / elapsedRef.current);
    elapsedRef.current = 0;
    framesRef.current = 0;
  });

  return null;
}

function SceneContent({
  data,
  showMesh,
  showPoints,
  showWireframe,
  meshColorMode,
  showNeighborDebug,
  pointSize,
  pointGammaCorrection,
  brightness,
  highlightSample,
  transformMatrix,
  inverseTransformMatrix,
  onHoverSampleChange,
}: SceneContentProps) {
  const displayMeshVertexColors = useMemo(
    () =>
      adjustDisplayColors(
        data.meshVertexColors,
        pointGammaCorrection && meshColorMode === "baked",
        brightness,
      ),
    [brightness, data.meshVertexColors, meshColorMode, pointGammaCorrection],
  );

  const displayPointColors = useMemo(
    () =>
      adjustDisplayColors(data.pointColors, pointGammaCorrection, brightness),
    [brightness, data.pointColors, pointGammaCorrection],
  );

  const meshGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(data.meshPositions, 3),
    );
    geometry.setAttribute(
      "color",
      new THREE.BufferAttribute(displayMeshVertexColors, 3),
    );
    geometry.setIndex(new THREE.BufferAttribute(data.meshIndices, 1));
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    return geometry;
  }, [data.meshIndices, data.meshPositions, displayMeshVertexColors]);

  const pointGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(data.pointPositions, 3),
    );
    geometry.setAttribute(
      "color",
      new THREE.BufferAttribute(displayPointColors, 3),
    );
    geometry.computeBoundingSphere();
    return geometry;
  }, [data.pointPositions, displayPointColors]);

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
        brightness: { value: brightness },
        gammaDecodeEnabled: {
          value: pointGammaCorrection ? 1 : 0,
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
    brightness,
    fragmentShaderCellOffsetTexture,
    fragmentShaderPointColorTexture,
    fragmentShaderPointPositionTexture,
    meshColorMode,
    pointGammaCorrection,
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

  const markerRadius = Math.max(data.bounds.radius * 0.018, 0.004);
  const effectivePointSize = Math.max(data.bounds.radius * pointSize, 0.008);
  const displayHighlightColor = useMemo(
    () =>
      highlightSample === null
        ? null
        : adjustDisplayTuple(
            highlightSample.color,
            pointGammaCorrection,
            brightness,
          ),
    [brightness, highlightSample, pointGammaCorrection],
  );
  const displayNeighborColors = useMemo(
    () =>
      highlightSample?.neighbors.map((neighbor) =>
        adjustDisplayTuple(neighbor.color, pointGammaCorrection, brightness),
      ) ?? [],
    [brightness, highlightSample?.neighbors, pointGammaCorrection],
  );

  const handleIntersection = (event: ThreeEvent<PointerEvent>): void => {
    event.stopPropagation();
    const worldPoint = new THREE.Vector3(
      event.point.x,
      event.point.y,
      event.point.z,
    );
    const localPoint = worldPoint.applyMatrix4(inverseTransformMatrix);
    const hitPoint = [localPoint.x, localPoint.y, localPoint.z] as const;
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
      <group matrixAutoUpdate={false} matrix={transformMatrix}>
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
              <meshBasicMaterial color={displayHighlightColor ?? "#ffffff"} />
            </mesh>
            {highlightSample.neighbors.map((neighbor, index) => (
              <mesh key={neighbor.index} position={neighbor.position}>
                <sphereGeometry args={[markerRadius * 0.78, 16, 16]} />
                <meshBasicMaterial
                  color={displayNeighborColors[index] ?? "#ffffff"}
                />
              </mesh>
            ))}
            {debugLineGeometry !== null ? (
              <lineSegments geometry={debugLineGeometry}>
                <lineBasicMaterial color="#f8fafc" transparent opacity={0.75} />
              </lineSegments>
            ) : null}
          </>
        ) : null}
      </group>
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
  pointGammaCorrection,
  brightness,
  swapYZ,
  selectedHit,
  cameraCommand,
  onHoverSampleChange,
  onCameraStateChange,
  onFramesPerSecondChange,
}: PointCloudPreviewSceneProps) {
  const transformMatrix = useMemo(() => {
    if (!swapYZ) {
      return new THREE.Matrix4().identity();
    }
    const translateToOrigin = new THREE.Matrix4().makeTranslation(
      -data.bounds.center[0],
      -data.bounds.center[1],
      -data.bounds.center[2],
    );
    const swapMatrix = new THREE.Matrix4().set(
      1,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      1,
    );
    const translateBack = new THREE.Matrix4().makeTranslation(
      data.bounds.center[0],
      data.bounds.center[1],
      data.bounds.center[2],
    );
    return new THREE.Matrix4()
      .multiplyMatrices(translateBack, swapMatrix)
      .multiply(translateToOrigin);
  }, [data.bounds.center, swapYZ]);

  const inverseTransformMatrix = useMemo(
    () => transformMatrix.clone().invert(),
    [transformMatrix],
  );

  return (
    <div
      data-testid="pointcloud-preview-canvas"
      className="h-[28rem] overflow-hidden rounded-[2rem] border border-white/10 bg-slate-950/60 shadow-2xl shadow-black/35 lg:h-[38rem]"
    >
      <Canvas
        camera={{ fov: 42, near: 0.01, far: 1000, position: [3, 2, 3] }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: false, preserveDrawingBuffer: true }}
      >
        <OrbitCameraControls
          data={data}
          command={cameraCommand}
          onCameraStateChange={onCameraStateChange}
        />
        <FpsSampler onFramesPerSecondChange={onFramesPerSecondChange} />
        <SceneContent
          data={data}
          showMesh={showMesh}
          showPoints={showPoints}
          showWireframe={showWireframe}
          meshColorMode={meshColorMode}
          showNeighborDebug={showNeighborDebug}
          pointSize={pointSize}
          pointGammaCorrection={pointGammaCorrection}
          brightness={brightness}
          swapYZ={swapYZ}
          highlightSample={selectedHit}
          transformMatrix={transformMatrix}
          inverseTransformMatrix={inverseTransformMatrix}
          onHoverSampleChange={onHoverSampleChange}
          onCameraStateChange={onCameraStateChange}
          onFramesPerSecondChange={onFramesPerSecondChange}
        />
      </Canvas>
    </div>
  );
}
