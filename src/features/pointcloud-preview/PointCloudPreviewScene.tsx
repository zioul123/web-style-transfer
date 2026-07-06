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
  ConvolutionKernelHitSample,
  ConvolutionKernelLevelData,
  ConvolutionKernelPathGroup,
  MeshColorMode,
  PointCloudHitSample,
  PointCloudMeshData,
  PointCloudPreviewBackgroundColor,
  PointCloudPreviewCameraCommand,
  PointCloudPreviewRenderMode,
  PointCloudPreviewViewAxis,
  PreviewCameraState,
} from "./types";

export const maxFragmentShaderPointsPerCell = 256;

type PointCloudPreviewSceneProps = {
  readonly data: PointCloudMeshData;
  readonly renderMode: PointCloudPreviewRenderMode;
  readonly showMesh: boolean;
  readonly showPoints: boolean;
  readonly showWireframe: boolean;
  readonly showSolidMesh: boolean;
  readonly renderPointsAsSpheres: boolean;
  readonly showGroundPlane: boolean;
  readonly meshColorMode: MeshColorMode;
  readonly kernelLevelIndex: number;
  readonly showNeighborDebug: boolean;
  readonly pointSize: number;
  readonly backgroundColor: PointCloudPreviewBackgroundColor;
  readonly pointGammaCorrection: boolean;
  readonly brightness: number;
  readonly swapYZ: boolean;
  readonly selectedHit: PointCloudHitSample | null;
  readonly selectedKernelHit: ConvolutionKernelHitSample | null;
  readonly cameraCommand: PointCloudPreviewCameraCommand;
  readonly onHoverSampleChange: (sample: PointCloudHitSample | null) => void;
  readonly onHoverKernelSampleChange: (
    sample: ConvolutionKernelHitSample | null,
  ) => void;
  readonly onCameraStateChange: (state: PreviewCameraState) => void;
  readonly onCameraCommandApplied?: (commandId: number) => void;
  readonly onFrameRendered?: (commandId: number) => void;
  readonly onFramesPerSecondChange: (fps: number) => void;
};

type OrbitCameraControlsProps = {
  readonly data: PointCloudMeshData;
  readonly command: PointCloudPreviewCameraCommand;
  readonly onCameraStateChange: (state: PreviewCameraState) => void;
  readonly onCameraCommandApplied?: (commandId: number) => void;
};

type SceneContentProps = Omit<
  PointCloudPreviewSceneProps,
  "selectedHit" | "selectedKernelHit" | "cameraCommand"
> & {
  readonly highlightSample: PointCloudHitSample | null;
  readonly highlightKernelSample: ConvolutionKernelHitSample | null;
  readonly transformMatrix: THREE.Matrix4;
  readonly inverseTransformMatrix: THREE.Matrix4;
};

const minimumSquaredDistance = 1e-16;
const fragmentShaderNeighborCellRadius = 1;
const textureMaxWidth = 2048;
const cameraStateEpsilon = 1e-5;
const backgroundColors: Record<PointCloudPreviewBackgroundColor, string> = {
  default: "#09111f",
  black: "#000000",
  white: "#ffffff",
};

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

const axisVectorByView: Record<
  PointCloudPreviewViewAxis,
  readonly [number, number, number]
> = {
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

const findKernelLevel = (
  levels: readonly ConvolutionKernelLevelData[],
  levelIndex: number,
): ConvolutionKernelLevelData | null =>
  levels.find((level) => level.levelIndex === levelIndex) ?? levels[0] ?? null;

const buildKernelPathPositions = (
  group: ConvolutionKernelPathGroup,
): {
  readonly linePositions: Float32Array;
  readonly pointPositions: Float32Array;
} => {
  const pointCount = group.reduce((sum, path) => sum + path.length, 0);
  const segmentCount = group.reduce(
    (sum, path) => sum + Math.max(0, path.length - 1),
    0,
  );
  const pointPositions = new Float32Array(pointCount * 3);
  const linePositions = new Float32Array(segmentCount * 6);
  let pointOffset = 0;
  let lineOffset = 0;

  group.forEach((path) => {
    path.forEach((point, pointIndex) => {
      pointPositions[pointOffset] = point[0];
      pointPositions[pointOffset + 1] = point[1];
      pointPositions[pointOffset + 2] = point[2];
      pointOffset += 3;

      if (pointIndex === 0) {
        return;
      }

      const previousPoint = path[pointIndex - 1];
      linePositions[lineOffset] = previousPoint[0];
      linePositions[lineOffset + 1] = previousPoint[1];
      linePositions[lineOffset + 2] = previousPoint[2];
      linePositions[lineOffset + 3] = point[0];
      linePositions[lineOffset + 4] = point[1];
      linePositions[lineOffset + 5] = point[2];
      lineOffset += 6;
    });
  });

  return { linePositions, pointPositions };
};

function OrbitCameraControls({
  data,
  command,
  onCameraStateChange,
  onCameraCommandApplied,
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

  useFrame((_, delta) => {
    const controls = controlsRef.current;
    if (controls === null) {
      return;
    }

    if (lastCommandIdRef.current !== command.id) {
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
                const deltaFromTarget = new THREE.Vector3()
                  .copy(camera.position)
                  .sub(controls.target);
                const distance = Math.max(
                  deltaFromTarget.length(),
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
      onCameraCommandApplied?.(command.id);
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

function FrameRenderedNotifier({
  commandId,
  onFrameRendered,
}: {
  readonly commandId: number;
  readonly onFrameRendered?: (commandId: number) => void;
}) {
  useFrame(() => {
    onFrameRendered?.(commandId);
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

type InstancedSphereMarkersProps = {
  readonly positions: Float32Array;
  readonly colors?: Float32Array;
  readonly color?: string;
  readonly radius: number;
  readonly opacity?: number;
  readonly depthTest?: boolean;
  readonly depthWrite?: boolean;
  readonly renderOrder?: number;
  readonly onPointerMove?: (event: ThreeEvent<PointerEvent>) => void;
  readonly onClick?: (event: ThreeEvent<PointerEvent>) => void;
  readonly onPointerOut?: () => void;
};

function InstancedSphereMarkers({
  positions,
  colors,
  color = "#ffffff",
  radius,
  opacity = 1,
  depthTest = true,
  depthWrite = false,
  renderOrder = 0,
  onPointerMove,
  onClick,
  onPointerOut,
}: InstancedSphereMarkersProps) {
  const mesh = useMemo(() => {
    const count = positions.length / 3;
    if (count === 0) {
      return null;
    }

    const geometry = new THREE.SphereGeometry(radius, 12, 8);
    const material = new THREE.MeshBasicMaterial({
      color,
      depthTest,
      depthWrite,
      opacity,
      transparent: opacity < 1,
    });
    const instancedMesh = new THREE.InstancedMesh(geometry, material, count);
    instancedMesh.renderOrder = renderOrder;

    const matrix = new THREE.Matrix4();
    const instanceColor = new THREE.Color();
    for (let index = 0; index < count; index += 1) {
      const offset = index * 3;
      matrix.makeTranslation(
        positions[offset],
        positions[offset + 1],
        positions[offset + 2],
      );
      instancedMesh.setMatrixAt(index, matrix);
      if (colors !== undefined) {
        instanceColor.setRGB(
          colors[offset] ?? 1,
          colors[offset + 1] ?? 1,
          colors[offset + 2] ?? 1,
        );
        instancedMesh.setColorAt(index, instanceColor);
      }
    }
    instancedMesh.instanceMatrix.needsUpdate = true;
    if (instancedMesh.instanceColor !== null) {
      instancedMesh.instanceColor.needsUpdate = true;
    }
    return instancedMesh;
  }, [
    color,
    colors,
    depthTest,
    depthWrite,
    opacity,
    positions,
    radius,
    renderOrder,
  ]);

  useEffect(
    () => () => {
      mesh?.geometry.dispose();
      const material = mesh?.material;
      if (Array.isArray(material)) {
        material.forEach((entry) => entry.dispose());
      } else {
        material?.dispose();
      }
    },
    [mesh],
  );

  if (mesh === null) {
    return null;
  }

  return (
    <primitive
      object={mesh}
      onPointerMove={onPointerMove}
      onClick={onClick}
      onPointerOut={onPointerOut}
    />
  );
}

function InstancedCylinderSegments({
  linePositions,
  color,
  radius,
  depthTest,
  renderOrder,
}: {
  readonly linePositions: Float32Array;
  readonly color: string;
  readonly radius: number;
  readonly depthTest: boolean;
  readonly renderOrder: number;
}) {
  const mesh = useMemo(() => {
    const count = linePositions.length / 6;
    if (count === 0) {
      return null;
    }

    const geometry = new THREE.CylinderGeometry(1, 1, 1, 10);
    const material = new THREE.MeshBasicMaterial({
      color,
      depthTest,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    const instancedMesh = new THREE.InstancedMesh(geometry, material, count);
    instancedMesh.renderOrder = renderOrder;

    const yAxis = new THREE.Vector3(0, 1, 0);
    const start = new THREE.Vector3();
    const end = new THREE.Vector3();
    const midpoint = new THREE.Vector3();
    const direction = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    const matrix = new THREE.Matrix4();

    for (let index = 0; index < count; index += 1) {
      const offset = index * 6;
      start.set(
        linePositions[offset],
        linePositions[offset + 1],
        linePositions[offset + 2],
      );
      end.set(
        linePositions[offset + 3],
        linePositions[offset + 4],
        linePositions[offset + 5],
      );
      direction.copy(end).sub(start);
      const length = direction.length();
      if (length <= minimumSquaredDistance) {
        scale.set(0, 0, 0);
        matrix.compose(start, quaternion.identity(), scale);
        instancedMesh.setMatrixAt(index, matrix);
        continue;
      }

      midpoint.copy(start).add(end).multiplyScalar(0.5);
      quaternion.setFromUnitVectors(yAxis, direction.normalize());
      scale.set(radius, length, radius);
      matrix.compose(midpoint, quaternion, scale);
      instancedMesh.setMatrixAt(index, matrix);
    }
    instancedMesh.instanceMatrix.needsUpdate = true;
    return instancedMesh;
  }, [color, depthTest, linePositions, radius, renderOrder]);

  useEffect(
    () => () => {
      mesh?.geometry.dispose();
      const material = mesh?.material;
      if (Array.isArray(material)) {
        material.forEach((entry) => entry.dispose());
      } else {
        material?.dispose();
      }
    },
    [mesh],
  );

  return mesh === null ? null : <primitive object={mesh} />;
}

function SceneContent({
  data,
  renderMode,
  showMesh,
  showPoints,
  showWireframe,
  showSolidMesh,
  renderPointsAsSpheres,
  showGroundPlane,
  meshColorMode,
  kernelLevelIndex,
  showNeighborDebug,
  pointSize,
  backgroundColor,
  pointGammaCorrection,
  brightness,
  highlightSample,
  highlightKernelSample,
  transformMatrix,
  inverseTransformMatrix,
  onHoverSampleChange,
  onHoverKernelSampleChange,
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

  const activeKernelLevel = useMemo(
    () =>
      renderMode === "kernels"
        ? findKernelLevel(data.convolutionKernelLevels, kernelLevelIndex)
        : null,
    [data.convolutionKernelLevels, kernelLevelIndex, renderMode],
  );

  const kernelAnchorGeometry = useMemo(() => {
    if (activeKernelLevel === null) {
      return null;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(activeKernelLevel.anchorPositions, 3),
    );
    geometry.computeBoundingSphere();
    return geometry;
  }, [activeKernelLevel]);

  const selectedKernelGroup = useMemo(() => {
    if (
      activeKernelLevel === null ||
      highlightKernelSample === null ||
      highlightKernelSample.levelIndex !== activeKernelLevel.levelIndex
    ) {
      return null;
    }
    return activeKernelLevel.groups[highlightKernelSample.groupIndex] ?? null;
  }, [activeKernelLevel, highlightKernelSample]);

  const selectedKernelPathPositions = useMemo(
    () =>
      selectedKernelGroup === null
        ? {
            linePositions: new Float32Array(),
            pointPositions: new Float32Array(),
          }
        : buildKernelPathPositions(selectedKernelGroup),
    [selectedKernelGroup],
  );

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
      wireframe: showWireframe && !showSolidMesh,
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
    showSolidMesh,
    showWireframe,
  ]);

  useEffect(
    () => () => {
      meshGeometry.dispose();
      pointGeometry.dispose();
      debugLineGeometry?.dispose();
      kernelAnchorGeometry?.dispose();
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
      kernelAnchorGeometry,
      meshGeometry,
      pointGeometry,
    ],
  );

  const sceneRadius = Math.max(data.bounds.radius, 0.25);
  const fogNear = sceneRadius * 4;
  const fogFar = sceneRadius * 12;
  const effectivePointSize = data.bounds.radius * pointSize;
  const effectivePointSphereRadius = effectivePointSize * 0.5;
  const effectiveKernelAnchorSize = data.bounds.radius * pointSize;
  const effectiveKernelAnchorSphereRadius = effectiveKernelAnchorSize * 0.5;
  const effectiveKernelPathPointSize = effectiveKernelAnchorSize * 0.72;
  const effectiveKernelPathSphereRadius = effectiveKernelPathPointSize * 0.5;
  const effectiveKernelPathCylinderRadius =
    effectiveKernelPathSphereRadius * 0.45;
  const sceneBackgroundColor = backgroundColors[backgroundColor];
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

  const handleKernelAnchorIntersection = (
    event: ThreeEvent<PointerEvent>,
  ): void => {
    event.stopPropagation();
    const groupIndex =
      typeof event.instanceId === "number"
        ? event.instanceId
        : typeof event.index === "number"
          ? event.index
          : -1;
    if (activeKernelLevel === null || groupIndex < 0) {
      return;
    }

    if (groupIndex < 0 || groupIndex >= activeKernelLevel.groupCount) {
      return;
    }

    const pointIndex = groupIndex * 3;
    const group = activeKernelLevel.groups[groupIndex];
    onHoverKernelSampleChange({
      levelIndex: activeKernelLevel.levelIndex,
      groupIndex,
      point: [
        activeKernelLevel.anchorPositions[pointIndex],
        activeKernelLevel.anchorPositions[pointIndex + 1],
        activeKernelLevel.anchorPositions[pointIndex + 2],
      ],
      pathCount: group?.length ?? 0,
    });
  };

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
      <color attach="background" args={[sceneBackgroundColor]} />
      <fog attach="fog" args={[sceneBackgroundColor, fogNear, fogFar]} />
      <group matrixAutoUpdate={false} matrix={transformMatrix}>
        {showGroundPlane ? (
          <gridHelper
            args={[data.bounds.radius * 6, 12, "#164e63", "#0f172a"]}
            position={[
              data.bounds.center[0],
              data.bounds.min[1],
              data.bounds.center[2],
            ]}
          />
        ) : null}
        {renderMode === "kernels" ? (
          <>
            {showMesh ? (
              <mesh geometry={meshGeometry}>
                <meshBasicMaterial
                  color={showSolidMesh ? "#64748b" : "#94a3b8"}
                  side={showSolidMesh ? THREE.FrontSide : THREE.DoubleSide}
                  wireframe={showWireframe && !showSolidMesh}
                  transparent={!showSolidMesh}
                  opacity={showSolidMesh ? 1 : showWireframe ? 0.85 : 0.38}
                  depthWrite={showSolidMesh}
                  polygonOffset
                  polygonOffsetFactor={1}
                  polygonOffsetUnits={1}
                />
              </mesh>
            ) : null}
            {showMesh && showWireframe && showSolidMesh ? (
              <mesh geometry={meshGeometry} renderOrder={1}>
                <meshBasicMaterial
                  color="#e2e8f0"
                  side={THREE.FrontSide}
                  wireframe
                  transparent
                  opacity={0.45}
                  depthWrite={false}
                  depthTest
                  polygonOffset
                  polygonOffsetFactor={-1}
                  polygonOffsetUnits={-1}
                />
              </mesh>
            ) : null}
            {showPoints && kernelAnchorGeometry !== null ? (
              renderPointsAsSpheres && activeKernelLevel !== null ? (
                <InstancedSphereMarkers
                  positions={activeKernelLevel.anchorPositions}
                  color="#f59e0b"
                  radius={effectiveKernelAnchorSphereRadius}
                  opacity={0.28}
                  depthWrite={false}
                  depthTest={showSolidMesh}
                  renderOrder={2}
                  onPointerMove={handleKernelAnchorIntersection}
                  onClick={handleKernelAnchorIntersection}
                  onPointerOut={() => onHoverKernelSampleChange(null)}
                />
              ) : (
                <points
                  geometry={kernelAnchorGeometry}
                  renderOrder={2}
                  onPointerMove={handleKernelAnchorIntersection}
                  onClick={handleKernelAnchorIntersection}
                  onPointerOut={() => onHoverKernelSampleChange(null)}
                >
                  <pointsMaterial
                    color="#f59e0b"
                    size={effectiveKernelAnchorSize}
                    sizeAttenuation
                    transparent
                    opacity={0.28}
                    depthWrite={false}
                    depthTest={showSolidMesh}
                  />
                </points>
              )
            ) : null}
            {highlightKernelSample !== null ? (
              <>
                <mesh position={highlightKernelSample.point} renderOrder={4}>
                  <sphereGeometry
                    args={[effectiveKernelAnchorSphereRadius, 16, 12]}
                  />
                  <meshBasicMaterial
                    color="#f59e0b"
                    depthTest={showSolidMesh}
                  />
                </mesh>
                {selectedKernelPathPositions.linePositions.length > 0 ? (
                  <InstancedCylinderSegments
                    linePositions={selectedKernelPathPositions.linePositions}
                    color="#06b6d4"
                    radius={effectiveKernelPathCylinderRadius}
                    depthTest={showSolidMesh}
                    renderOrder={3}
                  />
                ) : null}
                {selectedKernelPathPositions.pointPositions.length > 0 ? (
                  <InstancedSphereMarkers
                    positions={selectedKernelPathPositions.pointPositions}
                    color="#f8fafc"
                    radius={effectiveKernelPathSphereRadius}
                    depthWrite={false}
                    depthTest={showSolidMesh}
                    renderOrder={3}
                  />
                ) : null}
              </>
            ) : null}
          </>
        ) : (
          <>
            {showMesh ? (
              <mesh
                geometry={meshGeometry}
                onPointerMove={handleIntersection}
                onClick={handleIntersection}
                onPointerOut={() => onHoverSampleChange(null)}
              >
                {fragmentShaderMaterial !== null ? (
                  <primitive
                    object={fragmentShaderMaterial}
                    attach="material"
                  />
                ) : (
                  <meshBasicMaterial
                    side={THREE.DoubleSide}
                    vertexColors
                    wireframe={showWireframe && !showSolidMesh}
                    polygonOffset
                    polygonOffsetFactor={1}
                    polygonOffsetUnits={1}
                  />
                )}
              </mesh>
            ) : null}
            {showMesh && showWireframe && showSolidMesh ? (
              <mesh geometry={meshGeometry} renderOrder={1}>
                <meshBasicMaterial
                  color="#0f172a"
                  side={THREE.DoubleSide}
                  wireframe
                  transparent
                  opacity={0.45}
                  depthWrite={false}
                  depthTest
                  polygonOffset
                  polygonOffsetFactor={-1}
                  polygonOffsetUnits={-1}
                />
              </mesh>
            ) : null}
            {showPoints ? (
              renderPointsAsSpheres ? (
                <InstancedSphereMarkers
                  positions={data.pointPositions}
                  colors={displayPointColors}
                  radius={effectivePointSphereRadius}
                  depthWrite={false}
                  renderOrder={1}
                />
              ) : (
                <points geometry={pointGeometry} renderOrder={1}>
                  <pointsMaterial
                    size={effectivePointSize}
                    sizeAttenuation
                    vertexColors
                    depthWrite={false}
                  />
                </points>
              )
            ) : null}
            {showNeighborDebug && highlightSample !== null ? (
              <>
                <mesh position={highlightSample.point}>
                  <sphereGeometry args={[effectivePointSphereRadius, 16, 12]} />
                  <meshBasicMaterial
                    color={displayHighlightColor ?? "#ffffff"}
                  />
                </mesh>
                {highlightSample.neighbors.map((neighbor, index) => (
                  <mesh key={neighbor.index} position={neighbor.position}>
                    <sphereGeometry
                      args={[effectivePointSphereRadius * 0.78, 14, 10]}
                    />
                    <meshBasicMaterial
                      color={displayNeighborColors[index] ?? "#ffffff"}
                    />
                  </mesh>
                ))}
                {debugLineGeometry !== null ? (
                  <lineSegments geometry={debugLineGeometry}>
                    <lineBasicMaterial
                      color="#f8fafc"
                      transparent
                      opacity={0.75}
                    />
                  </lineSegments>
                ) : null}
              </>
            ) : null}
          </>
        )}
      </group>
    </>
  );
}

export function PointCloudPreviewScene({
  data,
  renderMode,
  showMesh,
  showPoints,
  showWireframe,
  showSolidMesh,
  renderPointsAsSpheres,
  showGroundPlane,
  meshColorMode,
  kernelLevelIndex,
  showNeighborDebug,
  pointSize,
  backgroundColor,
  pointGammaCorrection,
  brightness,
  swapYZ,
  selectedHit,
  selectedKernelHit,
  cameraCommand,
  onHoverSampleChange,
  onHoverKernelSampleChange,
  onCameraStateChange,
  onCameraCommandApplied,
  onFrameRendered,
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
  const pointRaycastThreshold = Math.max(data.bounds.radius * 0.035, 0.01);

  return (
    <div
      data-testid="pointcloud-preview-canvas"
      className="h-[28rem] overflow-hidden rounded-[2rem] border border-white/10 bg-slate-950/60 shadow-2xl shadow-black/35 lg:h-[38rem]"
    >
      <Canvas
        camera={{ fov: 42, near: 0.01, far: 1000, position: [3, 2, 3] }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: false, preserveDrawingBuffer: true }}
        raycaster={{
          params: {
            Mesh: {},
            Line: { threshold: 1 },
            LOD: {},
            Points: { threshold: pointRaycastThreshold },
            Sprite: {},
          },
        }}
      >
        <OrbitCameraControls
          data={data}
          command={cameraCommand}
          onCameraStateChange={onCameraStateChange}
          onCameraCommandApplied={onCameraCommandApplied}
        />
        <FrameRenderedNotifier
          commandId={cameraCommand.id}
          onFrameRendered={onFrameRendered}
        />
        <FpsSampler onFramesPerSecondChange={onFramesPerSecondChange} />
        <SceneContent
          data={data}
          renderMode={renderMode}
          showMesh={showMesh}
          showPoints={showPoints}
          showWireframe={showWireframe}
          showSolidMesh={showSolidMesh}
          renderPointsAsSpheres={renderPointsAsSpheres}
          showGroundPlane={showGroundPlane}
          meshColorMode={meshColorMode}
          kernelLevelIndex={kernelLevelIndex}
          showNeighborDebug={showNeighborDebug}
          pointSize={pointSize}
          backgroundColor={backgroundColor}
          pointGammaCorrection={pointGammaCorrection}
          brightness={brightness}
          swapYZ={swapYZ}
          highlightSample={selectedHit}
          highlightKernelSample={selectedKernelHit}
          transformMatrix={transformMatrix}
          inverseTransformMatrix={inverseTransformMatrix}
          onHoverSampleChange={onHoverSampleChange}
          onHoverKernelSampleChange={onHoverKernelSampleChange}
          onCameraStateChange={onCameraStateChange}
          onFramesPerSecondChange={onFramesPerSecondChange}
        />
      </Canvas>
    </div>
  );
}
