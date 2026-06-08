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

export const maxFragmentShaderPoints = 512;

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

const fragmentKnnVertexShader = `
  varying vec3 vSamplePosition;

  void main() {
    vSamplePosition = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const fragmentKnnFragmentShader = `
  precision highp float;

  uniform sampler2D pointPositionTexture;
  uniform sampler2D pointColorTexture;
  uniform float pointTextureWidth;
  uniform int pointCount;

  varying vec3 vSamplePosition;

  vec4 readPointTexture(sampler2D textureSampler, int index) {
    float u = (float(index) + 0.5) / pointTextureWidth;
    return texture2D(textureSampler, vec2(u, 0.5));
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

    for (int pointIndex = 0; pointIndex < ${maxFragmentShaderPoints}; pointIndex += 1) {
      if (pointIndex >= pointCount) {
        break;
      }

      vec3 pointPosition = readPointTexture(pointPositionTexture, pointIndex).xyz;
      vec3 pointColor = readPointTexture(pointColorTexture, pointIndex).xyz;
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

const buildPointTexture = (values: Float32Array): THREE.DataTexture => {
  const pointCount = values.length / 3;
  const textureData = new Float32Array(pointCount * 4);
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
    pointCount,
    1,
    THREE.RGBAFormat,
    THREE.FloatType,
  );
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
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
    return buildPointTexture(data.pointPositions);
  }, [data.pointPositions, meshColorMode]);

  const fragmentShaderPointColorTexture = useMemo(() => {
    if (meshColorMode !== "fragment-knn") {
      return null;
    }
    return buildPointTexture(data.pointColors);
  }, [data.pointColors, meshColorMode]);

  const fragmentShaderMaterial = useMemo(() => {
    if (
      meshColorMode !== "fragment-knn" ||
      fragmentShaderPointPositionTexture === null ||
      fragmentShaderPointColorTexture === null
    ) {
      return null;
    }

    return new THREE.ShaderMaterial({
      uniforms: {
        pointPositionTexture: { value: fragmentShaderPointPositionTexture },
        pointColorTexture: { value: fragmentShaderPointColorTexture },
        pointTextureWidth: { value: data.pointCount },
        pointCount: { value: data.pointCount },
      },
      vertexShader: fragmentKnnVertexShader,
      fragmentShader: fragmentKnnFragmentShader,
      side: THREE.DoubleSide,
      wireframe: showWireframe,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });
  }, [
    data.pointCount,
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
      fragmentShaderPointPositionTexture?.dispose();
      fragmentShaderPointColorTexture?.dispose();
      fragmentShaderMaterial?.dispose();
    },
    [
      debugLineGeometry,
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
