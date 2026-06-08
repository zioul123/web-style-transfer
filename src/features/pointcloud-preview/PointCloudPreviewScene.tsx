import {
  Canvas,
  useFrame,
  useThree,
  type ThreeEvent,
} from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import * as THREE from "three";
import {
  pointColorAt,
  pointPositionAt,
  sampleInterpolatedColor,
} from "./math/interpolation";
import type { PointCloudHitSample, PointCloudMeshData } from "./types";

type PointCloudPreviewSceneProps = {
  readonly data: PointCloudMeshData;
  readonly showMesh: boolean;
  readonly showPoints: boolean;
  readonly showWireframe: boolean;
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
  const controls = useMemo(
    () => new OrbitControls(camera, gl.domElement),
    [camera, gl.domElement],
  );

  useEffect(() => {
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(target[0], target[1], target[2]);
    controls.update();
    return () => {
      controls.dispose();
    };
  }, [controls, target]);

  useFrame(() => {
    controls.update();
  });

  return null;
}

type CameraFramerProps = {
  readonly data: PointCloudMeshData;
};

function CameraFramer({ data }: CameraFramerProps) {
  const { camera } = useThree();

  useEffect(() => {
    const perspectiveCamera = camera as THREE.PerspectiveCamera;
    const radius = Math.max(data.bounds.radius, 0.25);
    const distance = radius * 2.75;
    perspectiveCamera.position.set(
      data.bounds.center[0] + distance,
      data.bounds.center[1] + distance * 0.55,
      data.bounds.center[2] + distance,
    );
    perspectiveCamera.near = Math.max(radius / 200, 0.01);
    perspectiveCamera.far = Math.max(radius * 40, 50);
    perspectiveCamera.lookAt(...data.bounds.center);
    perspectiveCamera.updateProjectionMatrix();
  }, [camera, data.bounds]);

  return null;
}

type SceneContentProps = Omit<PointCloudPreviewSceneProps, "selectedHit"> & {
  readonly highlightSample: PointCloudHitSample | null;
};

function SceneContent({
  data,
  showMesh,
  showPoints,
  showWireframe,
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

  useEffect(
    () => () => {
      meshGeometry.dispose();
      pointGeometry.dispose();
      debugLineGeometry?.dispose();
    },
    [debugLineGeometry, meshGeometry, pointGeometry],
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
          <meshBasicMaterial
            side={THREE.DoubleSide}
            vertexColors
            wireframe={showWireframe}
          />
        </mesh>
      ) : null}
      {showPoints ? (
        <points geometry={pointGeometry}>
          <pointsMaterial
            size={effectivePointSize}
            sizeAttenuation
            vertexColors
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
          showNeighborDebug={showNeighborDebug}
          pointSize={pointSize}
          highlightSample={selectedHit}
          onHoverSampleChange={onHoverSampleChange}
        />
      </Canvas>
    </div>
  );
}
