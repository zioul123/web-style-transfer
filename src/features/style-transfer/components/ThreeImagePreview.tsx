import { Canvas, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

export type PreviewMode = "classic" | "r3f-row" | "r3f-overlay";

export type PreviewImage = {
  readonly key: "content" | "style" | "output";
  readonly label: "Content" | "Style" | "Output";
  readonly src: string | null;
  readonly resolution: {
    readonly width: number;
    readonly height: number;
  };
  readonly scale: number;
  readonly tint: string;
};

function useStableTexture(src: string | null): THREE.Texture | null {
  const [texture, setTexture] = useState<THREE.Texture | null>(null);
  const textureRef = useRef<THREE.Texture | null>(null);

  useEffect(() => {
    if (src === null) {
      textureRef.current?.dispose();
      textureRef.current = null;
      return;
    }

    let isCancelled = false;
    const loader = new THREE.TextureLoader();
    loader.load(src, (nextTexture) => {
      nextTexture.colorSpace = THREE.SRGBColorSpace;
      nextTexture.minFilter = THREE.LinearFilter;
      nextTexture.magFilter = THREE.LinearFilter;
      nextTexture.needsUpdate = true;
      if (isCancelled) {
        nextTexture.dispose();
        return;
      }
      textureRef.current?.dispose();
      textureRef.current = nextTexture;
      setTexture(nextTexture);
    });

    return () => {
      isCancelled = true;
    };
  }, [src]);

  useEffect(
    () => () => {
      textureRef.current?.dispose();
      textureRef.current = null;
    },
    [],
  );

  return texture;
}

type ImagePlaneProps = {
  readonly image: PreviewImage;
  readonly position: readonly [number, number, number];
  readonly opacity: number;
};

function ImagePlane({ image, position, opacity }: ImagePlaneProps) {
  const stableTexture = useStableTexture(image.src);
  const texture = image.src === null ? null : stableTexture;
  const width = image.resolution.width * image.scale;
  const height = image.resolution.height * image.scale;

  if (texture === null) {
    return (
      <mesh position={position}>
        <planeGeometry args={[width, height]} />
        <meshBasicMaterial color="#0f172a" transparent opacity={0.72} />
      </mesh>
    );
  }

  return (
    <mesh position={position}>
      <planeGeometry args={[width, height]} />
      <meshBasicMaterial map={texture} transparent opacity={opacity} />
    </mesh>
  );
}

type PreviewLabelProps = {
  readonly image: PreviewImage;
  readonly x: number;
  readonly y: number;
};

function PreviewLabel({ image, x, y }: PreviewLabelProps) {
  return (
    <div
      className="pointer-events-none absolute rounded-full border border-white/15 bg-slate-950/80 px-2.5 py-1 text-xs font-semibold text-slate-100 shadow-lg shadow-black/30"
      style={{
        left: `${x}%`,
        top: `${y}%`,
        transform: "translate(-50%, -50%)",
      }}
    >
      <span
        className="mr-1 inline-block h-2 w-2 rounded-full"
        style={{ backgroundColor: image.tint }}
      />
      {image.label} · {image.resolution.width}×{image.resolution.height}
    </div>
  );
}

type PreviewSceneProps = {
  readonly bounds: {
    readonly width: number;
    readonly height: number;
  };
  readonly images: readonly PreviewImage[];
  readonly mode: Exclude<PreviewMode, "classic">;
  readonly positions: readonly (readonly [number, number, number])[];
  readonly zoom: number;
};

function PreviewScene({
  bounds,
  images,
  mode,
  positions,
  zoom,
}: PreviewSceneProps) {
  const { size } = useThree();
  const sceneScale = useMemo(() => {
    const margin = 1.22;
    const fitX = size.width / Math.max(1, bounds.width * margin);
    const fitY = size.height / Math.max(1, bounds.height * margin);
    return Math.max(0.25, Math.min(fitX, fitY) * zoom);
  }, [bounds.height, bounds.width, size.height, size.width, zoom]);

  return (
    <group scale={[sceneScale, sceneScale, 1]}>
      <gridHelper
        args={[512, 16, "#1e3a8a", "#1e293b"]}
        rotation={[Math.PI / 2, 0, 0]}
      />
      {images.map((image, index) => (
        <ImagePlane
          key={image.key}
          image={image}
          position={positions[index]}
          opacity={mode === "r3f-overlay" ? 0.78 : 1}
        />
      ))}
    </group>
  );
}

type ThreeImagePreviewProps = {
  readonly mode: Exclude<PreviewMode, "classic">;
  readonly images: readonly PreviewImage[];
  readonly zoom: number;
};

export function ThreeImagePreview({
  mode,
  images,
  zoom,
}: ThreeImagePreviewProps) {
  const layout = useMemo(() => {
    if (mode === "r3f-overlay") {
      const maxWidth = Math.max(
        ...images.map((image) => image.resolution.width * image.scale),
      );
      const maxHeight = Math.max(
        ...images.map((image) => image.resolution.height * image.scale),
      );
      return {
        bounds: { width: maxWidth, height: maxHeight },
        positions: images.map(
          (_image, index) => [index * 12 - 12, 18 - index * 18, index] as const,
        ),
      };
    }

    const gap = 36;
    const widths = images.map((image) => image.resolution.width * image.scale);
    const heights = images.map(
      (image) => image.resolution.height * image.scale,
    );
    const totalWidth =
      widths.reduce((sum, width) => sum + width, 0) + gap * (images.length - 1);
    let cursor = -totalWidth / 2;
    return {
      bounds: { width: totalWidth, height: Math.max(...heights) },
      positions: images.map((_image, index) => {
        const width = widths[index];
        const x = cursor + width / 2;
        cursor += width + gap;
        return [x, 0, index] as const;
      }),
    };
  }, [images, mode]);

  return (
    <div className="relative h-[26rem] overflow-hidden rounded-lg border border-white/10 bg-[radial-gradient(circle_at_center,_rgba(56,189,248,0.16),_transparent_30rem),#020617]">
      <Canvas
        orthographic
        camera={{ position: [0, 0, 1000], near: 0.1, far: 2000, zoom: 1 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: true }}
      >
        <color attach="background" args={["#020617"]} />
        <PreviewScene
          bounds={layout.bounds}
          images={images}
          mode={mode}
          positions={layout.positions}
          zoom={zoom}
        />
      </Canvas>
      {mode === "r3f-row" ? (
        <>
          <PreviewLabel image={images[0]} x={18} y={91} />
          <PreviewLabel image={images[1]} x={50} y={91} />
          <PreviewLabel image={images[2]} x={82} y={91} />
        </>
      ) : (
        <>
          <PreviewLabel image={images[0]} x={18} y={16} />
          <PreviewLabel image={images[1]} x={50} y={16} />
          <PreviewLabel image={images[2]} x={82} y={16} />
        </>
      )}
    </div>
  );
}
