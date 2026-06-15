import type { MutableRefObject } from "react";
import { PointCloudHitInspector } from "./PointCloudHitInspector";
import { PointCloudPreviewScene } from "./PointCloudPreviewScene";
import type { LoadedAssetState } from "./pointCloudPreviewModels";
import type {
  MeshColorMode,
  PointCloudHitSample,
  PointCloudPreviewCameraCommand,
  PointCloudPreviewViewSettings,
  PreviewCameraState,
} from "./types";

const fpsLabel = (framesPerSecond: number): string =>
  framesPerSecond > 0 ? `${framesPerSecond.toFixed(1)} FPS` : "Sampling FPS";

type PointCloudPreviewViewportProps = {
  readonly previewHostRef: MutableRefObject<HTMLDivElement | null>;
  readonly assetState: LoadedAssetState;
  readonly viewSettings: PointCloudPreviewViewSettings;
  readonly effectiveMeshColorMode: MeshColorMode;
  readonly selectedHit: PointCloudHitSample | null;
  readonly framesPerSecond: number;
  readonly cameraCommand: PointCloudPreviewCameraCommand;
  readonly canBatchScreenshot: boolean;
  readonly onScreenshot: () => void;
  readonly onOpenBatchScreenshot: () => void;
  readonly onHoverSampleChange: (hit: PointCloudHitSample | null) => void;
  readonly onCameraStateChange: (state: PreviewCameraState) => void;
  readonly onCameraCommandApplied: (commandId: number) => void;
  readonly onFrameRendered: (commandId: number) => void;
  readonly onFramesPerSecondChange: (framesPerSecond: number) => void;
};

export function PointCloudPreviewViewport({
  previewHostRef,
  assetState,
  viewSettings,
  effectiveMeshColorMode,
  selectedHit,
  framesPerSecond,
  cameraCommand,
  canBatchScreenshot,
  onScreenshot,
  onOpenBatchScreenshot,
  onHoverSampleChange,
  onCameraStateChange,
  onCameraCommandApplied,
  onFrameRendered,
  onFramesPerSecondChange,
}: PointCloudPreviewViewportProps) {
  const data = assetState.data;

  return (
    <section className="flex h-[42rem] min-h-0 min-w-0 w-full shrink-0 flex-col rounded-[1.1rem] border border-white/10 bg-slate-950/55 p-4 shadow-xl shadow-black/20 2xl:h-auto 2xl:flex-1">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3 px-1">
        <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
          Interactive preview
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <div
            data-testid="pointcloud-fps"
            className="rounded-xl border border-emerald-300/25 bg-emerald-400/10 px-4 py-2 text-sm font-semibold text-emerald-100"
          >
            {fpsLabel(framesPerSecond)}
          </div>
          <button
            data-testid="screenshot-button"
            className="rounded-xl border border-emerald-300/20 bg-emerald-400/10 px-4 py-2 text-sm font-semibold text-emerald-100 transition hover:bg-emerald-400/20"
            type="button"
            onClick={onScreenshot}
          >
            Screenshot
          </button>
          {canBatchScreenshot ? (
            <button
              data-testid="batch-screenshot-button"
              className="rounded-xl border border-sky-300/20 bg-sky-400/10 px-4 py-2 text-sm font-semibold text-sky-100 transition hover:bg-sky-400/20"
              type="button"
              onClick={onOpenBatchScreenshot}
            >
              Batch screenshot
            </button>
          ) : null}
        </div>
      </div>
      <div ref={previewHostRef} className="relative min-h-0 flex-1">
        {data !== null ? (
          <PointCloudPreviewScene
            data={data}
            showMesh={viewSettings.showMesh}
            showPoints={viewSettings.showPoints}
            showWireframe={viewSettings.showWireframe}
            meshColorMode={effectiveMeshColorMode}
            showNeighborDebug
            pointSize={viewSettings.pointSize}
            backgroundColor={viewSettings.backgroundColor}
            pointGammaCorrection={!viewSettings.disableGammaDecoding}
            brightness={viewSettings.brightness}
            swapYZ={viewSettings.swapYZ}
            selectedHit={selectedHit}
            cameraCommand={cameraCommand}
            onHoverSampleChange={onHoverSampleChange}
            onCameraStateChange={onCameraStateChange}
            onCameraCommandApplied={onCameraCommandApplied}
            onFrameRendered={onFrameRendered}
            onFramesPerSecondChange={onFramesPerSecondChange}
          />
        ) : (
          <div className="flex h-full items-center justify-center rounded-[1.1rem] border border-dashed border-white/10 bg-slate-950/50 px-6 text-center text-sm text-slate-300">
            {assetState.status === "loading"
              ? `Loading ${assetState.sourceLabel}...`
              : "No point-cloud mesh data is loaded yet."}
          </div>
        )}

        {selectedHit !== null ? (
          <PointCloudHitInspector hit={selectedHit} />
        ) : null}
      </div>
    </section>
  );
}
