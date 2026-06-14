import { useMemo, useRef, useState } from "react";
import { BatchScreenshotModal } from "./BatchScreenshotModal";
import { PointCloudPreviewControlsPanel } from "./PointCloudPreviewControlsPanel";
import {
  DataSourcePanel,
  DatasetStatsPanel,
  RenderingAlgorithmPanel,
} from "./PointCloudPreviewDataPanels";
import { maxFragmentShaderPointsPerCell } from "./PointCloudPreviewScene";
import {
  PointCloudPreviewHeader,
  PointCloudPreviewInfoModal,
} from "./PointCloudPreviewUi";
import { PointCloudPreviewViewport } from "./PointCloudPreviewViewport";
import type { MeshColorMode } from "./types";
import {
  bundledMediumExampleUrl,
  bundledMediumSourceLabel,
  usePointCloudAssetsController,
} from "./usePointCloudAssetsController";
import { usePointCloudPreviewController } from "./usePointCloudPreviewController";
import { usePointCloudScreenshotsController } from "./usePointCloudScreenshotsController";
import { useSavedViewpointsController } from "./useSavedViewpointsController";

export function PointCloudPreviewApp() {
  const previewHostRef = useRef<HTMLDivElement | null>(null);
  const [showInfoModal, setShowInfoModal] = useState<boolean>(false);
  const previewController = usePointCloudPreviewController();
  const assetsController = usePointCloudAssetsController({
    currentCameraStateRef: previewController.cameraStateRef,
    updateViewSettings: previewController.updateViewSettings,
    issueCameraCommand: previewController.issueCameraCommand,
    resetPreviewInteraction: previewController.resetPreviewInteraction,
  });
  const viewpointsController = useSavedViewpointsController({
    currentCameraState: previewController.cameraState,
    swapYZ: previewController.viewSettings.swapYZ,
  });
  const screenshotsController = usePointCloudScreenshotsController({
    previewHostRef,
    assetState: assetsController.assetState,
    setAssetState: assetsController.setAssetState,
    uploadedFiles: assetsController.uploadedFiles,
    activeUploadedFileId: assetsController.activeUploadedFileId,
    setActiveUploadedFileId: assetsController.setActiveUploadedFileId,
    setUploadedFileStatus: assetsController.setUploadedFileStatus,
    savedViewpoints: viewpointsController.savedViewpoints,
    currentCameraState: previewController.cameraState,
    swapYZ: previewController.viewSettings.swapYZ,
    updateViewSettings: previewController.updateViewSettings,
    issueCameraCommand: previewController.issueCameraCommand,
    resetPreviewInteraction: previewController.resetPreviewInteraction,
  });

  const data = assetsController.assetState.data;
  const canUseFragmentKnnShading = useMemo(
    () =>
      data !== null &&
      data.spatialHash.maxPointsPerCell <= maxFragmentShaderPointsPerCell,
    [data],
  );
  const effectiveMeshColorMode: MeshColorMode =
    previewController.viewSettings.meshColorMode === "fragment-knn" &&
    canUseFragmentKnnShading
      ? "fragment-knn"
      : "baked";
  const meshColorModeDescription =
    effectiveMeshColorMode === "fragment-knn"
      ? `Spatial-hash fragment KNN shading active (${data?.pointCount ?? 0} points across ${data?.spatialHash.cellCount ?? 0} cells).`
      : previewController.viewSettings.meshColorMode === "fragment-knn" &&
          data !== null
        ? `Spatial-hash fragment KNN shading found a dense cell with ${data.spatialHash.maxPointsPerCell} points, which exceeds the current per-cell shader cap of ${maxFragmentShaderPointsPerCell}; this dataset falls back to baked vertex colours.`
        : !previewController.viewSettings.disableGammaDecoding
          ? "Baked vertex colours active with gamma decoding enabled."
          : "Baked vertex colours active with gamma decoding disabled.";

  return (
    <>
      <main className="h-screen w-screen overflow-hidden bg-[linear-gradient(180deg,_#071019_0%,_#0b1120_100%)] text-slate-100">
        <div className="flex h-full w-full flex-col px-4 py-4">
          <PointCloudPreviewHeader onShowInfo={() => setShowInfoModal(true)} />

          <div className="flex min-h-0 flex-1 items-stretch gap-4">
            <div className="flex h-full w-[25rem] shrink-0 flex-col gap-4 overflow-y-auto pr-1">
              <RenderingAlgorithmPanel
                meshColorMode={previewController.viewSettings.meshColorMode}
                description={meshColorModeDescription}
                onMeshColorModeChange={(meshColorMode) =>
                  previewController.updateViewSettings({ meshColorMode })
                }
              />
              <DataSourcePanel
                assetState={assetsController.assetState}
                uploadedFiles={assetsController.uploadedFiles}
                activeUploadedFileId={assetsController.activeUploadedFileId}
                onFileUpload={assetsController.handleFileUpload}
                onReload={() => {
                  void assetsController.loadBundledExample(
                    bundledMediumExampleUrl,
                    bundledMediumSourceLabel,
                    "Unable to reload the bundled medium example.",
                  );
                }}
                onSelectFile={(uploadedFile) => {
                  void assetsController.loadQueuedFile(uploadedFile);
                }}
                onRemoveFile={assetsController.removeQueuedFile}
              />
              <DatasetStatsPanel data={data} />
            </div>

            <PointCloudPreviewViewport
              previewHostRef={previewHostRef}
              assetState={assetsController.assetState}
              viewSettings={previewController.viewSettings}
              effectiveMeshColorMode={effectiveMeshColorMode}
              selectedHit={previewController.selectedHit}
              framesPerSecond={previewController.framesPerSecond}
              cameraCommand={previewController.cameraCommand}
              canBatchScreenshot={assetsController.uploadedFiles.length > 1}
              onScreenshot={screenshotsController.captureScreenshot}
              onOpenBatchScreenshot={
                screenshotsController.openBatchScreenshotModal
              }
              onHoverSampleChange={previewController.setSelectedHit}
              onCameraStateChange={previewController.handleCameraStateChange}
              onCameraCommandApplied={
                screenshotsController.handleCameraCommandApplied
              }
              onFrameRendered={screenshotsController.handlePreviewFrameRendered}
              onFramesPerSecondChange={previewController.setFramesPerSecond}
            />

            <PointCloudPreviewControlsPanel
              viewSettings={previewController.viewSettings}
              updateViewSettings={previewController.updateViewSettings}
              currentCameraState={previewController.cameraState}
              issueCameraCommand={previewController.issueCameraCommand}
              savedViewpoints={viewpointsController.savedViewpoints}
              onSaveViewpoint={viewpointsController.saveViewpoint}
              onRenameViewpoint={viewpointsController.renameViewpoint}
              onUpdateViewpoint={viewpointsController.updateViewpoint}
              onDeleteViewpoint={viewpointsController.deleteViewpoint}
            />
          </div>
        </div>
      </main>

      <BatchScreenshotModal
        open={screenshotsController.showBatchScreenshotModal}
        uploadedFileCount={assetsController.uploadedFiles.length}
        savedViewpoints={viewpointsController.savedViewpoints}
        selectedViewpointIds={screenshotsController.selectedBatchViewpointIds}
        setSelectedViewpointIds={
          screenshotsController.setSelectedBatchViewpointIds
        }
        progress={screenshotsController.batchScreenshotProgress}
        error={screenshotsController.batchScreenshotError}
        isRunning={screenshotsController.isBatchScreenshotRunning}
        onClose={screenshotsController.closeBatchScreenshotModal}
        onDownload={screenshotsController.captureBatchScreenshots}
      />
      <PointCloudPreviewInfoModal
        open={showInfoModal}
        onClose={() => setShowInfoModal(false)}
      />
    </>
  );
}
