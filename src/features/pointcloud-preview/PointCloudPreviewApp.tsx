import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PointCloudAblationTab } from "./ablation/PointCloudAblationTab";
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
import type { MeshColorMode, PointCloudPreviewRenderMode } from "./types";
import {
  bundledMediumExampleUrl,
  bundledMediumSourceLabel,
  usePointCloudAssetsController,
} from "./usePointCloudAssetsController";
import { usePointCloudPreviewController } from "./usePointCloudPreviewController";
import {
  usePointCloudScreenshotsController,
  type PointCloudAblationGridExportRequest,
  type PointCloudAblationMultiViewExportRequest,
} from "./usePointCloudScreenshotsController";
import { useSavedViewpointsController } from "./useSavedViewpointsController";

type PointCloudPreviewTab = "preview" | "ablation";

const pointCloudPreviewTabs: readonly {
  readonly id: PointCloudPreviewTab;
  readonly label: string;
  readonly testId: string;
}[] = [
  { id: "preview", label: "Preview", testId: "pointcloud-preview-tab" },
  { id: "ablation", label: "Ablation", testId: "pointcloud-ablation-tab" },
];

export function PointCloudPreviewApp() {
  const previewHostRef = useRef<HTMLDivElement | null>(null);
  const [activeTab, setActiveTab] = useState<PointCloudPreviewTab>("preview");
  const [showInfoModal, setShowInfoModal] = useState<boolean>(false);
  const [ablationExportLabel, setAblationExportLabel] = useState<string | null>(
    null,
  );
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
    viewSettings: previewController.viewSettings,
    updateViewSettings: previewController.updateViewSettings,
    issueCameraCommand: previewController.issueCameraCommand,
    resetPreviewInteraction: previewController.resetPreviewInteraction,
  });

  const currentRenderMode = previewController.viewSettings.renderMode;
  const currentKernelLevelIndex =
    previewController.viewSettings.kernelLevelIndex;
  const updatePreviewViewSettings = previewController.updateViewSettings;
  const data = assetsController.assetState.data;
  const kernelLevelOptions = useMemo(
    () =>
      data?.convolutionKernelLevels.map((level) => ({
        levelIndex: level.levelIndex,
        groupCount: level.groupCount,
      })) ?? [],
    [data],
  );
  const hasKernelPreview = kernelLevelOptions.length > 0;
  const effectiveRenderMode: PointCloudPreviewRenderMode =
    currentRenderMode === "kernels" && hasKernelPreview ? "kernels" : "surface";
  const activeKernelLevel = useMemo(
    () =>
      data?.convolutionKernelLevels.find(
        (level) => level.levelIndex === currentKernelLevelIndex,
      ) ?? data?.convolutionKernelLevels[0],
    [currentKernelLevelIndex, data],
  );
  const canUseFragmentKnnShading = useMemo(
    () =>
      data !== null &&
      data.spatialHash.maxPointsPerCell <= maxFragmentShaderPointsPerCell,
    [data],
  );
  const effectiveMeshColorMode: MeshColorMode =
    effectiveRenderMode === "surface" &&
    previewController.viewSettings.meshColorMode === "fragment-knn" &&
    canUseFragmentKnnShading
      ? "fragment-knn"
      : "baked";
  const meshColorModeDescription =
    effectiveRenderMode === "kernels"
      ? `Kernel preview active for ${activeKernelLevel?.label ?? "selected level"} with ${activeKernelLevel?.groupCount ?? 0} anchors.`
      : effectiveMeshColorMode === "fragment-knn"
        ? `Spatial-hash fragment KNN shading active (${data?.pointCount ?? 0} points across ${data?.spatialHash.cellCount ?? 0} cells).`
        : previewController.viewSettings.meshColorMode === "fragment-knn" &&
            data !== null
          ? `Spatial-hash fragment KNN shading found a dense cell with ${data.spatialHash.maxPointsPerCell} points, which exceeds the current per-cell shader cap of ${maxFragmentShaderPointsPerCell}; this dataset falls back to baked vertex colours.`
          : !previewController.viewSettings.disableGammaDecoding
            ? "Baked vertex colours active with gamma decoding enabled."
            : "Baked vertex colours active with gamma decoding disabled.";

  useEffect(() => {
    if (data === null) {
      return;
    }

    if (data.convolutionKernelLevels.length === 0) {
      if (currentRenderMode === "kernels") {
        updatePreviewViewSettings({ renderMode: "surface" });
      }
      return;
    }

    const hasSelectedKernelLevel = data.convolutionKernelLevels.some(
      (level) => level.levelIndex === currentKernelLevelIndex,
    );
    if (!hasSelectedKernelLevel) {
      updatePreviewViewSettings({
        kernelLevelIndex: data.convolutionKernelLevels[0]?.levelIndex ?? 0,
      });
    }
  }, [
    currentKernelLevelIndex,
    currentRenderMode,
    data,
    updatePreviewViewSettings,
  ]);

  const exportAblationGrid = useCallback(
    async (request: PointCloudAblationGridExportRequest): Promise<void> => {
      const previousTab = activeTab;
      setAblationExportLabel("Exporting ablation grid...");
      setActiveTab("preview");
      await new Promise<void>((resolve) => {
        window.requestAnimationFrame(() => {
          window.requestAnimationFrame(() => resolve());
        });
      });
      try {
        await screenshotsController.captureAblationGridPng(request);
      } finally {
        setActiveTab(previousTab);
        setAblationExportLabel(null);
      }
    },
    [activeTab, screenshotsController],
  );

  const exportAblationViewpoints = useCallback(
    async (
      request: PointCloudAblationMultiViewExportRequest,
    ): Promise<void> => {
      const previousTab = activeTab;
      setAblationExportLabel("Exporting viewpoint sheet...");
      setActiveTab("preview");
      await new Promise<void>((resolve) => {
        window.requestAnimationFrame(() => {
          window.requestAnimationFrame(() => resolve());
        });
      });
      try {
        await screenshotsController.captureAblationViewpointsPng(request);
      } finally {
        setActiveTab(previousTab);
        setAblationExportLabel(null);
      }
    },
    [activeTab, screenshotsController],
  );

  return (
    <>
      <main
        className="min-h-screen w-full overflow-x-hidden bg-[linear-gradient(180deg,_#071019_0%,_#0b1120_100%)] text-slate-100 2xl:h-screen 2xl:overflow-hidden"
        inert={ablationExportLabel !== null}
      >
        <div className="flex min-h-screen w-full flex-col px-4 py-4 2xl:h-full 2xl:min-h-0">
          <PointCloudPreviewHeader onShowInfo={() => setShowInfoModal(true)} />

          <div
            className="mb-4 flex shrink-0 flex-wrap gap-2"
            role="tablist"
            aria-label="Point-cloud preview mode"
            data-testid="pointcloud-preview-tab-switch"
          >
            {pointCloudPreviewTabs.map((tab) => {
              const isSelected = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  id={`pointcloud-${tab.id}-tab`}
                  className={`rounded-xl px-4 py-2 text-sm font-semibold transition ${
                    isSelected
                      ? "bg-sky-300 text-sky-950"
                      : "border border-white/10 bg-white/5 text-slate-100 hover:bg-white/10"
                  }`}
                  type="button"
                  role="tab"
                  aria-selected={isSelected}
                  aria-controls={`pointcloud-${tab.id}-panel`}
                  data-testid={tab.testId}
                  disabled={ablationExportLabel !== null}
                  onClick={() => setActiveTab(tab.id)}
                >
                  {tab.label}
                </button>
              );
            })}
          </div>

          <div
            id="pointcloud-preview-panel"
            data-testid="pointcloud-preview-tab-panel"
            role="tabpanel"
            aria-labelledby="pointcloud-preview-tab"
            hidden={activeTab !== "preview"}
            className={
              activeTab === "preview"
                ? "flex min-h-0 flex-1 flex-col items-stretch gap-4 2xl:flex-row"
                : "hidden"
            }
          >
            <div
              data-testid="pointcloud-left-panel"
              className="flex w-full flex-col gap-4 2xl:h-full 2xl:w-[25rem] 2xl:shrink-0 2xl:overflow-y-auto 2xl:pr-1"
            >
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
              renderMode={effectiveRenderMode}
              viewSettings={previewController.viewSettings}
              effectiveMeshColorMode={effectiveMeshColorMode}
              selectedHit={previewController.selectedHit}
              selectedKernelHit={previewController.selectedKernelHit}
              framesPerSecond={previewController.framesPerSecond}
              cameraCommand={previewController.cameraCommand}
              canBatchScreenshot={assetsController.uploadedFiles.length > 1}
              onScreenshot={screenshotsController.captureScreenshot}
              onOpenBatchScreenshot={
                screenshotsController.openBatchScreenshotModal
              }
              onHoverSampleChange={previewController.setSelectedHit}
              onHoverKernelSampleChange={previewController.setSelectedKernelHit}
              onCameraStateChange={previewController.handleCameraStateChange}
              onCameraCommandApplied={
                screenshotsController.handleCameraCommandApplied
              }
              onFrameRendered={screenshotsController.handlePreviewFrameRendered}
              onFramesPerSecondChange={previewController.setFramesPerSecond}
            />

            <PointCloudPreviewControlsPanel
              viewSettings={previewController.viewSettings}
              hasKernelPreview={hasKernelPreview}
              kernelLevelOptions={kernelLevelOptions}
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

          <div
            id="pointcloud-ablation-panel"
            data-testid="pointcloud-ablation-tab-panel"
            role="tabpanel"
            aria-labelledby="pointcloud-ablation-tab"
            hidden={activeTab !== "ablation"}
            className={
              activeTab === "ablation" ? "flex min-h-0 flex-1" : "hidden"
            }
          >
            <PointCloudAblationTab
              savedViewpoints={viewpointsController.savedViewpoints}
              onPreviewExperimentFile={async (file, sourceLabel) => {
                const didLoad = await assetsController.loadTransientFile(
                  file,
                  sourceLabel,
                );
                if (didLoad) {
                  setActiveTab("preview");
                }
                return didLoad;
              }}
              onExportAblationGrid={exportAblationGrid}
              onExportAblationViewpoints={exportAblationViewpoints}
            />
          </div>
        </div>
      </main>

      {ablationExportLabel !== null ? (
        <div
          className="fixed inset-0 z-[100] flex cursor-wait items-center justify-center bg-slate-950/75 px-4 backdrop-blur-sm"
          data-testid="pointcloud-ablation-export-overlay"
          role="status"
          aria-live="polite"
        >
          <div className="w-full max-w-md rounded-[1.1rem] border border-sky-300/20 bg-slate-950 p-6 text-center shadow-2xl shadow-black/50">
            <p className="text-lg font-semibold text-white">
              {ablationExportLabel}
            </p>
            <p className="mt-2 text-sm text-slate-300">
              Rendering saved camera views. Preview controls are temporarily
              locked.
            </p>
          </div>
        </div>
      ) : null}

      {activeTab === "preview" ? (
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
      ) : null}
      <PointCloudPreviewInfoModal
        open={showInfoModal}
        onClose={() => setShowInfoModal(false)}
      />
    </>
  );
}
