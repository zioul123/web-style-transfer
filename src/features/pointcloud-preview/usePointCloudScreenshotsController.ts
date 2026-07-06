import {
  useCallback,
  useRef,
  useState,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react";
import { createZipBlob } from "./createZipBlob";
import { parsePointCloudMeshText } from "./loadPointCloudMesh";
import type {
  BatchScreenshotProgress,
  LoadedAssetState,
  SavedViewpoint,
  UploadedPointCloudFile,
  UploadedPointCloudFileStatus,
} from "./pointCloudPreviewModels";
import type {
  PointCloudMeshData,
  PointCloudPreviewViewSettings,
  PreviewCameraState,
} from "./types";
import type { NextPointCloudPreviewCameraCommand } from "./usePointCloudPreviewController";

type PendingCommandSignal = {
  readonly commandId: number;
  readonly resolve: () => void;
};

export type PointCloudAblationGridExportCell =
  | {
      readonly status: "available";
      readonly xLabel: string;
      readonly yLabel: string;
      readonly file: File;
      readonly sourceLabel: string;
    }
  | {
      readonly status: "missing";
      readonly xLabel: string;
      readonly yLabel: string;
    };

export type PointCloudAblationGridExportRow = {
  readonly yLabel: string;
  readonly cells: readonly PointCloudAblationGridExportCell[];
};

export type PointCloudAblationGridExportRequest = {
  readonly xAxisLabel: string;
  readonly yAxisLabel: string;
  readonly xLabels: readonly string[];
  readonly rows: readonly PointCloudAblationGridExportRow[];
  readonly viewpoint: SavedViewpoint;
};

type UsePointCloudScreenshotsControllerOptions = {
  readonly previewHostRef: MutableRefObject<HTMLDivElement | null>;
  readonly assetState: LoadedAssetState;
  readonly setAssetState: Dispatch<SetStateAction<LoadedAssetState>>;
  readonly uploadedFiles: readonly UploadedPointCloudFile[];
  readonly activeUploadedFileId: number | null;
  readonly setActiveUploadedFileId: Dispatch<SetStateAction<number | null>>;
  readonly setUploadedFileStatus: (
    id: number,
    status: UploadedPointCloudFileStatus,
    errorMessage: string | null,
  ) => void;
  readonly savedViewpoints: readonly SavedViewpoint[];
  readonly currentCameraState: PreviewCameraState | null;
  readonly viewSettings: PointCloudPreviewViewSettings;
  readonly updateViewSettings: (
    patch: Partial<PointCloudPreviewViewSettings>,
  ) => void;
  readonly issueCameraCommand: (
    command: NextPointCloudPreviewCameraCommand,
  ) => number;
  readonly resetPreviewInteraction: () => void;
};

export type PointCloudScreenshotsController = {
  readonly showBatchScreenshotModal: boolean;
  readonly selectedBatchViewpointIds: readonly number[];
  readonly setSelectedBatchViewpointIds: Dispatch<
    SetStateAction<readonly number[]>
  >;
  readonly batchScreenshotProgress: BatchScreenshotProgress | null;
  readonly batchScreenshotError: string | null;
  readonly isBatchScreenshotRunning: boolean;
  readonly openBatchScreenshotModal: () => void;
  readonly closeBatchScreenshotModal: () => void;
  readonly captureScreenshot: () => void;
  readonly captureBatchScreenshots: () => Promise<void>;
  readonly captureAblationGridPng: (
    request: PointCloudAblationGridExportRequest,
  ) => Promise<void>;
  readonly handleCameraCommandApplied: (commandId: number) => void;
  readonly handlePreviewFrameRendered: (commandId: number) => void;
};

const screenshotFilenameStemForSourceLabel = (sourceLabel: string): string => {
  const trimmedLabel = sourceLabel.trim();
  const withoutExtension = trimmedLabel.replace(/\.[^.]+$/, "");
  const sanitized = withoutExtension
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return sanitized.length > 0 ? sanitized : "pointcloud-preview";
};

const screenshotTimestampLabel = (date: Date): string => {
  const pad = (value: number): string => value.toString().padStart(2, "0");
  return [
    date.getFullYear().toString(),
    pad(date.getMonth() + 1),
    pad(date.getDate()),
    "-",
    pad(date.getHours()),
    pad(date.getMinutes()),
    pad(date.getSeconds()),
  ].join("");
};

const canvasPngBytes = async (canvas: HTMLCanvasElement): Promise<Uint8Array> =>
  new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob === null) {
        reject(new Error("The preview canvas could not be encoded as PNG."));
        return;
      }
      void blob.arrayBuffer().then(
        (buffer) => resolve(new Uint8Array(buffer)),
        (error: unknown) => reject(error),
      );
    }, "image/png");
  });

const clearPointCloudExportPerformanceEntries = (): void => {
  performance.clearMeasures();
  performance.clearMarks();
};

const drawFittedImage = (
  context: CanvasRenderingContext2D,
  image: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
  targetX: number,
  targetY: number,
  targetWidth: number,
  targetHeight: number,
): void => {
  const sourceAspect = sourceWidth / sourceHeight;
  const targetAspect = targetWidth / targetHeight;
  const drawWidth =
    sourceAspect > targetAspect ? targetWidth : targetHeight * sourceAspect;
  const drawHeight =
    sourceAspect > targetAspect ? targetWidth / sourceAspect : targetHeight;
  context.drawImage(
    image,
    targetX + (targetWidth - drawWidth) / 2,
    targetY + (targetHeight - drawHeight) / 2,
    drawWidth,
    drawHeight,
  );
};

const drawWrappedText = (
  context: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  maxWidth: number,
  lineHeight: number,
  maxLines: number,
  align: CanvasTextAlign = "start",
): void => {
  const previousAlign = context.textAlign;
  context.textAlign = align;
  const words = text.split(/\s+/).filter((word) => word.length > 0);
  const lines: string[] = [];
  let currentLine = "";

  words.forEach((word) => {
    const nextLine = currentLine.length === 0 ? word : `${currentLine} ${word}`;
    if (context.measureText(nextLine).width <= maxWidth) {
      currentLine = nextLine;
      return;
    }

    if (currentLine.length > 0) {
      lines.push(currentLine);
    }
    currentLine = word;
  });

  if (currentLine.length > 0) {
    lines.push(currentLine);
  }

  const visibleLines = lines.slice(0, maxLines);
  visibleLines.forEach((line, index) => {
    const suffix =
      index === maxLines - 1 && lines.length > maxLines ? "..." : "";
    context.fillText(`${line}${suffix}`, x, y + index * lineHeight, maxWidth);
  });
  context.textAlign = previousAlign;
};

export const usePointCloudScreenshotsController = ({
  previewHostRef,
  assetState,
  setAssetState,
  uploadedFiles,
  activeUploadedFileId,
  setActiveUploadedFileId,
  setUploadedFileStatus,
  savedViewpoints,
  currentCameraState,
  viewSettings,
  updateViewSettings,
  issueCameraCommand,
  resetPreviewInteraction,
}: UsePointCloudScreenshotsControllerOptions): PointCloudScreenshotsController => {
  const cameraAppliedWaitersRef = useRef<PendingCommandSignal[]>([]);
  const frameRenderedWaitersRef = useRef<PendingCommandSignal[]>([]);
  const [showBatchScreenshotModal, setShowBatchScreenshotModal] =
    useState<boolean>(false);
  const [selectedBatchViewpointIds, setSelectedBatchViewpointIds] = useState<
    readonly number[]
  >([]);
  const [batchScreenshotProgress, setBatchScreenshotProgress] =
    useState<BatchScreenshotProgress | null>(null);
  const [batchScreenshotError, setBatchScreenshotError] = useState<
    string | null
  >(null);
  const [isBatchScreenshotRunning, setIsBatchScreenshotRunning] =
    useState<boolean>(false);

  const openBatchScreenshotModal = useCallback((): void => {
    setSelectedBatchViewpointIds([]);
    setBatchScreenshotError(null);
    setBatchScreenshotProgress(null);
    setShowBatchScreenshotModal(true);
  }, []);

  const closeBatchScreenshotModal = useCallback((): void => {
    if (!isBatchScreenshotRunning) {
      setShowBatchScreenshotModal(false);
    }
  }, [isBatchScreenshotRunning]);

  const waitForCommandSignal = useCallback(
    (
      waitersRef: MutableRefObject<PendingCommandSignal[]>,
      commandId: number,
      signalLabel: string,
    ): Promise<void> =>
      new Promise((resolve, reject) => {
        const timeoutId = window.setTimeout(() => {
          waitersRef.current = waitersRef.current.filter(
            (waiter) => waiter.commandId !== commandId,
          );
          reject(
            new Error(
              `Timed out waiting for the preview ${signalLabel} for camera command ${commandId}.`,
            ),
          );
        }, 10_000);
        waitersRef.current.push({
          commandId,
          resolve: () => {
            window.clearTimeout(timeoutId);
            resolve();
          },
        });
      }),
    [],
  );

  const resolveCommandSignals = useCallback(
    (
      waitersRef: MutableRefObject<PendingCommandSignal[]>,
      commandId: number,
    ): void => {
      const matchingWaiters = waitersRef.current.filter(
        (waiter) => waiter.commandId === commandId,
      );
      waitersRef.current = waitersRef.current.filter(
        (waiter) => waiter.commandId !== commandId,
      );
      matchingWaiters.forEach((waiter) => waiter.resolve());
    },
    [],
  );

  const handleCameraCommandApplied = useCallback(
    (commandId: number): void => {
      resolveCommandSignals(cameraAppliedWaitersRef, commandId);
    },
    [resolveCommandSignals],
  );

  const handlePreviewFrameRendered = useCallback(
    (commandId: number): void => {
      resolveCommandSignals(frameRenderedWaitersRef, commandId);
    },
    [resolveCommandSignals],
  );

  const viewSettingsForTemporaryData = useCallback(
    (
      data: PointCloudMeshData,
      swapYZ: boolean,
    ): PointCloudPreviewViewSettings => {
      const hasKernelPreview = data.convolutionKernelLevels.length > 0;
      const canKeepKernelMode =
        viewSettings.renderMode === "kernels" && hasKernelPreview;
      const hasSelectedKernelLevel =
        canKeepKernelMode &&
        data.convolutionKernelLevels.some(
          (level) => level.levelIndex === viewSettings.kernelLevelIndex,
        );

      return {
        ...viewSettings,
        renderMode: canKeepKernelMode ? "kernels" : "surface",
        kernelLevelIndex: hasSelectedKernelLevel
          ? viewSettings.kernelLevelIndex
          : (data.convolutionKernelLevels[0]?.levelIndex ??
            viewSettings.kernelLevelIndex),
        swapYZ,
      };
    },
    [viewSettings],
  );

  const restorePreviewAfterBatch = useCallback(
    async ({
      previousAssetState,
      previousActiveUploadedFileId,
      previousCameraState,
      previousViewSettings,
    }: {
      readonly previousAssetState: LoadedAssetState;
      readonly previousActiveUploadedFileId: number | null;
      readonly previousCameraState: PreviewCameraState;
      readonly previousViewSettings: PointCloudPreviewViewSettings;
    }): Promise<void> => {
      setAssetState(previousAssetState);
      setActiveUploadedFileId(previousActiveUploadedFileId);
      resetPreviewInteraction();
      updateViewSettings(previousViewSettings);
      const commandId = issueCameraCommand({
        type: "restore",
        camera: previousCameraState,
      });
      await waitForCommandSignal(
        cameraAppliedWaitersRef,
        commandId,
        "camera restore",
      );
      await waitForCommandSignal(
        frameRenderedWaitersRef,
        commandId,
        "restored frame",
      );
    },
    [
      issueCameraCommand,
      resetPreviewInteraction,
      setActiveUploadedFileId,
      setAssetState,
      updateViewSettings,
      waitForCommandSignal,
    ],
  );

  const captureBatchScreenshots = useCallback(async (): Promise<void> => {
    if (
      assetState.status !== "ready" ||
      currentCameraState === null ||
      uploadedFiles.length < 2
    ) {
      return;
    }
    const selectedViewpoints = savedViewpoints.filter((viewpoint) =>
      selectedBatchViewpointIds.includes(viewpoint.id),
    );
    if (selectedViewpoints.length === 0) {
      return;
    }

    const previousAssetState = assetState;
    const previousActiveUploadedFileId = activeUploadedFileId;
    const previousCameraState = currentCameraState;
    const previousViewSettings = viewSettings;
    const timestamp = screenshotTimestampLabel(new Date());
    const total = uploadedFiles.length * selectedViewpoints.length;
    const zipEntries: { readonly name: string; readonly data: Uint8Array }[] =
      [];

    setIsBatchScreenshotRunning(true);
    setBatchScreenshotError(null);
    setBatchScreenshotProgress({
      completed: 0,
      total,
      label: "Preparing batch screenshots",
    });

    try {
      for (const uploadedFile of uploadedFiles) {
        setUploadedFileStatus(uploadedFile.id, "loading", null);
        let meshData;
        try {
          meshData = parsePointCloudMeshText(await uploadedFile.file.text());
        } catch (error) {
          const message =
            error instanceof Error ? error.message : "Unable to parse JSON.";
          setUploadedFileStatus(uploadedFile.id, "error", message);
          throw new Error(`${uploadedFile.label}: ${message}`, {
            cause: error,
          });
        }

        setUploadedFileStatus(uploadedFile.id, "ready", null);
        setActiveUploadedFileId(uploadedFile.id);
        setAssetState({
          status: "ready",
          sourceLabel: uploadedFile.label,
          errorMessage: null,
          data: meshData,
        });
        resetPreviewInteraction();

        for (const viewpoint of selectedViewpoints) {
          setBatchScreenshotProgress({
            completed: zipEntries.length,
            total,
            label: `${uploadedFile.label} at ${viewpoint.label}`,
          });
          updateViewSettings(
            viewSettingsForTemporaryData(meshData, viewpoint.swapYZ),
          );
          const commandId = issueCameraCommand({
            type: "restore",
            camera: viewpoint.camera,
          });
          await waitForCommandSignal(
            cameraAppliedWaitersRef,
            commandId,
            "camera update",
          );
          await waitForCommandSignal(
            frameRenderedWaitersRef,
            commandId,
            "rendered frame",
          );

          const canvas = previewHostRef.current?.querySelector("canvas");
          if (!(canvas instanceof HTMLCanvasElement)) {
            throw new Error("The preview canvas is not available.");
          }
          const viewpointStem = screenshotFilenameStemForSourceLabel(
            viewpoint.label,
          );
          zipEntries.push({
            name: `${screenshotFilenameStemForSourceLabel(uploadedFile.label)}-upload-${uploadedFile.id}-view-${viewpoint.id}-${viewpointStem}-${timestamp}.png`,
            data: await canvasPngBytes(canvas),
          });
        }
      }

      await restorePreviewAfterBatch({
        previousAssetState,
        previousActiveUploadedFileId,
        previousCameraState,
        previousViewSettings,
      });
      setBatchScreenshotProgress({
        completed: total,
        total,
        label: "Creating ZIP archive",
      });
      const archiveUrl = URL.createObjectURL(createZipBlob(zipEntries));
      const downloadLink = document.createElement("a");
      downloadLink.href = archiveUrl;
      downloadLink.download = `pointcloud-batch-screenshots-${timestamp}.zip`;
      downloadLink.click();
      window.setTimeout(() => URL.revokeObjectURL(archiveUrl), 0);
      setShowBatchScreenshotModal(false);
      setBatchScreenshotProgress(null);
    } catch (error) {
      try {
        await restorePreviewAfterBatch({
          previousAssetState,
          previousActiveUploadedFileId,
          previousCameraState,
          previousViewSettings,
        });
      } catch {
        // Preserve the original batch error because it identifies the failed item.
      }
      setBatchScreenshotError(
        error instanceof Error
          ? error.message
          : "Unable to create batch screenshots.",
      );
      setBatchScreenshotProgress(null);
    } finally {
      setIsBatchScreenshotRunning(false);
    }
  }, [
    activeUploadedFileId,
    assetState,
    currentCameraState,
    issueCameraCommand,
    previewHostRef,
    resetPreviewInteraction,
    restorePreviewAfterBatch,
    savedViewpoints,
    selectedBatchViewpointIds,
    setActiveUploadedFileId,
    setAssetState,
    setUploadedFileStatus,
    updateViewSettings,
    uploadedFiles,
    viewSettings,
    viewSettingsForTemporaryData,
    waitForCommandSignal,
  ]);

  const captureAblationGridPng = useCallback(
    async (request: PointCloudAblationGridExportRequest): Promise<void> => {
      if (assetState.status !== "ready" || currentCameraState === null) {
        throw new Error("The preview must finish loading before grid export.");
      }

      if (request.xLabels.length === 0 || request.rows.length === 0) {
        throw new Error("The ablation matrix is empty.");
      }

      const previousAssetState = assetState;
      const previousActiveUploadedFileId = activeUploadedFileId;
      const previousCameraState = currentCameraState;
      const previousViewSettings = viewSettings;
      const timestamp = screenshotTimestampLabel(new Date());
      const leftLabelWidth = 220;
      const topLabelHeight = 88;
      const cellWidth = 360;
      const cellHeight = 270;
      const gap = 4;
      const width =
        leftLabelWidth +
        request.xLabels.length * cellWidth +
        Math.max(0, request.xLabels.length - 1) * gap;
      const height =
        topLabelHeight +
        request.rows.length * cellHeight +
        Math.max(0, request.rows.length - 1) * gap;
      const outputCanvas = document.createElement("canvas");
      outputCanvas.width = width;
      outputCanvas.height = height;
      const context = outputCanvas.getContext("2d");
      if (context === null) {
        throw new Error("The grid export canvas is not available.");
      }
      clearPointCloudExportPerformanceEntries();

      context.fillStyle = "#ffffff";
      context.fillRect(0, 0, width, height);
      context.fillStyle = "#f1f5f9";
      context.fillRect(0, 0, leftLabelWidth - gap, topLabelHeight - gap);
      context.strokeStyle = "#cbd5e1";
      context.lineWidth = 1;
      context.strokeRect(
        0.5,
        0.5,
        leftLabelWidth - gap - 1,
        topLabelHeight - gap - 1,
      );
      context.fillStyle = "#0f172a";
      context.font = "700 16px system-ui, sans-serif";
      drawWrappedText(
        context,
        `X: ${request.xAxisLabel}`,
        (leftLabelWidth - gap) / 2,
        27,
        leftLabelWidth - 24,
        19,
        1,
        "center",
      );
      context.font = "700 16px system-ui, sans-serif";
      drawWrappedText(
        context,
        `Y: ${request.yAxisLabel}`,
        (leftLabelWidth - gap) / 2,
        51,
        leftLabelWidth - 24,
        19,
        1,
        "center",
      );
      context.font = "500 12px system-ui, sans-serif";
      context.fillStyle = "#475569";
      drawWrappedText(
        context,
        request.viewpoint.label,
        (leftLabelWidth - gap) / 2,
        73,
        leftLabelWidth - 24,
        15,
        1,
        "center",
      );

      request.xLabels.forEach((label, columnIndex) => {
        const x = leftLabelWidth + columnIndex * (cellWidth + gap);
        context.fillStyle = "#f8fafc";
        context.fillRect(x, 0, cellWidth, topLabelHeight - gap);
        context.strokeStyle = "#cbd5e1";
        context.lineWidth = 1;
        context.strokeRect(
          x + 0.5,
          0.5,
          cellWidth - 1,
          topLabelHeight - gap - 1,
        );
        context.fillStyle = "#0f172a";
        context.font = "700 18px system-ui, sans-serif";
        drawWrappedText(
          context,
          label,
          x + cellWidth / 2,
          49,
          cellWidth - 28,
          22,
          2,
          "center",
        );
      });

      try {
        for (const [rowIndex, row] of request.rows.entries()) {
          const y = topLabelHeight + rowIndex * (cellHeight + gap);
          context.fillStyle = "#f8fafc";
          context.fillRect(0, y, leftLabelWidth - gap, cellHeight);
          context.strokeStyle = "#cbd5e1";
          context.lineWidth = 1;
          context.strokeRect(
            0.5,
            y + 0.5,
            leftLabelWidth - gap - 1,
            cellHeight - 1,
          );
          context.fillStyle = "#0f172a";
          context.font = "700 18px system-ui, sans-serif";
          drawWrappedText(
            context,
            row.yLabel,
            (leftLabelWidth - gap) / 2,
            y + cellHeight / 2,
            leftLabelWidth - 28,
            22,
            4,
            "center",
          );

          for (const [columnIndex, cell] of row.cells.entries()) {
            const x = leftLabelWidth + columnIndex * (cellWidth + gap);
            context.fillStyle = "#ffffff";
            context.fillRect(x, y, cellWidth, cellHeight);
            context.strokeStyle = "#cbd5e1";
            context.lineWidth = 1;
            context.strokeRect(x + 0.5, y + 0.5, cellWidth - 1, cellHeight - 1);

            if (cell.status === "missing") {
              context.setLineDash([8, 6]);
              context.strokeStyle = "#94a3b8";
              context.strokeRect(
                x + 24,
                y + 24,
                cellWidth - 48,
                cellHeight - 48,
              );
              context.setLineDash([]);
              context.fillStyle = "#64748b";
              context.font = "600 22px system-ui, sans-serif";
              context.textAlign = "center";
              context.fillText(
                "Missing",
                x + cellWidth / 2,
                y + cellHeight / 2,
              );
              context.textAlign = "start";
              continue;
            }

            let meshData;
            try {
              meshData = parsePointCloudMeshText(await cell.file.text());
            } catch (error) {
              const message =
                error instanceof Error
                  ? error.message
                  : "Unable to parse JSON.";
              throw new Error(`${cell.sourceLabel}: ${message}`, {
                cause: error,
              });
            }

            clearPointCloudExportPerformanceEntries();
            setAssetState({
              status: "ready",
              sourceLabel: cell.sourceLabel,
              errorMessage: null,
              data: meshData,
            });
            setActiveUploadedFileId(null);
            resetPreviewInteraction();
            updateViewSettings(
              viewSettingsForTemporaryData(meshData, request.viewpoint.swapYZ),
            );
            const commandId = issueCameraCommand({
              type: "restore",
              camera: request.viewpoint.camera,
            });
            await waitForCommandSignal(
              cameraAppliedWaitersRef,
              commandId,
              "camera update",
            );
            await waitForCommandSignal(
              frameRenderedWaitersRef,
              commandId,
              "rendered frame",
            );

            const previewCanvas =
              previewHostRef.current?.querySelector("canvas");
            if (!(previewCanvas instanceof HTMLCanvasElement)) {
              throw new Error("The preview canvas is not available.");
            }
            drawFittedImage(
              context,
              previewCanvas,
              previewCanvas.width,
              previewCanvas.height,
              x + 12,
              y + 12,
              cellWidth - 24,
              cellHeight - 24,
            );
            clearPointCloudExportPerformanceEntries();
          }
        }

        await restorePreviewAfterBatch({
          previousAssetState,
          previousActiveUploadedFileId,
          previousCameraState,
          previousViewSettings,
        });

        const pngBytes = await canvasPngBytes(outputCanvas);
        const pngBuffer = new ArrayBuffer(pngBytes.byteLength);
        new Uint8Array(pngBuffer).set(pngBytes);
        const pngUrl = URL.createObjectURL(
          new Blob([pngBuffer], { type: "image/png" }),
        );
        const downloadLink = document.createElement("a");
        downloadLink.href = pngUrl;
        downloadLink.download = `pointcloud-ablation-grid-${screenshotFilenameStemForSourceLabel(
          request.viewpoint.label,
        )}-${timestamp}.png`;
        downloadLink.click();
        window.setTimeout(() => URL.revokeObjectURL(pngUrl), 0);
        clearPointCloudExportPerformanceEntries();
      } catch (error) {
        try {
          await restorePreviewAfterBatch({
            previousAssetState,
            previousActiveUploadedFileId,
            previousCameraState,
            previousViewSettings,
          });
        } catch {
          // Preserve the original export error because it identifies the failed cell.
        }
        throw error;
      } finally {
        clearPointCloudExportPerformanceEntries();
      }
    },
    [
      activeUploadedFileId,
      assetState,
      currentCameraState,
      issueCameraCommand,
      previewHostRef,
      resetPreviewInteraction,
      restorePreviewAfterBatch,
      setActiveUploadedFileId,
      setAssetState,
      updateViewSettings,
      viewSettings,
      viewSettingsForTemporaryData,
      waitForCommandSignal,
    ],
  );

  const captureScreenshot = useCallback((): void => {
    const canvas = previewHostRef.current?.querySelector("canvas");
    if (!(canvas instanceof HTMLCanvasElement)) {
      return;
    }

    const downloadLink = document.createElement("a");
    downloadLink.href = canvas.toDataURL("image/png");
    downloadLink.download = `${screenshotFilenameStemForSourceLabel(assetState.sourceLabel)}-screenshot-${screenshotTimestampLabel(new Date())}.png`;
    downloadLink.click();
  }, [assetState.sourceLabel, previewHostRef]);

  return {
    showBatchScreenshotModal,
    selectedBatchViewpointIds,
    setSelectedBatchViewpointIds,
    batchScreenshotProgress,
    batchScreenshotError,
    isBatchScreenshotRunning,
    openBatchScreenshotModal,
    closeBatchScreenshotModal,
    captureScreenshot,
    captureBatchScreenshots,
    captureAblationGridPng,
    handleCameraCommandApplied,
    handlePreviewFrameRendered,
  };
};
