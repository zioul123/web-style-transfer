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
  PointCloudPreviewViewSettings,
  PreviewCameraState,
} from "./types";
import type { NextPointCloudPreviewCameraCommand } from "./usePointCloudPreviewController";

type PendingCommandSignal = {
  readonly commandId: number;
  readonly resolve: () => void;
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
  readonly swapYZ: boolean;
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
  swapYZ,
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

  const restorePreviewAfterBatch = useCallback(
    async ({
      previousAssetState,
      previousActiveUploadedFileId,
      previousCameraState,
      previousSwapYZ,
    }: {
      readonly previousAssetState: LoadedAssetState;
      readonly previousActiveUploadedFileId: number | null;
      readonly previousCameraState: PreviewCameraState;
      readonly previousSwapYZ: boolean;
    }): Promise<void> => {
      setAssetState(previousAssetState);
      setActiveUploadedFileId(previousActiveUploadedFileId);
      resetPreviewInteraction();
      updateViewSettings({ swapYZ: previousSwapYZ });
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
    const previousSwapYZ = swapYZ;
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
          updateViewSettings({ swapYZ: viewpoint.swapYZ });
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
        previousSwapYZ,
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
          previousSwapYZ,
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
    swapYZ,
    updateViewSettings,
    uploadedFiles,
    waitForCommandSignal,
  ]);

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
    handleCameraCommandApplied,
    handlePreviewFrameRendered,
  };
};
