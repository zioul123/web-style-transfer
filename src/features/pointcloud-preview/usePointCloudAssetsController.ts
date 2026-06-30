import {
  startTransition,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react";
import { assetUrl } from "../../shared/assetUrls";
import { parsePointCloudMeshText } from "./loadPointCloudMesh";
import type {
  LoadedAssetState,
  UploadedPointCloudFile,
  UploadedPointCloudFileStatus,
} from "./pointCloudPreviewModels";
import type {
  PointCloudMeshData,
  PointCloudPreviewViewSettings,
  PreviewCameraState,
} from "./types";
import type { NextPointCloudPreviewCameraCommand } from "./usePointCloudPreviewController";

export const bundledMediumExampleUrl = assetUrl(
  "pointcloud-style-transfer/pc-and-mesh-medium-example.json",
);
export const bundledMediumSourceLabel = "Bundled medium example";

type UsePointCloudAssetsControllerOptions = {
  readonly currentCameraStateRef: MutableRefObject<PreviewCameraState | null>;
  readonly updateViewSettings: (
    patch: Partial<PointCloudPreviewViewSettings>,
  ) => void;
  readonly issueCameraCommand: (
    command: NextPointCloudPreviewCameraCommand,
  ) => number;
  readonly resetPreviewInteraction: (options?: {
    readonly clearCameraState?: boolean;
  }) => void;
};

export type PointCloudAssetsController = {
  readonly assetState: LoadedAssetState;
  readonly setAssetState: Dispatch<SetStateAction<LoadedAssetState>>;
  readonly uploadedFiles: readonly UploadedPointCloudFile[];
  readonly activeUploadedFileId: number | null;
  readonly setActiveUploadedFileId: Dispatch<SetStateAction<number | null>>;
  readonly handleFileUpload: (
    event: ChangeEvent<HTMLInputElement>,
  ) => Promise<void>;
  readonly loadQueuedFile: (
    uploadedFile: UploadedPointCloudFile,
  ) => Promise<void>;
  readonly loadTransientFile: (
    file: File,
    sourceLabel: string,
  ) => Promise<boolean>;
  readonly loadBundledExample: (
    url: string,
    sourceLabel: string,
    fallbackErrorMessage: string,
  ) => Promise<void>;
  readonly removeQueuedFile: (removedFileId: number) => void;
  readonly setUploadedFileStatus: (
    id: number,
    status: UploadedPointCloudFileStatus,
    errorMessage: string | null,
  ) => void;
};

export const usePointCloudAssetsController = ({
  currentCameraStateRef,
  updateViewSettings,
  issueCameraCommand,
  resetPreviewInteraction,
}: UsePointCloudAssetsControllerOptions): PointCloudAssetsController => {
  const nextUploadedFileIdRef = useRef<number>(1);
  const activeLoadRequestIdRef = useRef<number>(0);
  const [assetState, setAssetState] = useState<LoadedAssetState>({
    status: "loading",
    sourceLabel: bundledMediumSourceLabel,
    errorMessage: null,
    data: null,
  });
  const [uploadedFiles, setUploadedFiles] = useState<
    readonly UploadedPointCloudFile[]
  >([]);
  const [activeUploadedFileId, setActiveUploadedFileId] = useState<
    number | null
  >(null);

  const beginAssetLoadRequest = useCallback((): number => {
    activeLoadRequestIdRef.current += 1;
    return activeLoadRequestIdRef.current;
  }, []);

  const applyReadyAsset = useCallback(
    (
      data: PointCloudMeshData,
      sourceLabel: string,
      options: {
        readonly preserveCurrentView?: boolean;
        readonly frameCamera?: boolean;
      } = {},
    ): void => {
      const shouldPreserveCurrentView = options.preserveCurrentView === true;
      const shouldFrameCamera =
        options.frameCamera ?? !shouldPreserveCurrentView;
      startTransition(() => {
        setAssetState({
          status: "ready",
          sourceLabel,
          errorMessage: null,
          data,
        });
        resetPreviewInteraction({
          clearCameraState: !shouldPreserveCurrentView,
        });
        if (!shouldPreserveCurrentView) {
          updateViewSettings({ swapYZ: false });
        }
        if (shouldFrameCamera) {
          issueCameraCommand({ type: "frame" });
        }
      });
    },
    [issueCameraCommand, resetPreviewInteraction, updateViewSettings],
  );

  const setUploadedFileStatus = useCallback(
    (
      id: number,
      status: UploadedPointCloudFileStatus,
      errorMessage: string | null,
    ): void => {
      setUploadedFiles((currentFiles) =>
        currentFiles.map((uploadedFile) =>
          uploadedFile.id === id
            ? {
                ...uploadedFile,
                status,
                errorMessage,
              }
            : uploadedFile,
        ),
      );
    },
    [],
  );

  const loadQueuedFile = useCallback(
    async (uploadedFile: UploadedPointCloudFile): Promise<void> => {
      if (
        activeUploadedFileId === uploadedFile.id &&
        assetState.status === "ready"
      ) {
        return;
      }

      const requestId = beginAssetLoadRequest();
      const shouldFrameCamera = currentCameraStateRef.current === null;
      setActiveUploadedFileId(uploadedFile.id);
      setUploadedFiles((currentFiles) =>
        currentFiles.map((currentFile) =>
          currentFile.id === uploadedFile.id
            ? {
                ...currentFile,
                status: "loading",
                errorMessage: null,
              }
            : currentFile.status === "loading"
              ? {
                  ...currentFile,
                  status: "idle",
                  errorMessage: null,
                }
              : currentFile,
        ),
      );
      setAssetState({
        status: "loading",
        sourceLabel: uploadedFile.label,
        errorMessage: null,
        data: null,
      });

      try {
        const data = parsePointCloudMeshText(await uploadedFile.file.text());
        if (activeLoadRequestIdRef.current !== requestId) {
          return;
        }
        applyReadyAsset(data, uploadedFile.label, {
          preserveCurrentView: true,
          frameCamera: shouldFrameCamera,
        });
        setUploadedFileStatus(uploadedFile.id, "ready", null);
      } catch (error) {
        if (activeLoadRequestIdRef.current !== requestId) {
          return;
        }
        const errorMessage =
          error instanceof Error
            ? error.message
            : "Unable to parse uploaded JSON.";
        setAssetState({
          status: "error",
          sourceLabel: uploadedFile.label,
          errorMessage,
          data: null,
        });
        setUploadedFileStatus(uploadedFile.id, "error", errorMessage);
      }
    },
    [
      activeUploadedFileId,
      applyReadyAsset,
      assetState.status,
      beginAssetLoadRequest,
      currentCameraStateRef,
      setUploadedFileStatus,
    ],
  );

  const loadTransientFile = useCallback(
    async (file: File, sourceLabel: string): Promise<boolean> => {
      const requestId = beginAssetLoadRequest();
      const cameraStateToRestore = currentCameraStateRef.current;
      setActiveUploadedFileId(null);
      setUploadedFiles((currentFiles) =>
        currentFiles.map((uploadedFile) =>
          uploadedFile.status === "loading"
            ? {
                ...uploadedFile,
                status: "idle",
                errorMessage: null,
              }
            : uploadedFile,
        ),
      );
      setAssetState({
        status: "loading",
        sourceLabel,
        errorMessage: null,
        data: null,
      });

      try {
        const data = parsePointCloudMeshText(await file.text());
        if (activeLoadRequestIdRef.current !== requestId) {
          return false;
        }
        applyReadyAsset(data, sourceLabel, {
          preserveCurrentView: cameraStateToRestore !== null,
          frameCamera: cameraStateToRestore === null,
        });
        if (cameraStateToRestore !== null) {
          issueCameraCommand({
            type: "restore",
            camera: cameraStateToRestore,
          });
        }
        return true;
      } catch (error) {
        if (activeLoadRequestIdRef.current !== requestId) {
          return false;
        }
        setAssetState({
          status: "error",
          sourceLabel,
          errorMessage:
            error instanceof Error
              ? error.message
              : "Unable to parse selected experiment JSON.",
          data: null,
        });
        return false;
      }
    },
    [
      applyReadyAsset,
      beginAssetLoadRequest,
      currentCameraStateRef,
      issueCameraCommand,
    ],
  );

  const loadBundledExample = useCallback(
    async (
      url: string,
      sourceLabel: string,
      fallbackErrorMessage: string,
    ): Promise<void> => {
      const requestId = beginAssetLoadRequest();
      setActiveUploadedFileId(null);
      setUploadedFiles((currentFiles) =>
        currentFiles.map((uploadedFile) =>
          uploadedFile.status === "loading"
            ? {
                ...uploadedFile,
                status: "idle",
                errorMessage: null,
              }
            : uploadedFile,
        ),
      );
      setAssetState({
        status: "loading",
        sourceLabel,
        errorMessage: null,
        data: null,
      });

      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const text = await response.text();
        if (activeLoadRequestIdRef.current !== requestId) {
          return;
        }
        applyReadyAsset(parsePointCloudMeshText(text), sourceLabel, {
          frameCamera: true,
        });
      } catch (error) {
        if (activeLoadRequestIdRef.current !== requestId) {
          return;
        }
        setAssetState({
          status: "error",
          sourceLabel,
          errorMessage:
            error instanceof Error ? error.message : fallbackErrorMessage,
          data: null,
        });
      }
    },
    [applyReadyAsset, beginAssetLoadRequest],
  );

  useEffect(() => {
    let isCancelled = false;

    const loadInitialExample = async (): Promise<void> => {
      const requestId = beginAssetLoadRequest();
      setActiveUploadedFileId(null);
      setAssetState({
        status: "loading",
        sourceLabel: bundledMediumSourceLabel,
        errorMessage: null,
        data: null,
      });

      try {
        const response = await fetch(bundledMediumExampleUrl);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = parsePointCloudMeshText(await response.text());
        if (!isCancelled && activeLoadRequestIdRef.current === requestId) {
          applyReadyAsset(data, bundledMediumSourceLabel, {
            frameCamera: true,
          });
        }
      } catch (error) {
        if (isCancelled || activeLoadRequestIdRef.current !== requestId) {
          return;
        }
        setAssetState({
          status: "error",
          sourceLabel: bundledMediumSourceLabel,
          errorMessage:
            error instanceof Error
              ? error.message
              : "Unable to load the bundled medium example.",
          data: null,
        });
      }
    };

    void loadInitialExample();
    return () => {
      isCancelled = true;
    };
  }, [applyReadyAsset, beginAssetLoadRequest]);

  const handleFileUpload = useCallback(
    async (event: ChangeEvent<HTMLInputElement>): Promise<void> => {
      const files = Array.from(event.target.files ?? []);
      if (files.length === 0) {
        return;
      }

      const nextUploadedFiles = files.map((file): UploadedPointCloudFile => {
        const id = nextUploadedFileIdRef.current;
        nextUploadedFileIdRef.current += 1;
        return {
          id,
          file,
          label: file.name,
          status: "idle",
          errorMessage: null,
        };
      });

      setUploadedFiles((currentFiles) => [
        ...currentFiles,
        ...nextUploadedFiles,
      ]);
      event.target.value = "";
      await loadQueuedFile(nextUploadedFiles[0]);
    },
    [loadQueuedFile],
  );

  const removeQueuedFile = useCallback(
    (removedFileId: number): void => {
      const currentIndex = uploadedFiles.findIndex(
        (uploadedFile) => uploadedFile.id === removedFileId,
      );
      if (currentIndex === -1) {
        return;
      }

      const nextUploadedFiles = uploadedFiles.filter(
        (uploadedFile) => uploadedFile.id !== removedFileId,
      );
      setUploadedFiles(nextUploadedFiles);

      if (activeUploadedFileId !== removedFileId) {
        return;
      }

      const nextActiveFile =
        nextUploadedFiles[currentIndex] ??
        nextUploadedFiles[currentIndex - 1] ??
        null;
      if (nextActiveFile !== null) {
        void loadQueuedFile(nextActiveFile);
        return;
      }

      void loadBundledExample(
        bundledMediumExampleUrl,
        bundledMediumSourceLabel,
        "Unable to load the bundled medium example.",
      );
    },
    [activeUploadedFileId, loadBundledExample, loadQueuedFile, uploadedFiles],
  );

  return {
    assetState,
    setAssetState,
    uploadedFiles,
    activeUploadedFileId,
    setActiveUploadedFileId,
    handleFileUpload,
    loadQueuedFile,
    loadTransientFile,
    loadBundledExample,
    removeQueuedFile,
    setUploadedFileStatus,
  };
};
