import {
  startTransition,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type ComponentPropsWithoutRef,
  type ReactNode,
} from "react";
import {
  PointCloudPreviewPointSizeControl,
  PreviewVisibilityToggleButton,
} from "./PreviewViewControls";
import {
  maxFragmentShaderPointsPerCell,
  PointCloudPreviewScene,
} from "./PointCloudPreviewScene";
import { PointCloudHitInspector } from "./PointCloudHitInspector";
import { createZipBlob } from "./createZipBlob";
import { parsePointCloudMeshText } from "./loadPointCloudMesh";
import type {
  MeshColorMode,
  PointCloudMeshData,
  PointCloudPreviewViewAxis,
  PreviewCameraState,
} from "./types";
import { usePointCloudPreviewController } from "./usePointCloudPreviewController";
import { assetUrl } from "../../shared/assetUrls";

const bundledMediumExampleUrl = assetUrl(
  "pointcloud-style-transfer/pc-and-mesh-medium-example.json",
);
const bundledMediumSourceLabel = "Bundled medium example";
const savedViewpointsStorageKey = "pointcloud-preview-saved-viewpoints";

type UploadedPointCloudFileStatus = "idle" | "loading" | "ready" | "error";

type UploadedPointCloudFile = {
  readonly id: number;
  readonly file: File;
  readonly label: string;
  readonly status: UploadedPointCloudFileStatus;
  readonly errorMessage: string | null;
};

type LoadedAssetState =
  | {
      readonly status: "loading";
      readonly sourceLabel: string;
      readonly errorMessage: null;
      readonly data: null;
    }
  | {
      readonly status: "ready";
      readonly sourceLabel: string;
      readonly errorMessage: null;
      readonly data: PointCloudMeshData;
    }
  | {
      readonly status: "error";
      readonly sourceLabel: string;
      readonly errorMessage: string;
      readonly data: null;
    };

type SavedViewpoint = {
  readonly id: number;
  readonly label: string;
  readonly camera: PreviewCameraState;
  readonly swapYZ: boolean;
};

type BatchScreenshotProgress = {
  readonly completed: number;
  readonly total: number;
  readonly label: string;
};

type PendingCommandSignal = {
  readonly commandId: number;
  readonly resolve: () => void;
};

const boundsLabel = (values: readonly [number, number, number]): string =>
  values.map((value) => value.toFixed(3)).join(", ");

const fpsLabel = (framesPerSecond: number): string =>
  framesPerSecond > 0 ? `${framesPerSecond.toFixed(1)} FPS` : "Sampling FPS";

const uploadedFileStatusLabel = (
  status: UploadedPointCloudFileStatus,
): string => (status === "idle" ? "queued" : status);

const brightnessLabel = (brightness: number): string =>
  `${brightness.toFixed(2)}x`;

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

const snapAxisButtons: readonly {
  readonly axis: PointCloudPreviewViewAxis;
  readonly label: string;
}[] = [
  { axis: "pos-x", label: "+X" },
  { axis: "pos-y", label: "+Y" },
  { axis: "pos-z", label: "+Z" },
  { axis: "neg-x", label: "-X" },
  { axis: "neg-y", label: "-Y" },
  { axis: "neg-z", label: "-Z" },
];

const parseSavedViewpointEntries = (
  entries: readonly unknown[],
  reindexIds: boolean,
): SavedViewpoint[] =>
  entries.flatMap((entry, index): SavedViewpoint[] => {
    if (typeof entry !== "object" || entry === null) {
      return [];
    }
    const candidate = entry as Record<string, unknown>;
    if (
      typeof candidate.label !== "string" ||
      typeof candidate.swapYZ !== "boolean" ||
      !isPreviewCameraState(candidate.camera)
    ) {
      return [];
    }
    const preservedId =
      typeof candidate.id === "number" ? candidate.id : index + 1;
    return [
      {
        id: reindexIds ? index + 1 : preservedId,
        label: candidate.label,
        camera: candidate.camera,
        swapYZ: candidate.swapYZ,
      },
    ];
  });

const readSavedViewpoints = (): readonly SavedViewpoint[] => {
  if (typeof window === "undefined") {
    return [];
  }

  try {
    const raw = window.localStorage.getItem(savedViewpointsStorageKey);
    if (raw === null) {
      return [];
    }
    const parsed = JSON.parse(raw) as unknown;
    if (Array.isArray(parsed)) {
      return parseSavedViewpointEntries(parsed, false);
    }
    if (typeof parsed !== "object" || parsed === null) {
      return [];
    }

    const legacyEntries = Object.values(parsed as Record<string, unknown>)
      .filter(Array.isArray)
      .flatMap((value) => value);
    return parseSavedViewpointEntries(legacyEntries, true);
  } catch {
    return [];
  }
};

const isPreviewCameraState = (value: unknown): value is PreviewCameraState => {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Record<string, unknown>;
  return (
    Array.isArray(candidate.position) &&
    candidate.position.length === 3 &&
    candidate.position.every((entry) => typeof entry === "number") &&
    Array.isArray(candidate.target) &&
    candidate.target.length === 3 &&
    candidate.target.every((entry) => typeof entry === "number")
  );
};

const nextViewpointIdFor = (viewpoints: readonly SavedViewpoint[]): number =>
  viewpoints.reduce(
    (highestId, viewpoint) => Math.max(highestId, viewpoint.id),
    0,
  ) + 1;

function InfoTooltip({ text }: { readonly text: string }) {
  return (
    <span
      className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-white/15 bg-white/5 text-[11px] font-semibold text-slate-300"
      title={text}
    >
      i
    </span>
  );
}

function LabelWithTooltip({
  label,
  tooltip,
}: {
  readonly label: string;
  readonly tooltip?: string;
}) {
  return (
    <span className="inline-flex items-center gap-2">
      <span>{label}</span>
      {tooltip !== undefined ? <InfoTooltip text={tooltip} /> : null}
    </span>
  );
}

function CollapsiblePanelCard({
  title,
  children,
  defaultOpen = true,
  testId,
}: {
  readonly title: string;
  readonly children: ReactNode;
  readonly defaultOpen?: boolean;
  readonly testId?: string;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <section
      className="rounded-[1.1rem] border border-white/10 bg-slate-950/55 p-5 shadow-xl shadow-black/20"
      data-testid={testId}
    >
      <button
        className="flex w-full items-center justify-between gap-3 text-left"
        type="button"
        aria-expanded={isOpen}
        onClick={() => setIsOpen((value) => !value)}
      >
        <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
          {title}
        </p>
        <span
          aria-hidden="true"
          className="text-lg font-semibold leading-none text-slate-300"
        >
          {isOpen ? "-" : "+"}
        </span>
      </button>
      {isOpen ? <div className="mt-4">{children}</div> : null}
    </section>
  );
}

function IconButton({
  children,
  className,
  ...props
}: Omit<ComponentPropsWithoutRef<"button">, "className"> & {
  readonly className: string;
  readonly children: ReactNode;
}) {
  return (
    <button
      {...props}
      className={`inline-flex items-center justify-center rounded-xl transition disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
      type={props.type ?? "button"}
    >
      {children}
    </button>
  );
}

function SaveIcon() {
  return (
    <svg
      aria-hidden="true"
      className="h-4 w-4"
      viewBox="0 0 16 16"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M3 2.5H11.5L13.5 4.5V13.5H2.5V3C2.5 2.72386 2.72386 2.5 3 2.5Z"
        stroke="currentColor"
        strokeWidth="1.2"
      />
      <path d="M5 2.5V6H10.5V2.5" stroke="currentColor" strokeWidth="1.2" />
      <path d="M5 13.5V9.5H11V13.5" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg
      aria-hidden="true"
      className="h-4 w-4"
      viewBox="0 0 16 16"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path d="M3.5 4.5H12.5" stroke="currentColor" strokeWidth="1.2" />
      <path
        d="M5 4.5V3.5H11V4.5M4.5 4.5L5 13.5H11L11.5 4.5"
        stroke="currentColor"
        strokeWidth="1.2"
      />
      <path d="M6.5 6.5V11.5" stroke="currentColor" strokeWidth="1.2" />
      <path d="M9.5 6.5V11.5" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}

function ArrowRightIcon() {
  return (
    <svg
      aria-hidden="true"
      className="h-4 w-4"
      viewBox="0 0 16 16"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M3 8H12M12 8L8.5 4.5M12 8L8.5 11.5"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function RotateCcwIcon() {
  return (
    <svg
      aria-hidden="true"
      className="h-4 w-4"
      viewBox="0 0 16 16"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M5.5 6H2.5V3M2.5 6C3.5 4.1 5.55 2.75 7.92 2.75C11.28 2.75 14 5.47 14 8.83C14 12.19 11.28 14.91 7.92 14.91C5.47 14.91 3.35 13.46 2.39 11.37"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function XIcon() {
  return (
    <svg
      aria-hidden="true"
      className="h-4 w-4"
      viewBox="0 0 16 16"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M4.25 4.25L11.75 11.75M11.75 4.25L4.25 11.75"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  );
}

function InfoModal({
  open,
  onClose,
  children,
}: {
  readonly open: boolean;
  readonly onClose: () => void;
  readonly children: ReactNode;
}) {
  if (!open) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 px-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-2xl rounded-[1.1rem] border border-white/10 bg-slate-950 p-6 shadow-2xl shadow-black/40"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
              Preview info
            </p>
            <h2 className="mt-2 text-2xl font-semibold text-white">
              Point-cloud preview route
            </h2>
          </div>
          <button
            className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm font-semibold text-slate-100 transition hover:bg-white/10"
            type="button"
            onClick={onClose}
          >
            Close
          </button>
        </div>
        <div className="mt-4 space-y-4 text-sm leading-6 text-slate-300">
          {children}
        </div>
      </div>
    </div>
  );
}

export function PointCloudPreviewApp() {
  const fileInputId = useId();
  const previewHostRef = useRef<HTMLDivElement | null>(null);
  const nextUploadedFileIdRef = useRef<number>(1);
  const activeLoadRequestIdRef = useRef<number>(0);
  const cameraAppliedWaitersRef = useRef<PendingCommandSignal[]>([]);
  const frameRenderedWaitersRef = useRef<PendingCommandSignal[]>([]);
  const [assetState, setAssetState] = useState<LoadedAssetState>({
    status: "loading",
    sourceLabel: bundledMediumSourceLabel,
    errorMessage: null,
    data: null,
  });
  const {
    viewSettings,
    updateViewSettings,
    selectedHit,
    setSelectedHit,
    framesPerSecond,
    setFramesPerSecond,
    cameraCommand,
    cameraState: currentCameraState,
    cameraStateRef: currentCameraStateRef,
    handleCameraStateChange,
    issueCameraCommand,
    resetPreviewInteraction,
  } = usePointCloudPreviewController();
  const [savedViewpoints, setSavedViewpoints] = useState<
    readonly SavedViewpoint[]
  >(() => readSavedViewpoints());
  const [nextViewpointId, setNextViewpointId] = useState<number>(() =>
    nextViewpointIdFor(readSavedViewpoints()),
  );
  const [showInfoModal, setShowInfoModal] = useState<boolean>(false);
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
  const [uploadedFiles, setUploadedFiles] = useState<
    readonly UploadedPointCloudFile[]
  >([]);
  const [activeUploadedFileId, setActiveUploadedFileId] = useState<
    number | null
  >(null);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(
      savedViewpointsStorageKey,
      JSON.stringify(savedViewpoints),
    );
  }, [savedViewpoints]);

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

  const setUploadedFileStatus = (
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
  };

  const loadQueuedFile = async (
    uploadedFile: UploadedPointCloudFile,
  ): Promise<void> => {
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
      const text = await uploadedFile.file.text();
      const data = parsePointCloudMeshText(text);
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
  };

  const loadBundledExample = async (
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
  };

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
        const text = await response.text();
        const data = parsePointCloudMeshText(text);
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

  const handleFileUpload = async (
    event: ChangeEvent<HTMLInputElement>,
  ): Promise<void> => {
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

    setUploadedFiles((currentFiles) => [...currentFiles, ...nextUploadedFiles]);
    event.target.value = "";
    await loadQueuedFile(nextUploadedFiles[0]);
  };

  const handleQueuedFileRemove = (removedFileId: number): void => {
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
  };

  const handleSaveViewpoint = (): void => {
    if (currentCameraState === null) {
      return;
    }
    const nextId = nextViewpointId;
    setSavedViewpoints((currentViews) => [
      {
        id: nextId,
        label: `View ${nextId}`,
        camera: currentCameraState,
        swapYZ: viewSettings.swapYZ,
      },
      ...currentViews,
    ]);
    setNextViewpointId(nextId + 1);
  };

  const handleRenameViewpoint = (id: number, label: string): void => {
    setSavedViewpoints((currentViews) =>
      currentViews.map((viewpoint) =>
        viewpoint.id === id ? { ...viewpoint, label } : viewpoint,
      ),
    );
  };

  const handleUpdateViewpoint = (id: number): void => {
    if (currentCameraState === null) {
      return;
    }
    setSavedViewpoints((currentViews) =>
      currentViews.map((viewpoint) =>
        viewpoint.id === id
          ? {
              ...viewpoint,
              camera: currentCameraState,
              swapYZ: viewSettings.swapYZ,
            }
          : viewpoint,
      ),
    );
  };

  const handleDeleteViewpoint = (id: number): void => {
    setSavedViewpoints((currentViews) =>
      currentViews.filter((viewpoint) => viewpoint.id !== id),
    );
  };

  const waitForCommandSignal = (
    waitersRef: {
      current: PendingCommandSignal[];
    },
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
    });

  const resolveCommandSignals = useCallback(
    (
      waitersRef: {
        current: PendingCommandSignal[];
      },
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

  const restorePreviewAfterBatch = async ({
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
    const cameraApplied = waitForCommandSignal(
      cameraAppliedWaitersRef,
      commandId,
      "camera restore",
    );
    await cameraApplied;
    await waitForCommandSignal(
      frameRenderedWaitersRef,
      commandId,
      "restored frame",
    );
  };

  const handleBatchScreenshot = async (): Promise<void> => {
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
    const previousSwapYZ = viewSettings.swapYZ;
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
        let meshData: PointCloudMeshData;
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
          const completed = zipEntries.length;
          setBatchScreenshotProgress({
            completed,
            total,
            label: `${uploadedFile.label} at ${viewpoint.label}`,
          });
          updateViewSettings({ swapYZ: viewpoint.swapYZ });
          const commandId = issueCameraCommand({
            type: "restore",
            camera: viewpoint.camera,
          });
          const cameraApplied = waitForCommandSignal(
            cameraAppliedWaitersRef,
            commandId,
            "camera update",
          );
          await cameraApplied;
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
        // Keep the original batch error, which is more actionable to the user.
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
  };

  const handleScreenshot = (): void => {
    const canvas = previewHostRef.current?.querySelector("canvas");
    if (!(canvas instanceof HTMLCanvasElement)) {
      return;
    }

    const downloadLink = document.createElement("a");
    downloadLink.href = canvas.toDataURL("image/png");
    downloadLink.download = `${screenshotFilenameStemForSourceLabel(assetState.sourceLabel)}-screenshot-${screenshotTimestampLabel(new Date())}.png`;
    downloadLink.click();
  };

  const data = assetState.data;
  const canUseFragmentKnnShading = useMemo(
    () =>
      data !== null &&
      data.spatialHash.maxPointsPerCell <= maxFragmentShaderPointsPerCell,
    [data],
  );
  const effectiveMeshColorMode: MeshColorMode =
    viewSettings.meshColorMode === "fragment-knn" && canUseFragmentKnnShading
      ? "fragment-knn"
      : "baked";
  const meshColorModeDescription =
    effectiveMeshColorMode === "fragment-knn"
      ? `Spatial-hash fragment KNN shading active (${data?.pointCount ?? 0} points across ${data?.spatialHash.cellCount ?? 0} cells).`
      : viewSettings.meshColorMode === "fragment-knn" && data !== null
        ? `Spatial-hash fragment KNN shading found a dense cell with ${data.spatialHash.maxPointsPerCell} points, which exceeds the current per-cell shader cap of ${maxFragmentShaderPointsPerCell}; this dataset falls back to baked vertex colours.`
        : !viewSettings.disableGammaDecoding
          ? "Baked vertex colours active with gamma decoding enabled."
          : "Baked vertex colours active with gamma decoding disabled.";

  return (
    <>
      <main className="h-screen w-screen overflow-hidden bg-[linear-gradient(180deg,_#071019_0%,_#0b1120_100%)] text-slate-100">
        <div className="flex h-full w-full flex-col px-4 py-4">
          <header className="mb-4 flex shrink-0 items-center justify-between rounded-[1.1rem] border border-white/10 bg-slate-950/65 px-5 py-4 shadow-xl shadow-black/20">
            <div className="flex items-center gap-6">
              <h1 className="text-2xl font-semibold tracking-tight text-white">
                Point-Cloud Mesh Preview
              </h1>
              <nav className="flex items-center gap-3 text-sm font-medium">
                <a
                  className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-slate-100 transition hover:bg-white/10"
                  href={assetUrl("")}
                >
                  Main app
                </a>
                <a
                  className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-slate-100 transition hover:bg-white/10"
                  href={assetUrl("benchmark")}
                >
                  Benchmark
                </a>
              </nav>
            </div>
            <button
              className="rounded-xl border border-sky-300/20 bg-sky-400/10 px-4 py-2 text-sm font-semibold text-sky-100 transition hover:bg-sky-400/20"
              type="button"
              onClick={() => setShowInfoModal(true)}
            >
              Info
            </button>
          </header>

          <div className="flex min-h-0 flex-1 items-stretch gap-4">
            <div className="flex h-full w-[25rem] shrink-0 flex-col gap-4 overflow-y-auto pr-1">
              <section className="rounded-[1.1rem] border border-white/10 bg-slate-950/55 p-5 shadow-xl shadow-black/20">
                <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
                  Rendering algorithm
                </p>
                <div className="mt-4">
                  <select
                    data-testid="mesh-color-mode-select"
                    className="w-full rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-4 py-3 text-sm text-slate-100 outline-none transition focus:border-amber-300"
                    value={viewSettings.meshColorMode}
                    onChange={(event) =>
                      updateViewSettings({
                        meshColorMode: event.target.value as MeshColorMode,
                      })
                    }
                  >
                    <option value="fragment-knn">Fragment KNN shading</option>
                    <option value="baked">Baked vertex colours</option>
                  </select>
                </div>
                <div
                  data-testid="mesh-color-mode-status"
                  className="mt-4 rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-4 py-3 text-sm text-slate-300"
                >
                  {meshColorModeDescription}
                </div>
              </section>

              <section className="rounded-[1.1rem] border border-white/10 bg-slate-950/55 p-5 shadow-xl shadow-black/20">
                <div className="flex items-center gap-2">
                  <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
                    Data source
                  </p>
                  <InfoTooltip text="Load the bundled medium preview or upload a point-cloud mesh JSON export from the Python pipeline." />
                </div>
                <div className="mt-4 grid grid-cols-[minmax(0,1fr)_auto] gap-3">
                  <label
                    className="flex cursor-pointer items-center justify-center rounded-xl bg-amber-300 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-amber-200"
                    htmlFor={fileInputId}
                  >
                    Upload point-cloud mesh
                  </label>
                  <button
                    className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-lg font-semibold text-slate-100 transition hover:bg-white/10"
                    type="button"
                    aria-label="Reload preview"
                    title="Reload preview"
                    onClick={() => {
                      void loadBundledExample(
                        bundledMediumExampleUrl,
                        bundledMediumSourceLabel,
                        "Unable to reload the bundled medium example.",
                      );
                    }}
                  >
                    <RotateCcwIcon />
                  </button>
                  <input
                    id={fileInputId}
                    className="sr-only"
                    type="file"
                    multiple
                    accept=".json,application/json"
                    onChange={(event) => {
                      void handleFileUpload(event);
                    }}
                  />
                </div>
                <div className="mt-4 space-y-2 rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-4 py-3 text-sm text-slate-200">
                  <div className="flex items-center justify-between gap-4">
                    <span className="text-slate-300">Current source</span>
                    <span
                      data-testid="pointcloud-source-label"
                      className="text-right font-semibold text-white"
                    >
                      {assetState.sourceLabel}
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-4">
                    <span className="text-slate-300">Status</span>
                    <span
                      data-testid="pointcloud-load-status"
                      className="text-right font-semibold capitalize text-white"
                    >
                      {assetState.status}
                    </span>
                  </div>
                  {assetState.errorMessage !== null ? (
                    <p className="rounded-[0.8rem] border border-rose-400/20 bg-rose-500/10 px-3 py-2 text-rose-200">
                      {assetState.errorMessage}
                    </p>
                  ) : null}
                </div>
                <div
                  data-testid="pointcloud-upload-list"
                  className="mt-4 max-h-[30rem] space-y-2 overflow-y-auto rounded-[0.95rem] border border-white/10 bg-slate-900/75 p-2"
                >
                  {uploadedFiles.length > 0 ? (
                    uploadedFiles.map((uploadedFile) => {
                      const isActive = activeUploadedFileId === uploadedFile.id;
                      return (
                        <div
                          key={uploadedFile.id}
                          data-testid={`pointcloud-upload-row-${uploadedFile.id}`}
                          className={`grid grid-cols-[minmax(0,1fr)_auto] items-stretch gap-2 rounded-[0.8rem] border p-1 ${
                            isActive
                              ? "border-amber-300/60 bg-amber-300/10"
                              : "border-white/10 bg-slate-950/45"
                          }`}
                        >
                          <button
                            data-testid={`pointcloud-upload-select-${uploadedFile.id}`}
                            className="min-w-0 rounded-[0.65rem] px-3 py-2 text-left transition hover:bg-white/10"
                            type="button"
                            aria-current={isActive ? "true" : undefined}
                            title={uploadedFile.label}
                            onClick={() => {
                              void loadQueuedFile(uploadedFile);
                            }}
                          >
                            <span className="block truncate text-sm font-semibold text-white">
                              {uploadedFile.label}
                            </span>
                            <span
                              data-testid={`pointcloud-upload-status-${uploadedFile.id}`}
                              className={`mt-1 block text-xs capitalize ${
                                uploadedFile.status === "error"
                                  ? "text-rose-200"
                                  : uploadedFile.status === "loading"
                                    ? "text-amber-100"
                                    : "text-slate-400"
                              }`}
                            >
                              {uploadedFileStatusLabel(uploadedFile.status)}
                            </span>
                          </button>
                          <button
                            data-testid={`pointcloud-upload-remove-${uploadedFile.id}`}
                            className="inline-flex h-10 w-10 items-center justify-center self-center rounded-[0.65rem] border border-white/10 bg-white/5 text-slate-200 transition hover:border-rose-300/40 hover:bg-rose-400/10 hover:text-rose-100"
                            type="button"
                            aria-label={`Remove ${uploadedFile.label}`}
                            title={`Remove ${uploadedFile.label}`}
                            onClick={() =>
                              handleQueuedFileRemove(uploadedFile.id)
                            }
                          >
                            <XIcon />
                          </button>
                        </div>
                      );
                    })
                  ) : (
                    <p className="px-3 py-2 text-sm text-slate-400">
                      No uploads queued.
                    </p>
                  )}
                </div>
              </section>

              <section className="rounded-[1.1rem] border border-white/10 bg-slate-950/55 p-5 shadow-xl shadow-black/20">
                <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
                  Dataset stats
                </p>
                {data !== null ? (
                  <div className="mt-4 overflow-hidden rounded-[0.95rem] border border-white/10 bg-slate-900/75">
                    <div className="flex items-center justify-between gap-4 px-4 py-3 text-sm text-slate-200">
                      <span>Mesh vertices</span>
                      <span
                        data-testid="mesh-vertex-count"
                        className="text-right font-semibold text-white"
                      >
                        {data.meshVertexCount}
                      </span>
                    </div>
                    <div className="h-px bg-white/10" />
                    <div className="flex items-center justify-between gap-4 px-4 py-3 text-sm text-slate-200">
                      <span>Mesh faces</span>
                      <span
                        data-testid="mesh-face-count"
                        className="text-right font-semibold text-white"
                      >
                        {data.meshFaceCount}
                      </span>
                    </div>
                    <div className="h-px bg-white/10" />
                    <div className="flex items-center justify-between gap-4 px-4 py-3 text-sm text-slate-200">
                      <LabelWithTooltip
                        label="Point samples"
                        tooltip="The number of aligned point-cloud samples available for interpolation and fragment shading."
                      />
                      <span
                        data-testid="point-sample-count"
                        className="text-right font-semibold text-white"
                      >
                        {data.pointCount}
                      </span>
                    </div>
                    <div className="h-px bg-white/10" />
                    <div className="flex items-center justify-between gap-4 px-4 py-3 text-sm text-slate-200">
                      <LabelWithTooltip
                        label="Bounds radius"
                        tooltip="The preview radius derived from the mesh bounding sphere. It drives framing and size scaling."
                      />
                      <span className="text-right font-semibold text-white">
                        {data.bounds.radius.toFixed(3)}
                      </span>
                    </div>
                    <div className="h-px bg-white/10" />
                    <div className="flex items-start justify-between gap-4 px-4 py-3 text-sm text-slate-200">
                      <LabelWithTooltip
                        label="Bounds"
                        tooltip="Axis-aligned min and max coordinates of the mesh positions used for this preview."
                      />
                      <span className="text-right font-mono text-[11px] leading-5 text-slate-100">
                        min [{boundsLabel(data.bounds.min)}]
                        <br />
                        max [{boundsLabel(data.bounds.max)}]
                      </span>
                    </div>
                  </div>
                ) : (
                  <p className="mt-4 text-sm text-slate-300">
                    Load a dataset to inspect mesh and point-cloud counts.
                  </p>
                )}
              </section>
            </div>

            <section className="flex min-w-0 min-h-0 flex-1 flex-col rounded-[1.1rem] border border-white/10 bg-slate-950/55 p-4 shadow-xl shadow-black/20">
              <div className="mb-3 flex items-center justify-between gap-3 px-1">
                <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
                  Interactive preview
                </p>
                <div className="flex items-center gap-3">
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
                    onClick={handleScreenshot}
                  >
                    Screenshot
                  </button>
                  {uploadedFiles.length > 1 ? (
                    <button
                      data-testid="batch-screenshot-button"
                      className="rounded-xl border border-sky-300/20 bg-sky-400/10 px-4 py-2 text-sm font-semibold text-sky-100 transition hover:bg-sky-400/20"
                      type="button"
                      onClick={() => {
                        setSelectedBatchViewpointIds([]);
                        setBatchScreenshotError(null);
                        setBatchScreenshotProgress(null);
                        setShowBatchScreenshotModal(true);
                      }}
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
                    pointGammaCorrection={!viewSettings.disableGammaDecoding}
                    brightness={viewSettings.brightness}
                    swapYZ={viewSettings.swapYZ}
                    selectedHit={selectedHit}
                    cameraCommand={cameraCommand}
                    onHoverSampleChange={setSelectedHit}
                    onCameraStateChange={handleCameraStateChange}
                    onCameraCommandApplied={handleCameraCommandApplied}
                    onFrameRendered={handlePreviewFrameRendered}
                    onFramesPerSecondChange={setFramesPerSecond}
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

            <div
              data-testid="pointcloud-right-panel"
              className="flex h-full w-[25rem] shrink-0 flex-col gap-4 overflow-y-auto pr-1"
            >
              <CollapsiblePanelCard
                title="View options"
                testId="view-options-card"
              >
                <div className="space-y-3 text-sm text-slate-200">
                  <div className="grid gap-3 sm:grid-cols-2">
                    <PreviewVisibilityToggleButton
                      label="Mesh"
                      testId="toggle-mesh-button"
                      visible={viewSettings.showMesh}
                      onClick={() =>
                        updateViewSettings({ showMesh: !viewSettings.showMesh })
                      }
                    />
                    <PreviewVisibilityToggleButton
                      label="Wireframe"
                      testId="toggle-wireframe-button"
                      visible={viewSettings.showWireframe}
                      disabled={!viewSettings.showMesh}
                      onClick={() =>
                        updateViewSettings({
                          showWireframe: !viewSettings.showWireframe,
                        })
                      }
                    />
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <PreviewVisibilityToggleButton
                      label="Point cloud"
                      testId="toggle-points-button"
                      visible={viewSettings.showPoints}
                      onClick={() =>
                        updateViewSettings({
                          showPoints: !viewSettings.showPoints,
                        })
                      }
                    />
                    <PointCloudPreviewPointSizeControl
                      viewSettings={viewSettings}
                      testId="point-size-slider"
                      onPointSizeChange={(pointSize) =>
                        updateViewSettings({ pointSize })
                      }
                    />
                  </div>
                </div>
              </CollapsiblePanelCard>

              <CollapsiblePanelCard
                title="Shading options"
                testId="shading-options-card"
              >
                <div className="space-y-4 text-sm text-slate-200">
                  <label className="flex items-center gap-3 rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-4 py-3">
                    <input
                      className="accent-amber-300"
                      type="checkbox"
                      checked={viewSettings.disableGammaDecoding}
                      onChange={(event) =>
                        updateViewSettings({
                          disableGammaDecoding: event.target.checked,
                        })
                      }
                    />
                    Disable gamma decoding
                  </label>
                  <label className="block rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-4 py-4">
                    <span className="mb-2 block text-slate-300">
                      Brightness: {brightnessLabel(viewSettings.brightness)}
                    </span>
                    <input
                      className="w-full accent-amber-300"
                      type="range"
                      min={0.5}
                      max={1.8}
                      step={0.05}
                      value={viewSettings.brightness}
                      onChange={(event) =>
                        updateViewSettings({
                          brightness: Number(event.target.value),
                        })
                      }
                    />
                  </label>
                </div>
              </CollapsiblePanelCard>

              <CollapsiblePanelCard
                title="Camera and orientation"
                testId="camera-card"
              >
                <div className="flex flex-wrap gap-3">
                  <button
                    data-testid="swap-yz-button"
                    className={`rounded-xl px-4 py-2 text-sm font-semibold transition ${
                      viewSettings.swapYZ
                        ? "bg-amber-300 text-slate-950 hover:bg-amber-200"
                        : "border border-white/10 bg-white/5 text-slate-100 hover:bg-white/10"
                    }`}
                    type="button"
                    onClick={() =>
                      updateViewSettings({
                        swapYZ: !viewSettings.swapYZ,
                      })
                    }
                  >
                    {viewSettings.swapYZ ? "Y/Z swapped" : "Flip Y and Z"}
                  </button>
                  <button
                    className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-slate-100 transition hover:bg-white/10"
                    type="button"
                    onClick={() => issueCameraCommand({ type: "frame" })}
                  >
                    Reset camera
                  </button>
                </div>

                <div className="mt-4 grid grid-cols-3 gap-2">
                  {snapAxisButtons.map((button) => (
                    <button
                      key={button.axis}
                      data-testid={`snap-axis-${button.axis}`}
                      className="rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-3 py-3 text-sm font-semibold text-slate-100 transition hover:bg-white/10"
                      type="button"
                      onClick={() =>
                        issueCameraCommand({
                          type: "snap-axis",
                          axis: button.axis,
                        })
                      }
                    >
                      {button.label}
                    </button>
                  ))}
                </div>
                <output
                  data-testid="camera-state"
                  className="sr-only"
                  aria-live="off"
                >
                  {currentCameraState === null
                    ? "unavailable"
                    : JSON.stringify(currentCameraState)}
                </output>

                <div className="mt-4 rounded-[0.95rem] border border-white/10 bg-slate-900/75 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-semibold text-white">
                      Saved viewpoints
                    </span>
                    <IconButton
                      data-testid="save-viewpoint-button"
                      className="h-10 w-10 border border-white/10 bg-white/5 text-slate-100 hover:bg-white/10"
                      disabled={currentCameraState === null}
                      aria-label="Save current viewpoint"
                      title="Save current viewpoint"
                      onClick={handleSaveViewpoint}
                    >
                      <SaveIcon />
                      <span className="sr-only">Save current viewpoint</span>
                    </IconButton>
                  </div>
                  {savedViewpoints.length > 0 ? (
                    <div className="mt-3 flex flex-col gap-3">
                      {savedViewpoints.map((viewpoint) => (
                        <div
                          key={viewpoint.id}
                          className="rounded-[0.95rem] border border-white/10 bg-slate-950/40 p-3"
                        >
                          <div className="min-w-0">
                            <div className="flex items-center gap-3">
                              <input
                                data-testid={`viewpoint-name-${viewpoint.id}`}
                                className="min-w-0 flex-1 rounded-[0.8rem] border border-white/10 bg-slate-900/75 px-3 py-2 text-sm font-semibold text-white outline-none transition focus:border-amber-300"
                                value={viewpoint.label}
                                onChange={(event) =>
                                  handleRenameViewpoint(
                                    viewpoint.id,
                                    event.target.value,
                                  )
                                }
                              />
                              {viewpoint.swapYZ ? (
                                <span className="rounded-full border border-amber-300/20 bg-amber-300/10 px-2 py-1 text-[11px] font-semibold uppercase tracking-[0.15em] text-amber-200">
                                  Y/Z
                                </span>
                              ) : null}
                            </div>
                            <div className="mt-3 grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] gap-2">
                              <button
                                data-testid={`viewpoint-go-${viewpoint.id}`}
                                className="inline-flex items-center justify-center gap-2 rounded-xl border border-emerald-300/30 bg-emerald-400/10 px-4 py-2 text-sm font-semibold text-emerald-100 transition hover:bg-emerald-400/20"
                                type="button"
                                onClick={() => {
                                  updateViewSettings({
                                    swapYZ: viewpoint.swapYZ,
                                  });
                                  issueCameraCommand({
                                    type: "restore",
                                    camera: viewpoint.camera,
                                  });
                                }}
                              >
                                <span>Go</span>
                                <ArrowRightIcon />
                              </button>
                              <button
                                data-testid={`viewpoint-update-${viewpoint.id}`}
                                className="inline-flex items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-slate-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
                                type="button"
                                disabled={currentCameraState === null}
                                title="Update this saved viewpoint to match the current camera"
                                onClick={() =>
                                  handleUpdateViewpoint(viewpoint.id)
                                }
                              >
                                <SaveIcon />
                                <span>Update</span>
                              </button>
                              <IconButton
                                data-testid={`viewpoint-delete-${viewpoint.id}`}
                                className="h-10 w-10 border border-rose-300/20 bg-rose-400/10 text-rose-100 hover:bg-rose-400/20"
                                aria-label={`Delete ${viewpoint.label}`}
                                title="Delete viewpoint"
                                onClick={() =>
                                  handleDeleteViewpoint(viewpoint.id)
                                }
                              >
                                <TrashIcon />
                              </IconButton>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="mt-3 text-sm text-slate-400">
                      Save a viewpoint to keep a reusable camera preset for any
                      dataset you load here.
                    </p>
                  )}
                </div>
              </CollapsiblePanelCard>
            </div>
          </div>
        </div>
      </main>

      {showBatchScreenshotModal ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/75 px-4"
          onClick={() => {
            if (!isBatchScreenshotRunning) {
              setShowBatchScreenshotModal(false);
            }
          }}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="batch-screenshot-title"
            data-testid="batch-screenshot-modal"
            className="w-full max-w-xl rounded-[1.1rem] border border-white/10 bg-slate-950 p-6 shadow-2xl shadow-black/40"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
                  Batch export
                </p>
                <h2
                  id="batch-screenshot-title"
                  className="mt-2 text-2xl font-semibold text-white"
                >
                  Select saved viewpoints
                </h2>
                <p className="mt-2 text-sm leading-6 text-slate-300">
                  Capture {uploadedFiles.length} uploaded meshes at each
                  selected viewpoint and download one ZIP archive.
                </p>
              </div>
              <button
                className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm font-semibold text-slate-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
                type="button"
                disabled={isBatchScreenshotRunning}
                onClick={() => setShowBatchScreenshotModal(false)}
              >
                Close
              </button>
            </div>

            {savedViewpoints.length > 0 ? (
              <>
                <div className="mt-5 flex items-center justify-between gap-3">
                  <span className="text-sm font-semibold text-slate-200">
                    {selectedBatchViewpointIds.length} of{" "}
                    {savedViewpoints.length} selected
                  </span>
                  <button
                    data-testid="batch-screenshot-select-all"
                    className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm font-semibold text-slate-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
                    type="button"
                    disabled={isBatchScreenshotRunning}
                    onClick={() =>
                      setSelectedBatchViewpointIds(
                        selectedBatchViewpointIds.length ===
                          savedViewpoints.length
                          ? []
                          : savedViewpoints.map((viewpoint) => viewpoint.id),
                      )
                    }
                  >
                    {selectedBatchViewpointIds.length === savedViewpoints.length
                      ? "Clear all"
                      : "Select all"}
                  </button>
                </div>
                <div className="mt-3 max-h-72 space-y-2 overflow-y-auto">
                  {savedViewpoints.map((viewpoint) => (
                    <label
                      key={viewpoint.id}
                      className="flex cursor-pointer items-center gap-3 rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-4 py-3 text-sm text-slate-100"
                    >
                      <input
                        data-testid={`batch-viewpoint-${viewpoint.id}`}
                        className="accent-amber-300"
                        type="checkbox"
                        disabled={isBatchScreenshotRunning}
                        checked={selectedBatchViewpointIds.includes(
                          viewpoint.id,
                        )}
                        onChange={(event) =>
                          setSelectedBatchViewpointIds((currentIds) =>
                            event.target.checked
                              ? [...currentIds, viewpoint.id]
                              : currentIds.filter((id) => id !== viewpoint.id),
                          )
                        }
                      />
                      <span className="min-w-0 flex-1 truncate font-semibold">
                        {viewpoint.label}
                      </span>
                      {viewpoint.swapYZ ? (
                        <span className="text-xs font-semibold uppercase tracking-[0.15em] text-amber-200">
                          Y/Z
                        </span>
                      ) : null}
                    </label>
                  ))}
                </div>
              </>
            ) : (
              <p
                data-testid="batch-screenshot-empty"
                className="mt-5 rounded-[0.95rem] border border-amber-300/20 bg-amber-300/10 px-4 py-3 text-sm text-amber-100"
              >
                Save at least one viewpoint before starting a batch export.
              </p>
            )}

            {batchScreenshotProgress !== null ? (
              <div
                data-testid="batch-screenshot-progress"
                className="mt-4 rounded-[0.95rem] border border-sky-300/20 bg-sky-400/10 px-4 py-3 text-sm text-sky-100"
              >
                <div className="flex justify-between gap-3 font-semibold">
                  <span>{batchScreenshotProgress.label}</span>
                  <span>
                    {batchScreenshotProgress.completed}/
                    {batchScreenshotProgress.total}
                  </span>
                </div>
              </div>
            ) : null}
            {batchScreenshotError !== null ? (
              <p
                data-testid="batch-screenshot-error"
                className="mt-4 rounded-[0.95rem] border border-rose-300/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-100"
              >
                {batchScreenshotError}
              </p>
            ) : null}

            <button
              data-testid="batch-screenshot-download"
              className="mt-5 w-full rounded-xl bg-amber-300 px-4 py-3 text-sm font-semibold text-slate-950 transition hover:bg-amber-200 disabled:cursor-not-allowed disabled:opacity-50"
              type="button"
              disabled={
                isBatchScreenshotRunning ||
                savedViewpoints.length === 0 ||
                selectedBatchViewpointIds.length === 0
              }
              onClick={() => {
                void handleBatchScreenshot();
              }}
            >
              {isBatchScreenshotRunning
                ? "Capturing screenshots..."
                : `Download ${uploadedFiles.length * selectedBatchViewpointIds.length} screenshots as ZIP`}
            </button>
          </div>
        </div>
      ) : null}

      <InfoModal open={showInfoModal} onClose={() => setShowInfoModal(false)}>
        <p>
          This route renders a mesh with precomputed 3-nearest-neighbor surface
          colours, overlays the aligned point cloud, and shows the exact
          neighbour blend at mesh hit points without introducing differentiable
          rendering into the web app.
        </p>
        <p>
          Use the rendering algorithm controls to compare fragment-space KNN
          shading against baked vertex colours, then use the shading controls to
          inspect gamma and brightness differences between point and mesh
          displays.
        </p>
        <p>
          Saved viewpoints persist in local storage for this browser across all
          datasets on this route. Screenshot downloads the current canvas as a
          PNG, while batch screenshot cycles queued uploads through selected
          viewpoints and downloads one ZIP.
        </p>
      </InfoModal>
    </>
  );
}
