import {
  startTransition,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type ReactNode,
} from "react";
import {
  maxFragmentShaderPointsPerCell,
  PointCloudPreviewScene,
  type PointCloudPreviewCameraCommand,
} from "./PointCloudPreviewScene";
import { parsePointCloudMeshText } from "./loadPointCloudMesh";
import type {
  PointCloudHitSample,
  PointCloudMeshData,
  PreviewCameraState,
} from "./types";
import { assetUrl } from "../../shared/assetUrls";

const bundledMediumExampleUrl = assetUrl(
  "pointcloud-style-transfer/pc-and-mesh-medium-example.json",
);
const bundledMediumSourceLabel = "Bundled medium example";
const savedViewpointsStorageKey = "pointcloud-preview-saved-viewpoints";

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

type MeshColorMode = "baked" | "fragment-knn";
type ViewAxis = "pos-x" | "neg-x" | "pos-y" | "neg-y" | "pos-z" | "neg-z";

type SavedViewpoint = {
  readonly id: number;
  readonly label: string;
  readonly camera: PreviewCameraState;
  readonly swapYZ: boolean;
};

type ViewSettings = {
  readonly showMesh: boolean;
  readonly showPoints: boolean;
  readonly showWireframe: boolean;
  readonly meshColorMode: MeshColorMode;
  readonly pointSize: number;
  readonly disableGammaDecoding: boolean;
  readonly brightness: number;
  readonly swapYZ: boolean;
};

type NextCameraCommand =
  | {
      readonly type: "frame";
    }
  | {
      readonly type: "restore";
      readonly camera: PreviewCameraState;
    }
  | {
      readonly type: "snap-axis";
      readonly axis: ViewAxis;
    };

const defaultViewSettings: ViewSettings = {
  showMesh: true,
  showPoints: true,
  showWireframe: false,
  meshColorMode: "fragment-knn",
  pointSize: 0.035,
  disableGammaDecoding: false,
  brightness: 1,
  swapYZ: false,
};

const colorToCss = (color: readonly [number, number, number]): string =>
  `rgb(${Math.round(color[0] * 255)} ${Math.round(color[1] * 255)} ${Math.round(
    color[2] * 255,
  )})`;

const boundsLabel = (values: readonly [number, number, number]): string =>
  values.map((value) => value.toFixed(3)).join(", ");

const fpsLabel = (framesPerSecond: number): string =>
  framesPerSecond > 0 ? `${framesPerSecond.toFixed(1)} FPS` : "Sampling FPS";

const brightnessLabel = (brightness: number): string =>
  `${brightness.toFixed(2)}x`;

const snapAxisButtons: readonly {
  readonly axis: ViewAxis;
  readonly label: string;
}[] = [
  { axis: "pos-x", label: "+X" },
  { axis: "pos-y", label: "+Y" },
  { axis: "pos-z", label: "+Z" },
  { axis: "neg-x", label: "-X" },
  { axis: "neg-y", label: "-Y" },
  { axis: "neg-z", label: "-Z" },
];

const readSavedViewpointMap = (): Record<string, readonly SavedViewpoint[]> => {
  if (typeof window === "undefined") {
    return {};
  }

  try {
    const raw = window.localStorage.getItem(savedViewpointsStorageKey);
    if (raw === null) {
      return {};
    }
    const parsed = JSON.parse(raw) as unknown;
    if (typeof parsed !== "object" || parsed === null) {
      return {};
    }

    const output: Record<string, readonly SavedViewpoint[]> = {};
    Object.entries(parsed as Record<string, unknown>).forEach(
      ([key, value]) => {
        if (!Array.isArray(value)) {
          return;
        }
        output[key] = value.flatMap((entry): SavedViewpoint[] => {
          if (typeof entry !== "object" || entry === null) {
            return [];
          }
          const candidate = entry as Record<string, unknown>;
          if (
            typeof candidate.id !== "number" ||
            typeof candidate.label !== "string" ||
            typeof candidate.swapYZ !== "boolean" ||
            !isPreviewCameraState(candidate.camera)
          ) {
            return [];
          }
          return [
            {
              id: candidate.id,
              label: candidate.label,
              camera: candidate.camera,
              swapYZ: candidate.swapYZ,
            },
          ];
        });
      },
    );
    return output;
  } catch {
    return {};
  }
};

const readSavedViewpointsForSource = (
  sourceLabel: string,
): readonly SavedViewpoint[] => readSavedViewpointMap()[sourceLabel] ?? [];
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
  const [assetState, setAssetState] = useState<LoadedAssetState>({
    status: "loading",
    sourceLabel: bundledMediumSourceLabel,
    errorMessage: null,
    data: null,
  });
  const [viewSettings, setViewSettings] =
    useState<ViewSettings>(defaultViewSettings);
  const [selectedHit, setSelectedHit] = useState<PointCloudHitSample | null>(
    null,
  );
  const [cameraCommand, setCameraCommand] =
    useState<PointCloudPreviewCameraCommand>({
      id: 0,
      type: "frame",
    });
  const [currentCameraState, setCurrentCameraState] =
    useState<PreviewCameraState | null>(null);
  const [savedViewpoints, setSavedViewpoints] = useState<
    readonly SavedViewpoint[]
  >([]);
  const [nextViewpointId, setNextViewpointId] = useState<number>(1);
  const [framesPerSecond, setFramesPerSecond] = useState<number>(0);
  const [showInfoModal, setShowInfoModal] = useState<boolean>(false);

  useEffect(() => {
    if (assetState.status !== "ready") {
      return;
    }
    const currentSourceLabel = assetState.sourceLabel;
    const savedViewpointMap = readSavedViewpointMap();
    window.localStorage.setItem(
      savedViewpointsStorageKey,
      JSON.stringify({
        ...savedViewpointMap,
        [currentSourceLabel]: savedViewpoints,
      }),
    );
  }, [assetState.sourceLabel, assetState.status, savedViewpoints]);

  const issueCameraCommand = (
    nextCommand: NextCameraCommand | ((currentId: number) => NextCameraCommand),
  ): void => {
    setCameraCommand((currentCommand) => {
      const resolvedCommand =
        typeof nextCommand === "function"
          ? nextCommand(currentCommand.id)
          : nextCommand;
      return {
        ...resolvedCommand,
        id: currentCommand.id + 1,
      };
    });
  };

  const updateViewSettings = (patch: Partial<ViewSettings>): void => {
    setViewSettings((currentSettings) => ({
      ...currentSettings,
      ...patch,
    }));
  };

  const applyReadyAsset = (
    data: PointCloudMeshData,
    sourceLabel: string,
  ): void => {
    const savedViewpointsForSource = readSavedViewpointsForSource(sourceLabel);
    startTransition(() => {
      setAssetState({
        status: "ready",
        sourceLabel,
        errorMessage: null,
        data,
      });
      setSelectedHit(null);
      setSavedViewpoints(savedViewpointsForSource);
      setNextViewpointId(
        savedViewpointsForSource.reduce(
          (highestId, viewpoint) => Math.max(highestId, viewpoint.id),
          0,
        ) + 1,
      );
      setCurrentCameraState(null);
      setFramesPerSecond(0);
      setViewSettings((currentSettings) => ({
        ...currentSettings,
        swapYZ: false,
      }));
      issueCameraCommand({ type: "frame" });
    });
  };

  const loadBundledExample = async (
    url: string,
    sourceLabel: string,
    fallbackErrorMessage: string,
  ): Promise<void> => {
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
      applyReadyAsset(parsePointCloudMeshText(text), sourceLabel);
    } catch (error) {
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
        if (!isCancelled) {
          const savedViewpointsForSource = readSavedViewpointsForSource(
            bundledMediumSourceLabel,
          );
          startTransition(() => {
            setAssetState({
              status: "ready",
              sourceLabel: bundledMediumSourceLabel,
              errorMessage: null,
              data,
            });
            setSelectedHit(null);
            setSavedViewpoints(savedViewpointsForSource);
            setNextViewpointId(
              savedViewpointsForSource.reduce(
                (highestId, viewpoint) => Math.max(highestId, viewpoint.id),
                0,
              ) + 1,
            );
            setCurrentCameraState(null);
            setFramesPerSecond(0);
            setViewSettings((currentSettings) => ({
              ...currentSettings,
              swapYZ: false,
            }));
            issueCameraCommand({ type: "frame" });
          });
        }
      } catch (error) {
        if (isCancelled) {
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
  }, []);

  const handleFileUpload = async (
    event: ChangeEvent<HTMLInputElement>,
  ): Promise<void> => {
    const file = event.target.files?.[0] ?? null;
    if (file === null) {
      return;
    }

    setAssetState({
      status: "loading",
      sourceLabel: file.name,
      errorMessage: null,
      data: null,
    });

    try {
      const text = await file.text();
      applyReadyAsset(parsePointCloudMeshText(text), file.name);
    } catch (error) {
      setAssetState({
        status: "error",
        sourceLabel: file.name,
        errorMessage:
          error instanceof Error
            ? error.message
            : "Unable to parse uploaded JSON.",
        data: null,
      });
    } finally {
      event.target.value = "";
    }
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

  const handleScreenshot = (): void => {
    const canvas = previewHostRef.current?.querySelector("canvas");
    if (!(canvas instanceof HTMLCanvasElement)) {
      return;
    }

    const downloadLink = document.createElement("a");
    downloadLink.href = canvas.toDataURL("image/png");
    downloadLink.download = `pointcloud-preview-${Date.now()}.png`;
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
      <main className="min-h-screen w-screen bg-[linear-gradient(180deg,_#071019_0%,_#0b1120_100%)] text-slate-100">
        <div className="w-full px-4 py-4">
          <header className="mb-4 flex items-center justify-between rounded-[1.1rem] border border-white/10 bg-slate-950/65 px-5 py-4 shadow-xl shadow-black/20">
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

          <div className="flex items-stretch gap-4">
            <div className="flex w-[25rem] shrink-0 flex-col gap-4">
              <section className="rounded-[1.1rem] border border-white/10 bg-slate-950/55 p-5 shadow-xl shadow-black/20">
                <div className="flex items-center gap-2">
                  <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
                    Data source
                  </p>
                  <InfoTooltip text="Load the bundled medium preview or upload a point-cloud mesh JSON export from the Python pipeline." />
                </div>
                <div className="mt-4 flex flex-wrap gap-3">
                  <label
                    className="cursor-pointer rounded-xl bg-amber-300 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-amber-200"
                    htmlFor={fileInputId}
                  >
                    Upload point-cloud mesh JSON
                  </label>
                  <button
                    className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-slate-100 transition hover:bg-white/10"
                    type="button"
                    onClick={() => {
                      void loadBundledExample(
                        bundledMediumExampleUrl,
                        bundledMediumSourceLabel,
                        "Unable to reload the bundled medium example.",
                      );
                    }}
                  >
                    Reload preview
                  </button>
                  <input
                    id={fileInputId}
                    className="sr-only"
                    type="file"
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

            <section className="min-w-0 flex-1 rounded-[1.1rem] border border-white/10 bg-slate-950/55 p-4 shadow-xl shadow-black/20">
              <div className="mb-3 flex items-center justify-between gap-3 px-1">
                <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
                  Interactive preview
                </p>
                <div
                  data-testid="pointcloud-fps"
                  className="rounded-xl border border-emerald-300/25 bg-emerald-400/10 px-4 py-2 text-sm font-semibold text-emerald-100"
                >
                  {fpsLabel(framesPerSecond)}
                </div>
              </div>
              <div
                ref={previewHostRef}
                className="relative h-[calc(100vh-10.5rem)]"
              >
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
                    onCameraStateChange={setCurrentCameraState}
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
                  <aside className="pointer-events-none absolute right-4 top-4 z-10 w-[21rem] rounded-[1rem] border border-white/10 bg-slate-950/88 p-4 shadow-2xl shadow-black/35 backdrop-blur-sm">
                    <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
                      Hit inspector
                    </p>
                    <div className="mt-3 space-y-3 text-sm text-slate-200">
                      <div className="flex items-center gap-3 rounded-[0.95rem] border border-white/10 bg-slate-900/75 p-3">
                        <span
                          aria-label="Hit color swatch"
                          className="h-10 w-10 rounded-full border border-white/15"
                          style={{
                            backgroundColor: colorToCss(selectedHit.color),
                          }}
                        />
                        <div>
                          <div className="font-semibold text-white">
                            Interpolated hit colour
                          </div>
                          <div className="font-mono text-xs text-slate-300">
                            {selectedHit.color
                              .map((value) => value.toFixed(4))
                              .join(", ")}
                          </div>
                        </div>
                      </div>
                      <div className="rounded-[0.95rem] border border-white/10 bg-slate-900/75 p-3">
                        <div className="font-semibold text-white">
                          Hit point
                        </div>
                        <div className="mt-1 font-mono text-xs text-slate-300">
                          {selectedHit.point
                            .map((value) => value.toFixed(4))
                            .join(", ")}
                        </div>
                      </div>
                      {selectedHit.neighbors.map((neighbor, index) => (
                        <div
                          key={`${neighbor.index}-${index}`}
                          className="rounded-[0.95rem] border border-white/10 bg-slate-900/75 p-3"
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div className="font-semibold text-white">
                              Neighbor {index + 1}
                            </div>
                            <div className="text-xs text-slate-300">
                              index {neighbor.index}
                            </div>
                          </div>
                          <div className="mt-2 font-mono text-xs leading-6 text-slate-300">
                            distance {neighbor.distance.toFixed(5)}
                            <br />
                            pos{" "}
                            {neighbor.position
                              .map((value) => value.toFixed(4))
                              .join(", ")}
                            <br />
                            rgb{" "}
                            {neighbor.color
                              .map((value) => value.toFixed(4))
                              .join(", ")}
                          </div>
                        </div>
                      ))}
                    </div>
                  </aside>
                ) : null}
              </div>
            </section>

            <div className="flex w-[25rem] shrink-0 flex-col gap-4">
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
                <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
                  View options
                </p>
                <div className="mt-4 space-y-3 text-sm text-slate-200">
                  <div className="grid gap-3 sm:grid-cols-2">
                    <label className="flex items-center gap-3 rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-4 py-4">
                      <input
                        className="accent-amber-300"
                        type="checkbox"
                        checked={viewSettings.showMesh}
                        onChange={(event) =>
                          updateViewSettings({ showMesh: event.target.checked })
                        }
                      />
                      Show mesh
                    </label>
                    <label
                      className={`flex items-center gap-3 rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-4 py-4 ${
                        viewSettings.showMesh
                          ? ""
                          : "cursor-not-allowed text-slate-500"
                      }`}
                    >
                      <input
                        className="accent-amber-300"
                        type="checkbox"
                        checked={viewSettings.showWireframe}
                        disabled={!viewSettings.showMesh}
                        onChange={(event) =>
                          updateViewSettings({
                            showWireframe: event.target.checked,
                          })
                        }
                      />
                      Wireframe mesh
                    </label>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <label className="flex items-center gap-3 rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-4 py-4">
                      <input
                        className="accent-amber-300"
                        type="checkbox"
                        checked={viewSettings.showPoints}
                        onChange={(event) =>
                          updateViewSettings({
                            showPoints: event.target.checked,
                          })
                        }
                      />
                      Show point cloud
                    </label>
                    <label className="rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-4 py-4">
                      <span
                        className={`mb-2 block ${
                          viewSettings.showPoints
                            ? "text-slate-300"
                            : "text-slate-500"
                        }`}
                      >
                        Point size: {viewSettings.pointSize.toFixed(3)}
                      </span>
                      <input
                        data-testid="point-size-slider"
                        className="w-full accent-amber-300 disabled:cursor-not-allowed disabled:opacity-50"
                        type="range"
                        min={0.01}
                        max={0.12}
                        step={0.005}
                        value={viewSettings.pointSize}
                        disabled={!viewSettings.showPoints}
                        onChange={(event) =>
                          updateViewSettings({
                            pointSize: Number(event.target.value),
                          })
                        }
                      />
                    </label>
                  </div>
                </div>
              </section>

              <section className="rounded-[1.1rem] border border-white/10 bg-slate-950/55 p-5 shadow-xl shadow-black/20">
                <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
                  Shading options
                </p>
                <div className="mt-4 space-y-4 text-sm text-slate-200">
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
              </section>

              <section className="rounded-[1.1rem] border border-white/10 bg-slate-950/55 p-5 shadow-xl shadow-black/20">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
                    Camera and orientation
                  </p>
                  <button
                    data-testid="screenshot-button"
                    className="rounded-xl border border-emerald-300/20 bg-emerald-400/10 px-4 py-2 text-sm font-semibold text-emerald-100 transition hover:bg-emerald-400/20"
                    type="button"
                    onClick={handleScreenshot}
                  >
                    Screenshot
                  </button>
                </div>

                <div className="mt-4 flex flex-wrap gap-3">
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

                <div className="mt-4 rounded-[0.95rem] border border-white/10 bg-slate-900/75 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-semibold text-white">
                      Saved viewpoints
                    </span>
                    <button
                      data-testid="save-viewpoint-button"
                      className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-slate-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
                      type="button"
                      disabled={currentCameraState === null}
                      onClick={handleSaveViewpoint}
                    >
                      Save this viewpoint
                    </button>
                  </div>
                  {savedViewpoints.length > 0 ? (
                    <div className="mt-3 flex flex-col gap-3">
                      {savedViewpoints.map((viewpoint) => (
                        <div
                          key={viewpoint.id}
                          className="rounded-[0.95rem] border border-white/10 bg-slate-950/40 p-3"
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0 flex-1">
                              <input
                                data-testid={`viewpoint-name-${viewpoint.id}`}
                                className="w-full rounded-[0.8rem] border border-white/10 bg-slate-900/75 px-3 py-2 text-sm font-semibold text-white outline-none transition focus:border-amber-300"
                                value={viewpoint.label}
                                onChange={(event) =>
                                  handleRenameViewpoint(
                                    viewpoint.id,
                                    event.target.value,
                                  )
                                }
                              />
                              <div className="mt-2 text-xs text-slate-400">
                                {viewpoint.swapYZ
                                  ? "Y/Z swapped"
                                  : "Default axes"}
                              </div>
                            </div>
                            <button
                              data-testid={`viewpoint-go-${viewpoint.id}`}
                              className="rounded-xl border border-emerald-300/30 bg-emerald-400/10 px-4 py-2 text-sm font-semibold text-emerald-100 transition hover:bg-emerald-400/20"
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
                              Go to viewpoint
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="mt-3 text-sm text-slate-400">
                      Save a viewpoint to keep a reusable camera stack for this
                      dataset.
                    </p>
                  )}
                </div>
              </section>
            </div>
          </div>
        </div>
      </main>

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
          Saved viewpoints persist in local storage for this browser, and the
          screenshot button downloads the current canvas view as a PNG.
        </p>
      </InfoModal>
    </>
  );
}
