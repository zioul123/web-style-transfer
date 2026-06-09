import {
  startTransition,
  useEffect,
  useId,
  useMemo,
  useState,
  type ChangeEvent,
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

const colorToCss = (color: readonly [number, number, number]): string =>
  `rgb(${Math.round(color[0] * 255)} ${Math.round(color[1] * 255)} ${Math.round(
    color[2] * 255,
  )})`;

const boundsLabel = (values: readonly [number, number, number]): string =>
  values.map((value) => value.toFixed(3)).join(", ");

const fpsLabel = (framesPerSecond: number): string =>
  framesPerSecond > 0 ? `${framesPerSecond.toFixed(1)} FPS` : "Sampling FPS";

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

export function PointCloudPreviewApp() {
  const fileInputId = useId();
  const [assetState, setAssetState] = useState<LoadedAssetState>({
    status: "loading",
    sourceLabel: bundledMediumSourceLabel,
    errorMessage: null,
    data: null,
  });
  const [showMesh, setShowMesh] = useState<boolean>(true);
  const [showPoints, setShowPoints] = useState<boolean>(true);
  const [showWireframe, setShowWireframe] = useState<boolean>(false);
  const [meshColorMode, setMeshColorMode] =
    useState<MeshColorMode>("fragment-knn");
  const [showNeighborDebug, setShowNeighborDebug] = useState<boolean>(true);
  const [pointSize, setPointSize] = useState<number>(0.035);
  const [pointGammaCorrection, setPointGammaCorrection] =
    useState<boolean>(false);
  const [swapYZ, setSwapYZ] = useState<boolean>(false);
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

  const applyReadyAsset = (
    data: PointCloudMeshData,
    sourceLabel: string,
  ): void => {
    startTransition(() => {
      setAssetState({
        status: "ready",
        sourceLabel,
        errorMessage: null,
        data,
      });
      setSelectedHit(null);
      setSwapYZ(false);
      setSavedViewpoints([]);
      setNextViewpointId(1);
      setCurrentCameraState(null);
      setFramesPerSecond(0);
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
          startTransition(() => {
            setAssetState({
              status: "ready",
              sourceLabel: bundledMediumSourceLabel,
              errorMessage: null,
              data,
            });
            setSelectedHit(null);
            setSwapYZ(false);
            setSavedViewpoints([]);
            setNextViewpointId(1);
            setCurrentCameraState(null);
            setFramesPerSecond(0);
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

  const data = assetState.data;
  const canUseFragmentKnnShading = useMemo(
    () =>
      data !== null &&
      data.spatialHash.maxPointsPerCell <= maxFragmentShaderPointsPerCell,
    [data],
  );
  const effectiveMeshColorMode: MeshColorMode =
    meshColorMode === "fragment-knn" && canUseFragmentKnnShading
      ? "fragment-knn"
      : "baked";
  const meshColorModeDescription =
    effectiveMeshColorMode === "fragment-knn"
      ? `Spatial-hash fragment KNN shading active (${data?.pointCount ?? 0} points across ${data?.spatialHash.cellCount ?? 0} cells).`
      : meshColorMode === "fragment-knn" && data !== null
        ? `Spatial-hash fragment KNN shading found a dense cell with ${data.spatialHash.maxPointsPerCell} points, which exceeds the current per-cell shader cap of ${maxFragmentShaderPointsPerCell}; this dataset falls back to baked vertex colours.`
        : pointGammaCorrection
          ? "Baked vertex colours active with sRGB gamma decode applied for comparison."
          : "Baked vertex colours active.";

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
        swapYZ,
      },
      ...currentViews,
    ]);
    setNextViewpointId(nextId + 1);
  };

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(251,146,60,0.16),_transparent_28rem),radial-gradient(circle_at_bottom_right,_rgba(59,130,246,0.18),_transparent_32rem),linear-gradient(160deg,_#0b1120_0%,_#101826_48%,_#071019_100%)] text-slate-100">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-5 px-4 py-5 sm:px-6 sm:py-8">
        <section className="rounded-[2rem] border border-white/10 bg-slate-950/50 p-5 shadow-xl shadow-black/20">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <p className="text-sm font-semibold uppercase tracking-[0.24em] text-amber-300">
                Point-Cloud Mesh Preview
              </p>
              <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                Browser preview for mesh-aligned point clouds
              </h1>
              <p className="mt-3 text-sm leading-6 text-slate-300 sm:text-base">
                This route renders a mesh with precomputed 3-nearest-neighbor
                surface colours, overlays the aligned point cloud, and inspects
                the exact neighbour blend at mesh hit points without introducing
                differentiable rendering into the web app.
              </p>
            </div>
            <div className="flex flex-wrap gap-3 text-sm font-medium">
              <a
                className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-slate-100 transition hover:bg-white/10"
                href={assetUrl("")}
              >
                Main app
              </a>
              <a
                className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-slate-100 transition hover:bg-white/10"
                href={assetUrl("benchmark")}
              >
                Benchmark
              </a>
            </div>
          </div>
        </section>

        <div className="grid gap-5 xl:grid-cols-[minmax(18rem,0.72fr)_minmax(0,1.5fr)_minmax(20rem,0.92fr)]">
          <div className="flex flex-col gap-5">
            <section className="rounded-[2rem] border border-white/10 bg-slate-950/50 p-5 shadow-xl shadow-black/20">
              <div className="flex flex-col gap-4">
                <div>
                  <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
                    Data source
                  </p>
                  <p className="mt-2 text-sm text-slate-300">
                    Start from the bundled medium preview or upload a JSON
                    export from the Python mesh style-transfer pipeline.
                  </p>
                </div>
                <div className="flex flex-wrap gap-3">
                  <label
                    className="cursor-pointer rounded-full bg-amber-300 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-amber-200"
                    htmlFor={fileInputId}
                  >
                    Upload point-cloud mesh JSON
                  </label>
                  <button
                    className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-slate-100 transition hover:bg-white/10"
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
                <div className="rounded-2xl border border-white/10 bg-slate-900/70 p-4 text-sm text-slate-200">
                  <div className="flex items-center justify-between gap-3">
                    <span>Current source</span>
                    <span
                      data-testid="pointcloud-source-label"
                      className="font-semibold"
                    >
                      {assetState.sourceLabel}
                    </span>
                  </div>
                  <div className="mt-2 flex items-center justify-between gap-3">
                    <span>Status</span>
                    <span
                      data-testid="pointcloud-load-status"
                      className="font-semibold capitalize"
                    >
                      {assetState.status}
                    </span>
                  </div>
                  {assetState.errorMessage !== null ? (
                    <p className="mt-3 rounded-xl border border-rose-400/20 bg-rose-500/10 px-3 py-2 text-rose-200">
                      {assetState.errorMessage}
                    </p>
                  ) : null}
                </div>
              </div>
            </section>

            <section className="rounded-[2rem] border border-white/10 bg-slate-950/50 p-5 shadow-xl shadow-black/20">
              <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
                Dataset stats
              </p>
              {data !== null ? (
                <div className="mt-4 grid gap-3 text-sm text-slate-200 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
                  <div className="rounded-2xl border border-white/10 bg-slate-900/70 p-4">
                    <div className="text-slate-400">Mesh vertices</div>
                    <div
                      data-testid="mesh-vertex-count"
                      className="mt-1 text-2xl font-semibold text-white"
                    >
                      {data.meshVertexCount}
                    </div>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-slate-900/70 p-4">
                    <div className="text-slate-400">Mesh faces</div>
                    <div
                      data-testid="mesh-face-count"
                      className="mt-1 text-2xl font-semibold text-white"
                    >
                      {data.meshFaceCount}
                    </div>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-slate-900/70 p-4">
                    <div className="text-slate-400">Point samples</div>
                    <div
                      data-testid="point-sample-count"
                      className="mt-1 text-2xl font-semibold text-white"
                    >
                      {data.pointCount}
                    </div>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-slate-900/70 p-4">
                    <div className="text-slate-400">Bounds radius</div>
                    <div className="mt-1 text-2xl font-semibold text-white">
                      {data.bounds.radius.toFixed(3)}
                    </div>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-slate-900/70 p-4">
                    <div className="text-slate-400">Live FPS</div>
                    <div className="mt-1 text-2xl font-semibold text-white">
                      {framesPerSecond > 0 ? framesPerSecond.toFixed(1) : "--"}
                    </div>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-slate-900/70 p-4">
                    <div className="text-slate-400">Axis orientation</div>
                    <div className="mt-1 text-lg font-semibold text-white">
                      {swapYZ ? "Y/Z swapped" : "Default XYZ"}
                    </div>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-slate-900/70 p-4 sm:col-span-2 xl:col-span-1 2xl:col-span-2">
                    <div className="text-slate-400">Bounds min/max</div>
                    <div className="mt-1 font-mono text-xs leading-6 text-slate-200">
                      min [{boundsLabel(data.bounds.min)}]
                      <br />
                      max [{boundsLabel(data.bounds.max)}]
                    </div>
                  </div>
                </div>
              ) : (
                <p className="mt-4 text-sm text-slate-300">
                  Load a dataset to inspect mesh and point-cloud counts.
                </p>
              )}
            </section>
          </div>

          <section className="rounded-[2rem] border border-white/10 bg-slate-950/50 p-4 shadow-xl shadow-black/20">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-3 px-1">
              <div>
                <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
                  Interactive preview
                </p>
                <p className="mt-1 text-sm text-slate-300">
                  Orbit freely, snap to axes, and save viewpoints for repeatable
                  comparisons.
                </p>
              </div>
              <div
                data-testid="pointcloud-fps"
                className="rounded-full border border-emerald-300/25 bg-emerald-400/10 px-4 py-2 text-sm font-semibold text-emerald-100"
              >
                {fpsLabel(framesPerSecond)}
              </div>
            </div>
            {data !== null ? (
              <PointCloudPreviewScene
                data={data}
                showMesh={showMesh}
                showPoints={showPoints}
                showWireframe={showWireframe}
                meshColorMode={effectiveMeshColorMode}
                showNeighborDebug={showPoints && showNeighborDebug}
                pointSize={pointSize}
                pointGammaCorrection={pointGammaCorrection}
                swapYZ={swapYZ}
                selectedHit={selectedHit}
                cameraCommand={cameraCommand}
                onHoverSampleChange={setSelectedHit}
                onCameraStateChange={setCurrentCameraState}
                onFramesPerSecondChange={setFramesPerSecond}
              />
            ) : (
              <div className="flex h-[28rem] items-center justify-center rounded-[2rem] border border-dashed border-white/10 bg-slate-950/50 px-6 text-center text-sm text-slate-300 lg:h-[38rem]">
                {assetState.status === "loading"
                  ? `Loading ${assetState.sourceLabel}...`
                  : "No point-cloud mesh data is loaded yet."}
              </div>
            )}
          </section>

          <div className="flex flex-col gap-5">
            <section className="rounded-[2rem] border border-white/10 bg-slate-950/50 p-5 shadow-xl shadow-black/20">
              <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
                Viewer controls
              </p>
              <div className="mt-4 grid gap-4 text-sm text-slate-200">
                <div
                  data-testid="mesh-color-mode-status"
                  className="rounded-2xl border border-white/10 bg-slate-900/70 px-4 py-3 text-sm text-slate-300"
                >
                  {meshColorModeDescription}
                </div>

                <label>
                  <span className="mb-2 block text-slate-300">
                    Mesh colouring
                  </span>
                  <select
                    data-testid="mesh-color-mode-select"
                    className="w-full rounded-2xl border border-white/10 bg-slate-900/70 px-4 py-3 text-sm text-slate-100 outline-none transition focus:border-amber-300"
                    value={meshColorMode}
                    onChange={(event) =>
                      setMeshColorMode(event.target.value as MeshColorMode)
                    }
                  >
                    <option value="fragment-knn">Fragment KNN shading</option>
                    <option value="baked">Baked vertex colours</option>
                  </select>
                </label>

                <div className="grid gap-3 sm:grid-cols-2">
                  <label className="flex items-center gap-3 rounded-2xl border border-white/10 bg-slate-900/70 px-4 py-4">
                    <input
                      className="accent-amber-300"
                      type="checkbox"
                      checked={showMesh}
                      onChange={(event) => setShowMesh(event.target.checked)}
                    />
                    Show mesh
                  </label>
                  <label
                    className={`flex items-center gap-3 rounded-2xl border border-white/10 bg-slate-900/70 px-4 py-4 ${
                      showMesh ? "" : "cursor-not-allowed text-slate-500"
                    }`}
                  >
                    <input
                      className="accent-amber-300"
                      type="checkbox"
                      checked={showWireframe}
                      disabled={!showMesh}
                      onChange={(event) =>
                        setShowWireframe(event.target.checked)
                      }
                    />
                    Wireframe mesh
                  </label>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  <label className="flex items-center gap-3 rounded-2xl border border-white/10 bg-slate-900/70 px-4 py-4">
                    <input
                      className="accent-amber-300"
                      type="checkbox"
                      checked={showPoints}
                      onChange={(event) => setShowPoints(event.target.checked)}
                    />
                    Show point cloud
                  </label>
                  <label className="rounded-2xl border border-white/10 bg-slate-900/70 px-4 py-4">
                    <span
                      className={`mb-2 block ${
                        showPoints ? "text-slate-300" : "text-slate-500"
                      }`}
                    >
                      Point size: {pointSize.toFixed(3)}
                    </span>
                    <input
                      data-testid="point-size-slider"
                      className="w-full accent-amber-300 disabled:cursor-not-allowed disabled:opacity-50"
                      type="range"
                      min={0.01}
                      max={0.12}
                      step={0.005}
                      value={pointSize}
                      disabled={!showPoints}
                      onChange={(event) =>
                        setPointSize(Number(event.target.value))
                      }
                    />
                  </label>
                </div>

                <label className="flex items-center gap-3 rounded-2xl border border-white/10 bg-slate-900/70 px-4 py-3">
                  <input
                    className="accent-amber-300"
                    type="checkbox"
                    checked={pointGammaCorrection}
                    onChange={(event) =>
                      setPointGammaCorrection(event.target.checked)
                    }
                  />
                  Gamma-decode point cloud and baked mesh colours
                </label>

                {showPoints ? (
                  <label className="flex items-center gap-3 rounded-2xl border border-white/10 bg-slate-900/70 px-4 py-3">
                    <input
                      className="accent-amber-300"
                      type="checkbox"
                      checked={showNeighborDebug}
                      onChange={(event) =>
                        setShowNeighborDebug(event.target.checked)
                      }
                    />
                    Show 3-neighbour debug
                  </label>
                ) : null}
              </div>
            </section>

            <section className="rounded-[2rem] border border-white/10 bg-slate-950/50 p-5 shadow-xl shadow-black/20">
              <div className="flex flex-col gap-4">
                <div>
                  <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
                    Camera and orientation
                  </p>
                  <p className="mt-2 text-sm text-slate-300">
                    Swap the vertical/depth axes, snap to cardinal views, and
                    save repeatable camera presets.
                  </p>
                </div>

                <div className="flex flex-wrap gap-3">
                  <button
                    data-testid="swap-yz-button"
                    className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                      swapYZ
                        ? "bg-amber-300 text-slate-950 hover:bg-amber-200"
                        : "border border-white/10 bg-white/5 text-slate-100 hover:bg-white/10"
                    }`}
                    type="button"
                    onClick={() => setSwapYZ((currentValue) => !currentValue)}
                  >
                    {swapYZ ? "Y/Z swapped" : "Flip Y and Z"}
                  </button>
                  <button
                    className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-slate-100 transition hover:bg-white/10"
                    type="button"
                    onClick={() => issueCameraCommand({ type: "frame" })}
                  >
                    Reset camera
                  </button>
                </div>

                <div className="grid grid-cols-3 gap-2">
                  {snapAxisButtons.map((button) => (
                    <button
                      key={button.axis}
                      className="rounded-2xl border border-white/10 bg-slate-900/70 px-3 py-3 text-sm font-semibold text-slate-100 transition hover:bg-white/10"
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

                <div className="rounded-2xl border border-white/10 bg-slate-900/70 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-semibold text-white">
                      Saved viewpoints
                    </span>
                    <button
                      data-testid="save-viewpoint-button"
                      className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-slate-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
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
                          className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-white/10 bg-slate-950/40 px-4 py-3"
                        >
                          <div>
                            <div className="font-semibold text-white">
                              {viewpoint.label}
                            </div>
                            <div className="text-xs text-slate-400">
                              {viewpoint.swapYZ
                                ? "Y/Z swapped"
                                : "Default axes"}
                            </div>
                          </div>
                          <button
                            data-testid={`viewpoint-go-${viewpoint.id}`}
                            className="rounded-full border border-emerald-300/30 bg-emerald-400/10 px-4 py-2 text-sm font-semibold text-emerald-100 transition hover:bg-emerald-400/20"
                            type="button"
                            onClick={() => {
                              setSwapYZ(viewpoint.swapYZ);
                              issueCameraCommand({
                                type: "restore",
                                camera: viewpoint.camera,
                              });
                            }}
                          >
                            Go to viewpoint
                          </button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="mt-3 text-sm text-slate-400">
                      Save a viewpoint after orbiting or snapping the camera to
                      build a quick comparison stack.
                    </p>
                  )}
                </div>
              </div>
            </section>

            <section className="rounded-[2rem] border border-white/10 bg-slate-950/50 p-5 shadow-xl shadow-black/20">
              <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
                Hit inspector
              </p>
              {selectedHit !== null ? (
                <div className="mt-4 space-y-3 text-sm text-slate-200">
                  <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-slate-900/70 p-4">
                    <span
                      aria-label="Hit color swatch"
                      className="h-10 w-10 rounded-full border border-white/15"
                      style={{ backgroundColor: colorToCss(selectedHit.color) }}
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
                  <div className="rounded-2xl border border-white/10 bg-slate-900/70 p-4">
                    <div className="font-semibold text-white">Hit point</div>
                    <div className="mt-1 font-mono text-xs text-slate-300">
                      {selectedHit.point
                        .map((value) => value.toFixed(4))
                        .join(", ")}
                    </div>
                  </div>
                  {selectedHit.neighbors.map((neighbor, index) => (
                    <div
                      key={`${neighbor.index}-${index}`}
                      className="rounded-2xl border border-white/10 bg-slate-900/70 p-4"
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
              ) : (
                <p className="mt-4 text-sm text-slate-300">
                  Hover the mesh surface to inspect the exact
                  3-nearest-neighbour blend at that hit point.
                </p>
              )}
            </section>
          </div>
        </div>
      </div>
    </main>
  );
}
