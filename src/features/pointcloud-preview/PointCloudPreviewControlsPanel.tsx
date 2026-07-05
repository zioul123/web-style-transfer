import { useEffect, useRef, useState } from "react";
import {
  PointCloudPreviewPointSizeControl,
  PreviewVisibilityToggleButton,
} from "./PreviewViewControls";
import type { SavedViewpoint } from "./pointCloudPreviewModels";
import type {
  PointCloudPreviewBackgroundColor,
  PointCloudPreviewRenderMode,
  PointCloudPreviewViewAxis,
  PointCloudPreviewViewSettings,
  PreviewCameraState,
} from "./types";
import type { NextPointCloudPreviewCameraCommand } from "./usePointCloudPreviewController";
import {
  ArrowRightIcon,
  CollapsiblePanelCard,
  CopyIcon,
  IconButton,
  InfoIcon,
  SaveIcon,
  TrashIcon,
} from "./PointCloudPreviewUi";

const brightnessLabel = (brightness: number): string =>
  `${brightness.toFixed(2)}x`;

const cameraTupleText = (tuple: readonly [number, number, number]): string =>
  `(${tuple.map((value) => value.toFixed(1)).join(", ")})`;

const savedViewpointCameraText = (viewpoint: SavedViewpoint): string =>
  `position = ${cameraTupleText(viewpoint.camera.position)}\n` +
  `focal_point = ${cameraTupleText(viewpoint.camera.target)}`;

type KernelLevelControlOption = {
  readonly levelIndex: number;
  readonly groupCount: number;
};

function SavedViewpointInfoPopover({
  viewpoint,
}: {
  readonly viewpoint: SavedViewpoint;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [copyStatus, setCopyStatus] = useState<"idle" | "copied" | "failed">(
    "idle",
  );
  const containerRef = useRef<HTMLDivElement>(null);
  const popoverId = `viewpoint-info-popover-${viewpoint.id}`;
  const cameraText = savedViewpointCameraText(viewpoint);

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    const handleOutsidePointerDown = (event: PointerEvent): void => {
      if (
        event.target instanceof Node &&
        !containerRef.current?.contains(event.target)
      ) {
        setIsOpen(false);
        setCopyStatus("idle");
      }
    };

    document.addEventListener("pointerdown", handleOutsidePointerDown);
    return () => {
      document.removeEventListener("pointerdown", handleOutsidePointerDown);
    };
  }, [isOpen]);

  const copyCameraText = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(cameraText);
      setCopyStatus("copied");
    } catch {
      setCopyStatus("failed");
    }
  };

  return (
    <div ref={containerRef} className="relative shrink-0">
      <IconButton
        data-testid={`viewpoint-info-${viewpoint.id}`}
        className="h-9 w-9 rounded-full border border-sky-300/20 bg-sky-400/10 text-sky-100 hover:bg-sky-400/20"
        aria-label={`Show camera details for ${viewpoint.label}`}
        aria-expanded={isOpen}
        aria-controls={popoverId}
        onClick={() => {
          setIsOpen((open) => !open);
          setCopyStatus("idle");
        }}
      >
        <InfoIcon />
        <span className="sr-only">Show camera details</span>
      </IconButton>
      {isOpen ? (
        <div
          id={popoverId}
          data-testid={`viewpoint-info-popover-${viewpoint.id}`}
          className="absolute right-0 top-11 z-20 w-72 max-w-[calc(100vw-3rem)] rounded-[0.95rem] border border-sky-300/20 bg-slate-950 p-4 shadow-2xl shadow-black/40"
          role="dialog"
          aria-label={`Camera details for ${viewpoint.label}`}
        >
          <pre
            data-testid={`viewpoint-camera-details-${viewpoint.id}`}
            className="whitespace-pre-wrap font-mono text-xs leading-5 text-slate-200"
          >
            {cameraText}
          </pre>
          <button
            data-testid={`viewpoint-copy-${viewpoint.id}`}
            className="mt-3 inline-flex w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs font-semibold text-slate-100 transition hover:bg-white/10"
            type="button"
            onClick={() => void copyCameraText()}
          >
            <CopyIcon />
            <span>
              {copyStatus === "copied"
                ? "Copied"
                : copyStatus === "failed"
                  ? "Copy failed"
                  : "Copy to clipboard"}
            </span>
          </button>
        </div>
      ) : null}
    </div>
  );
}

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

type PointCloudPreviewControlsPanelProps = {
  readonly viewSettings: PointCloudPreviewViewSettings;
  readonly hasKernelPreview: boolean;
  readonly kernelLevelOptions: readonly KernelLevelControlOption[];
  readonly updateViewSettings: (
    patch: Partial<PointCloudPreviewViewSettings>,
  ) => void;
  readonly currentCameraState: PreviewCameraState | null;
  readonly issueCameraCommand: (
    command: NextPointCloudPreviewCameraCommand,
  ) => number;
  readonly savedViewpoints: readonly SavedViewpoint[];
  readonly onSaveViewpoint: () => void;
  readonly onRenameViewpoint: (id: number, label: string) => void;
  readonly onUpdateViewpoint: (id: number) => void;
  readonly onDeleteViewpoint: (id: number) => void;
};

export function PointCloudPreviewControlsPanel({
  viewSettings,
  hasKernelPreview,
  kernelLevelOptions,
  updateViewSettings,
  currentCameraState,
  issueCameraCommand,
  savedViewpoints,
  onSaveViewpoint,
  onRenameViewpoint,
  onUpdateViewpoint,
  onDeleteViewpoint,
}: PointCloudPreviewControlsPanelProps) {
  const selectedRenderModeClassName = "bg-amber-300 text-slate-950";
  const inactiveRenderModeClassName =
    "border border-white/10 bg-white/5 text-slate-100 hover:bg-white/10";
  const renderModeButtons: readonly {
    readonly mode: PointCloudPreviewRenderMode;
    readonly label: string;
    readonly testId: string;
  }[] = [
    { mode: "surface", label: "Surface", testId: "render-mode-surface-button" },
    { mode: "kernels", label: "Kernels", testId: "render-mode-kernels-button" },
  ];

  return (
    <div
      data-testid="pointcloud-right-panel"
      className="flex w-full flex-col gap-4 2xl:h-full 2xl:w-[25rem] 2xl:shrink-0 2xl:overflow-y-auto 2xl:pr-1"
    >
      <CollapsiblePanelCard title="View options" testId="view-options-card">
        <div className="space-y-3 text-sm text-slate-200">
          {hasKernelPreview ? (
            <div className="rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-4 py-4">
              <span className="mb-3 block text-slate-300">Render mode</span>
              <div className="grid gap-2 sm:grid-cols-2">
                {renderModeButtons.map((button) => {
                  const isSelected = viewSettings.renderMode === button.mode;
                  return (
                    <button
                      key={button.mode}
                      data-testid={button.testId}
                      className={`rounded-xl px-4 py-2 text-sm font-semibold transition ${
                        isSelected
                          ? selectedRenderModeClassName
                          : inactiveRenderModeClassName
                      }`}
                      type="button"
                      aria-pressed={isSelected}
                      onClick={() =>
                        updateViewSettings({
                          renderMode: button.mode,
                          showPoints:
                            button.mode === "kernels"
                              ? true
                              : viewSettings.showPoints,
                        })
                      }
                    >
                      {button.label}
                    </button>
                  );
                })}
              </div>
            </div>
          ) : null}
          {hasKernelPreview && viewSettings.renderMode === "kernels" ? (
            <label className="block rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-4 py-4">
              <span className="mb-2 block text-slate-300">Kernel level</span>
              <select
                data-testid="kernel-level-select"
                className="w-full rounded-xl border border-white/10 bg-slate-950 px-3 py-2 text-slate-100"
                value={viewSettings.kernelLevelIndex}
                onChange={(event) =>
                  updateViewSettings({
                    kernelLevelIndex: Number(event.target.value),
                  })
                }
              >
                {kernelLevelOptions.map((level) => (
                  <option key={level.levelIndex} value={level.levelIndex}>
                    Level {level.levelIndex} ({level.groupCount} anchors)
                  </option>
                ))}
              </select>
            </label>
          ) : null}
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
          <label className="flex items-center gap-3 rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-4 py-3">
            <input
              data-testid="ground-plane-axis-checkbox"
              className="accent-amber-300"
              type="checkbox"
              checked={viewSettings.showGroundPlane}
              onChange={(event) =>
                updateViewSettings({
                  showGroundPlane: event.target.checked,
                })
              }
            />
            Ground plane axis
          </label>
        </div>
      </CollapsiblePanelCard>

      <CollapsiblePanelCard
        title="Shading options"
        testId="shading-options-card"
      >
        <div className="space-y-4 text-sm text-slate-200">
          <label className="block rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-4 py-4">
            <span className="mb-2 block text-slate-300">Background colour</span>
            <select
              data-testid="background-color-select"
              className="w-full rounded-xl border border-white/10 bg-slate-950 px-3 py-2 text-slate-100"
              value={viewSettings.backgroundColor}
              onChange={(event) =>
                updateViewSettings({
                  backgroundColor: event.target
                    .value as PointCloudPreviewBackgroundColor,
                })
              }
            >
              <option value="default">Default</option>
              <option value="black">Black</option>
              <option value="white">White</option>
            </select>
          </label>
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

      <CollapsiblePanelCard title="Camera and orientation" testId="camera-card">
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
        <output data-testid="camera-state" className="sr-only" aria-live="off">
          {currentCameraState === null
            ? "unavailable"
            : JSON.stringify(currentCameraState)}
        </output>

        <div className="mt-4 rounded-[0.95rem] border border-white/10 bg-slate-900/75 p-4">
          <div className="flex items-center justify-between gap-3">
            <span className="font-semibold text-white">Saved viewpoints</span>
            <IconButton
              data-testid="save-viewpoint-button"
              className="h-10 w-10 border border-white/10 bg-white/5 text-slate-100 hover:bg-white/10"
              disabled={currentCameraState === null}
              aria-label="Save current viewpoint"
              title="Save current viewpoint"
              onClick={onSaveViewpoint}
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
                          onRenameViewpoint(viewpoint.id, event.target.value)
                        }
                      />
                      <SavedViewpointInfoPopover viewpoint={viewpoint} />
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
                        onClick={() => onUpdateViewpoint(viewpoint.id)}
                      >
                        <SaveIcon />
                        <span>Update</span>
                      </button>
                      <IconButton
                        data-testid={`viewpoint-delete-${viewpoint.id}`}
                        className="h-10 w-10 border border-rose-300/20 bg-rose-400/10 text-rose-100 hover:bg-rose-400/20"
                        aria-label={`Delete ${viewpoint.label}`}
                        title="Delete viewpoint"
                        onClick={() => onDeleteViewpoint(viewpoint.id)}
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
              Save a viewpoint to keep a reusable camera preset for any dataset
              you load here.
            </p>
          )}
        </div>
      </CollapsiblePanelCard>
    </div>
  );
}
