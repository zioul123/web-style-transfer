import {
  PointCloudPreviewPointSizeControl,
  PreviewVisibilityToggleButton,
} from "./PreviewViewControls";
import type { SavedViewpoint } from "./pointCloudPreviewModels";
import type {
  PointCloudPreviewViewAxis,
  PointCloudPreviewViewSettings,
  PreviewCameraState,
} from "./types";
import type { NextPointCloudPreviewCameraCommand } from "./usePointCloudPreviewController";
import {
  ArrowRightIcon,
  CollapsiblePanelCard,
  IconButton,
  SaveIcon,
  TrashIcon,
} from "./PointCloudPreviewUi";

const brightnessLabel = (brightness: number): string =>
  `${brightness.toFixed(2)}x`;

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
  updateViewSettings,
  currentCameraState,
  issueCameraCommand,
  savedViewpoints,
  onSaveViewpoint,
  onRenameViewpoint,
  onUpdateViewpoint,
  onDeleteViewpoint,
}: PointCloudPreviewControlsPanelProps) {
  return (
    <div
      data-testid="pointcloud-right-panel"
      className="flex w-full flex-col gap-4 2xl:h-full 2xl:w-[25rem] 2xl:shrink-0 2xl:overflow-y-auto 2xl:pr-1"
    >
      <CollapsiblePanelCard title="View options" testId="view-options-card">
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
