import { useId, type ChangeEvent } from "react";
import type {
  LoadedAssetState,
  UploadedPointCloudFile,
} from "./pointCloudPreviewModels";
import { uploadedFileStatusLabel } from "./pointCloudPreviewModels";
import type { MeshColorMode, PointCloudMeshData } from "./types";
import {
  InfoTooltip,
  LabelWithTooltip,
  RotateCcwIcon,
  XIcon,
} from "./PointCloudPreviewUi";

const boundsLabel = (values: readonly [number, number, number]): string =>
  values.map((value) => value.toFixed(3)).join(", ");

export function RenderingAlgorithmPanel({
  meshColorMode,
  description,
  onMeshColorModeChange,
}: {
  readonly meshColorMode: MeshColorMode;
  readonly description: string;
  readonly onMeshColorModeChange: (meshColorMode: MeshColorMode) => void;
}) {
  return (
    <section className="rounded-[1.1rem] border border-white/10 bg-slate-950/55 p-5 shadow-xl shadow-black/20">
      <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
        Rendering algorithm
      </p>
      <div className="mt-4">
        <select
          data-testid="mesh-color-mode-select"
          className="w-full rounded-[0.95rem] border border-white/10 bg-slate-900/75 px-4 py-3 text-sm text-slate-100 outline-none transition focus:border-amber-300"
          value={meshColorMode}
          onChange={(event) =>
            onMeshColorModeChange(event.target.value as MeshColorMode)
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
        {description}
      </div>
    </section>
  );
}

export function DataSourcePanel({
  assetState,
  uploadedFiles,
  activeUploadedFileId,
  onFileUpload,
  onReload,
  onSelectFile,
  onRemoveFile,
}: {
  readonly assetState: LoadedAssetState;
  readonly uploadedFiles: readonly UploadedPointCloudFile[];
  readonly activeUploadedFileId: number | null;
  readonly onFileUpload: (
    event: ChangeEvent<HTMLInputElement>,
  ) => Promise<void>;
  readonly onReload: () => void;
  readonly onSelectFile: (file: UploadedPointCloudFile) => void;
  readonly onRemoveFile: (id: number) => void;
}) {
  const fileInputId = useId();

  return (
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
          onClick={onReload}
        >
          <RotateCcwIcon />
        </button>
        <input
          id={fileInputId}
          className="sr-only"
          type="file"
          multiple
          accept=".json,application/json"
          data-testid="pointcloud-upload-input"
          onChange={(event) => {
            void onFileUpload(event);
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
                  onClick={() => onSelectFile(uploadedFile)}
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
                  onClick={() => onRemoveFile(uploadedFile.id)}
                >
                  <XIcon />
                </button>
              </div>
            );
          })
        ) : (
          <p className="px-3 py-2 text-sm text-slate-400">No uploads queued.</p>
        )}
      </div>
    </section>
  );
}

export function DatasetStatsPanel({
  data,
}: {
  readonly data: PointCloudMeshData | null;
}) {
  return (
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
  );
}
