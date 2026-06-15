import type { Dispatch, SetStateAction } from "react";
import type {
  BatchScreenshotProgress,
  SavedViewpoint,
} from "./pointCloudPreviewModels";

type BatchScreenshotModalProps = {
  readonly open: boolean;
  readonly uploadedFileCount: number;
  readonly savedViewpoints: readonly SavedViewpoint[];
  readonly selectedViewpointIds: readonly number[];
  readonly setSelectedViewpointIds: Dispatch<SetStateAction<readonly number[]>>;
  readonly progress: BatchScreenshotProgress | null;
  readonly error: string | null;
  readonly isRunning: boolean;
  readonly onClose: () => void;
  readonly onDownload: () => Promise<void>;
};

export function BatchScreenshotModal({
  open,
  uploadedFileCount,
  savedViewpoints,
  selectedViewpointIds,
  setSelectedViewpointIds,
  progress,
  error,
  isRunning,
  onClose,
  onDownload,
}: BatchScreenshotModalProps) {
  if (!open) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/75 px-4"
      onClick={onClose}
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
              Capture {uploadedFileCount} uploaded meshes at each selected
              viewpoint and download one ZIP archive.
            </p>
          </div>
          <button
            className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm font-semibold text-slate-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
            type="button"
            disabled={isRunning}
            onClick={onClose}
          >
            Close
          </button>
        </div>

        {savedViewpoints.length > 0 ? (
          <>
            <div className="mt-5 flex items-center justify-between gap-3">
              <span className="text-sm font-semibold text-slate-200">
                {selectedViewpointIds.length} of {savedViewpoints.length}{" "}
                selected
              </span>
              <button
                data-testid="batch-screenshot-select-all"
                className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm font-semibold text-slate-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
                type="button"
                disabled={isRunning}
                onClick={() =>
                  setSelectedViewpointIds(
                    selectedViewpointIds.length === savedViewpoints.length
                      ? []
                      : savedViewpoints.map((viewpoint) => viewpoint.id),
                  )
                }
              >
                {selectedViewpointIds.length === savedViewpoints.length
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
                    disabled={isRunning}
                    checked={selectedViewpointIds.includes(viewpoint.id)}
                    onChange={(event) =>
                      setSelectedViewpointIds((currentIds) =>
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

        {progress !== null ? (
          <div
            data-testid="batch-screenshot-progress"
            className="mt-4 rounded-[0.95rem] border border-sky-300/20 bg-sky-400/10 px-4 py-3 text-sm text-sky-100"
          >
            <div className="flex justify-between gap-3 font-semibold">
              <span>{progress.label}</span>
              <span>
                {progress.completed}/{progress.total}
              </span>
            </div>
          </div>
        ) : null}
        {error !== null ? (
          <p
            data-testid="batch-screenshot-error"
            className="mt-4 rounded-[0.95rem] border border-rose-300/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-100"
          >
            {error}
          </p>
        ) : null}

        <button
          data-testid="batch-screenshot-download"
          className="mt-5 w-full rounded-xl bg-amber-300 px-4 py-3 text-sm font-semibold text-slate-950 transition hover:bg-amber-200 disabled:cursor-not-allowed disabled:opacity-50"
          type="button"
          disabled={
            isRunning ||
            savedViewpoints.length === 0 ||
            selectedViewpointIds.length === 0
          }
          onClick={() => {
            void onDownload();
          }}
        >
          {isRunning
            ? "Capturing screenshots..."
            : `Download ${uploadedFileCount * selectedViewpointIds.length} screenshots as ZIP`}
        </button>
      </div>
    </div>
  );
}
