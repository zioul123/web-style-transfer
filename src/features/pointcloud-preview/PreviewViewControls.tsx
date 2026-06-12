import type { PointCloudPreviewViewSettings } from "./types";

type PreviewControlVariant = "regular" | "compact";

type PreviewVisibilityToggleButtonProps = {
  readonly label: string;
  readonly visible: boolean;
  readonly disabled?: boolean;
  readonly testId: string;
  readonly variant?: PreviewControlVariant;
  readonly onClick: () => void;
};

type PointCloudPreviewPointSizeControlProps = {
  readonly viewSettings: Pick<
    PointCloudPreviewViewSettings,
    "showPoints" | "pointSize"
  >;
  readonly testId: string;
  readonly variant?: PreviewControlVariant;
  readonly onPointSizeChange: (pointSize: number) => void;
};

const visibilityToggleVariantClassNames: Record<PreviewControlVariant, string> =
  {
    regular: "rounded-[0.95rem] px-4 py-4",
    compact: "min-h-14 rounded-lg px-3 py-3",
  };

const pointSizeControlVariantClassNames: Record<PreviewControlVariant, string> =
  {
    regular: "rounded-[0.95rem] px-4 py-4",
    compact: "mt-2 block rounded-lg px-3 py-3",
  };

function EyeOpenIcon() {
  return (
    <svg
      aria-hidden="true"
      className="h-4 w-4"
      viewBox="0 0 16 16"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M1.75 8C2.9 5.82 5.2 4.25 8 4.25C10.8 4.25 13.1 5.82 14.25 8C13.1 10.18 10.8 11.75 8 11.75C5.2 11.75 2.9 10.18 1.75 8Z"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinejoin="round"
      />
      <circle cx="8" cy="8" r="1.75" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}

function EyeClosedIcon() {
  return (
    <svg
      aria-hidden="true"
      className="h-4 w-4"
      viewBox="0 0 16 16"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M2.25 2.25L13.75 13.75"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
      />
      <path
        d="M6.55 4.46C7.01 4.32 7.49 4.25 8 4.25C10.8 4.25 13.1 5.82 14.25 8C13.75 8.95 13.03 9.79 12.14 10.45M9.52 9.54C9.13 9.95 8.58 10.2 8 10.2C6.83 10.2 5.88 9.25 5.88 8.08C5.88 7.5 6.12 6.97 6.51 6.58M3.86 5.56C2.98 6.22 2.26 7.06 1.75 8C2.9 10.18 5.2 11.75 8 11.75C8.5 11.75 8.99 11.68 9.45 11.54"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function PreviewVisibilityToggleButton({
  label,
  visible,
  disabled = false,
  testId,
  variant = "regular",
  onClick,
}: PreviewVisibilityToggleButtonProps) {
  return (
    <button
      data-testid={testId}
      className={`flex items-center justify-between gap-3 border text-sm font-semibold transition ${
        visibilityToggleVariantClassNames[variant]
      } ${
        disabled
          ? "cursor-not-allowed border-white/10 bg-slate-900/55 text-slate-500"
          : "border-white/10 bg-slate-900/75 text-slate-100 hover:bg-white/10"
      }`}
      type="button"
      aria-pressed={visible}
      aria-label={`${visible ? "Hide" : "Show"} ${label}`}
      disabled={disabled}
      onClick={onClick}
    >
      <span>{label}</span>
      {visible ? <EyeOpenIcon /> : <EyeClosedIcon />}
    </button>
  );
}

export function PointCloudPreviewPointSizeControl({
  viewSettings,
  testId,
  variant = "regular",
  onPointSizeChange,
}: PointCloudPreviewPointSizeControlProps) {
  const labelClassName =
    variant === "compact"
      ? `mb-2 block text-xs uppercase ${
          viewSettings.showPoints ? "text-slate-400" : "text-slate-500"
        }`
      : `mb-2 block ${
          viewSettings.showPoints ? "text-slate-300" : "text-slate-500"
        }`;

  return (
    <label
      className={`border border-white/10 bg-slate-900/75 ${
        pointSizeControlVariantClassNames[variant]
      }`}
    >
      <span className={labelClassName}>
        Point size: {viewSettings.pointSize.toFixed(3)}
      </span>
      <input
        data-testid={testId}
        className="w-full accent-amber-300 disabled:cursor-not-allowed disabled:opacity-50"
        type="range"
        min={0.01}
        max={0.12}
        step={0.005}
        value={viewSettings.pointSize}
        disabled={!viewSettings.showPoints}
        onChange={(event) => onPointSizeChange(Number(event.target.value))}
      />
    </label>
  );
}
