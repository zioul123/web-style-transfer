import { useState, type ComponentPropsWithoutRef, type ReactNode } from "react";
import { assetUrl } from "../../shared/assetUrls";

export function InfoTooltip({ text }: { readonly text: string }) {
  return (
    <span
      className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-white/15 bg-white/5 text-[11px] font-semibold text-slate-300"
      title={text}
    >
      i
    </span>
  );
}

export function LabelWithTooltip({
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

export function CollapsiblePanelCard({
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

export function IconButton({
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

export function SaveIcon() {
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

export function TrashIcon() {
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

export function ArrowRightIcon() {
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

export function RotateCcwIcon() {
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

export function XIcon() {
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

export function PointCloudPreviewHeader({
  onShowInfo,
}: {
  readonly onShowInfo: () => void;
}) {
  return (
    <header className="mb-4 flex shrink-0 flex-col gap-4 rounded-[1.1rem] border border-white/10 bg-slate-950/65 px-5 py-4 shadow-xl shadow-black/20 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:gap-6">
        <h1 className="text-2xl font-semibold tracking-tight text-white">
          Point-Cloud Mesh Preview
        </h1>
        <nav className="flex flex-wrap items-center gap-3 text-sm font-medium">
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
        onClick={onShowInfo}
      >
        Info
      </button>
    </header>
  );
}

export function PointCloudPreviewInfoModal({
  open,
  onClose,
}: {
  readonly open: boolean;
  readonly onClose: () => void;
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
          <p>
            This route renders a mesh with precomputed 3-nearest-neighbor
            surface colours, overlays the aligned point cloud, and shows the
            exact neighbour blend at mesh hit points without introducing
            differentiable rendering into the web app.
          </p>
          <p>
            Use the rendering algorithm controls to compare fragment-space KNN
            shading against baked vertex colours, then use the shading controls
            to inspect gamma and brightness differences between point and mesh
            displays.
          </p>
          <p>
            Saved viewpoints persist in local storage for this browser across
            all datasets on this route. Screenshot downloads the current canvas
            as a PNG, while batch screenshot cycles queued uploads through
            selected viewpoints and downloads one ZIP.
          </p>
        </div>
      </div>
    </div>
  );
}
