import { useRef, useState } from "react";
import type { Dispatch, RefObject, SetStateAction } from "react";
import type {
  ResolutionPreset,
  StyleTransferControls,
  StyleTransferImages,
  UseStyleTransferControllerResult,
} from "../types/controller";
import { isResolutionPreset, resolutionOptions } from "../uiOptions";
import {
  ThreeImagePreview,
  type PreviewImage,
  type PreviewMode,
} from "./ThreeImagePreview";

type ImageCard =
  | {
      kind: "source";
      label: "Content" | "Style";
      src: string | null;
      inputRef: RefObject<HTMLInputElement | null>;
      target: "content" | "style";
    }
  | {
      kind: "output";
      label: "Output";
      src: string | null;
    };

type PreviewPanelProps = {
  readonly controls: StyleTransferControls;
  readonly images: StyleTransferImages;
  readonly onUpload: UseStyleTransferControllerResult["onUpload"];
};

type DoubleBufferedImageProps = {
  readonly src: string;
  readonly alt: string;
};

function DoubleBufferedImage({ src, alt }: DoubleBufferedImageProps) {
  const [visibleSrc, setVisibleSrc] = useState<string>(src);
  const loadingSrc = src === visibleSrc ? null : src;

  return (
    <>
      <img
        className="h-full w-full object-contain"
        src={visibleSrc}
        alt={alt}
      />
      {loadingSrc === null ? null : (
        <img
          className="absolute inset-0 h-full w-full object-contain opacity-0"
          src={loadingSrc}
          alt=""
          aria-hidden="true"
          onLoad={() => setVisibleSrc(loadingSrc)}
        />
      )}
    </>
  );
}

const parseResolutionPreset = (
  preset: ResolutionPreset,
): { readonly width: number; readonly height: number } => {
  const [widthText, heightText] = preset.split("x");
  return { width: Number(widthText), height: Number(heightText) };
};

const previewTabs: readonly {
  readonly mode: PreviewMode;
  readonly label: string;
}[] = [
  { mode: "classic", label: "Cards" },
  { mode: "r3f-row", label: "R3F scaled row" },
  { mode: "r3f-overlay", label: "R3F overlay" },
];

type PreviewScaleControlProps = {
  readonly label: string;
  readonly value: number;
  readonly setValue: Dispatch<SetStateAction<number>>;
};

function PreviewScaleControl({
  label,
  value,
  setValue,
}: PreviewScaleControlProps) {
  return (
    <label className="flex flex-col gap-1">
      {label}: {value.toFixed(2)}×
      <input
        className="accent-sky-300"
        type="range"
        min={0.5}
        max={2}
        step={0.05}
        value={value}
        onChange={(event) => setValue(Number(event.target.value))}
      />
    </label>
  );
}

export function PreviewPanel({
  controls,
  images,
  onUpload,
}: PreviewPanelProps) {
  const [previewMode, setPreviewMode] = useState<PreviewMode>("classic");
  const [previewZoom, setPreviewZoom] = useState<number>(1);
  const [contentPreviewScale, setContentPreviewScale] = useState<number>(1);
  const [stylePreviewScale, setStylePreviewScale] = useState<number>(1);
  const [outputPreviewScale, setOutputPreviewScale] = useState<number>(1);
  const contentFileInputRef = useRef<HTMLInputElement | null>(null);
  const styleFileInputRef = useRef<HTMLInputElement | null>(null);
  const imageCards: readonly ImageCard[] = [
    {
      kind: "source",
      label: "Content",
      src: images.contentImage,
      inputRef: contentFileInputRef,
      target: "content",
    },
    {
      kind: "source",
      label: "Style",
      src: images.styleImage,
      inputRef: styleFileInputRef,
      target: "style",
    },
    { kind: "output", label: "Output", src: images.outputImage },
  ];
  const contentPreviewResolution = parseResolutionPreset(
    controls.contentResolution,
  );
  const stylePreviewResolution = parseResolutionPreset(
    controls.styleResolution,
  );
  const previewImages: readonly PreviewImage[] = [
    {
      key: "content",
      label: "Content",
      src: images.contentImage,
      resolution: contentPreviewResolution,
      scale: contentPreviewScale,
      tint: "#38bdf8",
    },
    {
      key: "style",
      label: "Style",
      src: images.styleImage,
      resolution: stylePreviewResolution,
      scale: stylePreviewScale,
      tint: "#f0abfc",
    },
    {
      key: "output",
      label: "Output",
      src: images.outputImage,
      resolution: contentPreviewResolution,
      scale: outputPreviewScale,
      tint: "#34d399",
    },
  ];

  return (
    <section className="rounded-lg border border-white/10 bg-slate-950/55 p-4 shadow-xl shadow-black/20">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="text-sm font-semibold uppercase text-sky-300">
            Image preview experiments
          </p>
          <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-300">
            Toggle between the original cards and React Three Fiber views. The
            R3F views size planes from the selected content/style resolutions so
            changing presets shows the accurate relative canvas scale.
          </p>
        </div>
        <div
          className="flex flex-wrap gap-2"
          role="tablist"
          aria-label="Preview mode"
        >
          {previewTabs.map((tab) => (
            <button
              key={tab.mode}
              className={`rounded-lg px-3 py-2 text-sm font-semibold transition ${
                previewMode === tab.mode
                  ? "bg-sky-300 text-sky-950"
                  : "bg-slate-900 text-slate-200 hover:bg-slate-800"
              }`}
              type="button"
              role="tab"
              aria-selected={previewMode === tab.mode}
              onClick={() => setPreviewMode(tab.mode)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-4 grid gap-3 rounded-lg border border-white/10 bg-slate-900/65 p-3 text-sm lg:grid-cols-6">
        <label className="flex flex-col gap-1">
          Content resolution
          <select
            className="rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-slate-100"
            value={controls.contentResolution}
            onChange={(event) => {
              const nextResolution = event.target.value;
              if (isResolutionPreset(nextResolution)) {
                controls.setContentResolution(nextResolution);
              }
            }}
          >
            {resolutionOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          Style resolution
          <select
            className="rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm text-slate-100"
            value={controls.styleResolution}
            onChange={(event) => {
              const nextResolution = event.target.value;
              if (isResolutionPreset(nextResolution)) {
                controls.setStyleResolution(nextResolution);
              }
            }}
          >
            {resolutionOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          Preview zoom: {previewZoom.toFixed(2)}×
          <input
            className="accent-sky-300"
            type="range"
            min={0.5}
            max={4}
            step={0.05}
            value={previewZoom}
            onChange={(event) => setPreviewZoom(Number(event.target.value))}
          />
        </label>
        <PreviewScaleControl
          label="Content visual scale"
          value={contentPreviewScale}
          setValue={setContentPreviewScale}
        />
        <PreviewScaleControl
          label="Style visual scale"
          value={stylePreviewScale}
          setValue={setStylePreviewScale}
        />
        <PreviewScaleControl
          label="Output visual scale"
          value={outputPreviewScale}
          setValue={setOutputPreviewScale}
        />
        <div className="flex items-end lg:col-span-6">
          <button
            className="rounded-lg bg-slate-200 px-3 py-2 font-semibold text-slate-950 transition hover:bg-white"
            type="button"
            onClick={() => {
              setPreviewZoom(1);
              setContentPreviewScale(1);
              setStylePreviewScale(1);
              setOutputPreviewScale(1);
            }}
          >
            Reset preview controls
          </button>
        </div>
      </div>

      <div className="mt-4">
        {previewMode === "classic" ? (
          <div className="grid gap-4 md:grid-cols-3">
            {imageCards.map((image) => (
              <figure
                className="overflow-hidden rounded-lg border border-white/10 bg-slate-950/65 shadow-xl shadow-black/20"
                key={image.label}
              >
                <div className="relative flex aspect-[4/5] items-center justify-center bg-slate-900/80">
                  {image.src === null ? (
                    <p className="px-4 text-center text-sm text-slate-500">
                      Waiting for {image.label.toLowerCase()} image
                    </p>
                  ) : (
                    <DoubleBufferedImage src={image.src} alt={image.label} />
                  )}
                </div>
                <figcaption className="flex items-center justify-between border-t border-white/10 px-4 py-3 text-sm">
                  <span className="font-semibold text-slate-100">
                    {image.label}
                  </span>
                  {image.kind === "source" ? (
                    <>
                      <input
                        className="hidden"
                        ref={image.inputRef}
                        type="file"
                        accept="image/*"
                        onChange={(event) => void onUpload(event, image.target)}
                      />
                      <button
                        className="rounded-lg bg-sky-300 px-3 py-1.5 font-semibold text-sky-950 transition hover:bg-sky-200"
                        type="button"
                        onClick={() => image.inputRef.current?.click()}
                      >
                        Select
                      </button>
                    </>
                  ) : (
                    <span className="text-slate-400">Optimized</span>
                  )}
                </figcaption>
              </figure>
            ))}
          </div>
        ) : (
          <ThreeImagePreview
            mode={previewMode}
            images={previewImages}
            zoom={previewZoom}
          />
        )}
      </div>

      <p className="mt-3 rounded-lg border border-amber-300/25 bg-amber-300/10 px-3 py-2 text-sm leading-6 text-amber-100">
        First-pass flicker note: the DOM card view now preloads the next data
        URL behind the visible image, and the R3F texture hook keeps the
        previous texture until the new optimized frame finishes loading. Please
        still watch this during long optimizer runs, especially on mobile GPUs,
        because repeated PNG data URLs can still pressure texture memory.
      </p>
    </section>
  );
}
