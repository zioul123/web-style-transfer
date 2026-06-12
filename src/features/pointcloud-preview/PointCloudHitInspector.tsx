import type { PointCloudHitSample } from "./types";

type PointCloudHitInspectorProps = {
  readonly hit: PointCloudHitSample;
  readonly showNeighborDetails?: boolean;
  readonly testId?: string;
  readonly className?: string;
};

const defaultClassName =
  "pointer-events-none absolute right-4 top-4 z-10 w-[21rem] rounded-[1rem] border border-white/10 bg-slate-950/88 p-4 shadow-2xl shadow-black/35 backdrop-blur-sm";

const colorToCss = (color: readonly [number, number, number]): string =>
  `rgb(${Math.round(color[0] * 255)} ${Math.round(color[1] * 255)} ${Math.round(
    color[2] * 255,
  )})`;

const tupleLabel = (
  values: readonly [number, number, number],
  fractionDigits: number,
): string => values.map((value) => value.toFixed(fractionDigits)).join(", ");

export function PointCloudHitInspector({
  hit,
  showNeighborDetails = true,
  testId = "pointcloud-hit-inspector",
  className = defaultClassName,
}: PointCloudHitInspectorProps) {
  return (
    <aside data-testid={testId} className={className}>
      <p className="text-sm font-semibold uppercase tracking-[0.2em] text-sky-300">
        Hit inspector
      </p>
      <div className="mt-3 space-y-3 text-sm text-slate-200">
        <div
          data-testid={`${testId}-color`}
          className="flex items-center gap-3 rounded-[0.95rem] border border-white/10 bg-slate-900/75 p-3"
        >
          <span
            aria-label="Hit color swatch"
            className="h-10 w-10 rounded-full border border-white/15"
            style={{
              backgroundColor: colorToCss(hit.color),
            }}
          />
          <div>
            <div className="font-semibold text-white">
              Interpolated hit colour
            </div>
            <div className="font-mono text-xs text-slate-300">
              {tupleLabel(hit.color, 4)}
            </div>
          </div>
        </div>
        <div
          data-testid={`${testId}-point`}
          className="rounded-[0.95rem] border border-white/10 bg-slate-900/75 p-3"
        >
          <div className="font-semibold text-white">Hit point</div>
          <div className="mt-1 font-mono text-xs text-slate-300">
            {tupleLabel(hit.point, 4)}
          </div>
        </div>
        {showNeighborDetails
          ? hit.neighbors.map((neighbor, index) => (
              <div
                key={`${neighbor.index}-${index}`}
                data-testid={`${testId}-neighbor-detail`}
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
                  pos {tupleLabel(neighbor.position, 4)}
                  <br />
                  rgb {tupleLabel(neighbor.color, 4)}
                </div>
              </div>
            ))
          : null}
      </div>
    </aside>
  );
}
