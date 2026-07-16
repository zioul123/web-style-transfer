import { expect, test, type Locator, type Page } from "@playwright/test";
import type { PointCloudMeshBaseJson } from "../src/features/pointcloud-preview/types";
import { gotoStableApp } from "../tests/helpers/appPage";

type MeshColorMode = "baked" | "fragment-knn";

type FrameStats = {
  readonly sampleCount: number;
  readonly medianMs: number;
  readonly p95Ms: number;
  readonly minMs: number;
  readonly maxMs: number;
};

type ModeMeasurement = {
  readonly mode: MeshColorMode;
  readonly stats: FrameStats;
  readonly trials: readonly FrameStats[];
};

type BenchmarkWorkload = {
  readonly name: "regular-grid" | "dense-within-cap";
  readonly fileName: string;
  readonly pointCount: number;
  readonly expectedMaxPointsPerCell: number;
  readonly json: PointCloudMeshBaseJson;
};

const benchmarkViewport = { width: 1600, height: 900 } as const;
const warmupFrameCount = 12;
const trialCount = 3;
const framesPerTrial = 12;

const cubeVertices = [
  [-1, -1, -1],
  [1, -1, -1],
  [1, 1, -1],
  [-1, 1, -1],
  [-1, -1, 1],
  [1, -1, 1],
  [1, 1, 1],
  [-1, 1, 1],
] as const;

const cubeFaces = [
  [0, 2, 1],
  [0, 3, 2],
  [4, 5, 6],
  [4, 6, 7],
  [0, 1, 5],
  [0, 5, 4],
  [1, 2, 6],
  [1, 6, 5],
  [2, 3, 7],
  [2, 7, 6],
  [3, 0, 4],
  [3, 4, 7],
] as const;

const buildRegularGridWorkload = (): BenchmarkWorkload => {
  const pointsPerAxis = 12;
  const pointPositions: [number, number, number][] = [];
  const pointColors: [number, number, number][] = [];

  for (let z = 0; z < pointsPerAxis; z += 1) {
    for (let y = 0; y < pointsPerAxis; y += 1) {
      for (let x = 0; x < pointsPerAxis; x += 1) {
        pointPositions.push([
          -0.95 + (1.9 * x) / (pointsPerAxis - 1),
          -0.95 + (1.9 * y) / (pointsPerAxis - 1),
          -0.95 + (1.9 * z) / (pointsPerAxis - 1),
        ]);
        pointColors.push([
          x / (pointsPerAxis - 1),
          y / (pointsPerAxis - 1),
          z / (pointsPerAxis - 1),
        ]);
      }
    }
  }

  return {
    name: "regular-grid",
    fileName: "fragment-knn-regular-grid.json",
    pointCount: pointPositions.length,
    expectedMaxPointsPerCell: 8,
    json: {
      pc_xyz: pointPositions,
      pc_rgb: pointColors,
      m_verts: cubeVertices,
      m_faces: cubeFaces,
    },
  };
};

const buildDenseWithinCapWorkload = (): BenchmarkWorkload => {
  const pointCount = 128;
  const pointPositions = Array.from(
    { length: pointCount },
    (): [number, number, number] => [0, 0, 0],
  );
  const pointColors = Array.from(
    { length: pointCount },
    (_, index): [number, number, number] => [
      (index % 11) / 10,
      (index % 17) / 16,
      (index % 23) / 22,
    ],
  );

  return {
    name: "dense-within-cap",
    fileName: "fragment-knn-dense-within-cap.json",
    pointCount,
    expectedMaxPointsPerCell: pointCount,
    json: {
      pc_xyz: pointPositions,
      pc_rgb: pointColors,
      m_verts: cubeVertices,
      m_faces: cubeFaces,
    },
  };
};

const percentile = (sortedValues: readonly number[], ratio: number): number =>
  sortedValues[
    Math.min(
      sortedValues.length - 1,
      Math.ceil(sortedValues.length * ratio) - 1,
    )
  ] ?? Number.NaN;

const summarizeFrames = (samples: readonly number[]): FrameStats => {
  const sortedSamples = [...samples].sort((left, right) => left - right);
  const middleIndex = Math.floor(sortedSamples.length / 2);
  const median =
    sortedSamples.length % 2 === 0
      ? ((sortedSamples[middleIndex - 1] ?? Number.NaN) +
          (sortedSamples[middleIndex] ?? Number.NaN)) /
        2
      : (sortedSamples[middleIndex] ?? Number.NaN);
  const rounded = (value: number): number => Number(value.toFixed(3));

  return {
    sampleCount: sortedSamples.length,
    medianMs: rounded(median),
    p95Ms: rounded(percentile(sortedSamples, 0.95)),
    minMs: rounded(sortedSamples[0] ?? Number.NaN),
    maxMs: rounded(sortedSamples.at(-1) ?? Number.NaN),
  };
};

const collectSynchronizedFrameSamples = async (
  canvas: Locator,
  frameCount: number,
): Promise<number[]> =>
  canvas.evaluate(async (element, count) => {
    const target = element as HTMLCanvasElement;
    const context = target.getContext("webgl2");
    if (context === null) {
      throw new Error("Point-cloud preview WebGL2 context is unavailable.");
    }

    const nextCompletedFrame = (): Promise<number> =>
      new Promise((resolve) => {
        window.requestAnimationFrame(() => {
          // R3F schedules its next frame before this callback. finish() makes
          // each sample include the renderer work completed for that frame.
          context.finish();
          resolve(performance.now());
        });
      });

    const samples: number[] = [];
    let previousFrame = await nextCompletedFrame();
    for (let frame = 0; frame < count; frame += 1) {
      const completedFrame = await nextCompletedFrame();
      samples.push(completedFrame - previousFrame);
      previousFrame = completedFrame;
    }
    return samples;
  }, frameCount);

const selectMeshColorMode = async (
  page: Page,
  mode: MeshColorMode,
): Promise<void> => {
  const modeSelect = page.getByTestId("mesh-color-mode-select");
  await modeSelect.selectOption(mode);
  await expect(modeSelect).toHaveValue(mode);
  await expect(page.getByTestId("mesh-color-mode-status")).toContainText(
    mode === "fragment-knn"
      ? "Spatial-hash fragment KNN shading active"
      : "Baked vertex colours active",
  );
};

const measureMode = async (
  page: Page,
  canvas: Locator,
  mode: MeshColorMode,
): Promise<ModeMeasurement> => {
  await selectMeshColorMode(page, mode);
  await collectSynchronizedFrameSamples(canvas, warmupFrameCount);

  const trialSamples: number[][] = [];
  for (let trial = 0; trial < trialCount; trial += 1) {
    trialSamples.push(
      await collectSynchronizedFrameSamples(canvas, framesPerTrial),
    );
  }

  const samples = trialSamples.flat();
  return {
    mode,
    stats: summarizeFrames(samples),
    trials: trialSamples.map(summarizeFrames),
  };
};

const expectValidMeasurement = (measurement: ModeMeasurement): void => {
  expect(measurement.stats.sampleCount).toBe(trialCount * framesPerTrial);
  for (const value of [
    measurement.stats.medianMs,
    measurement.stats.p95Ms,
    measurement.stats.minMs,
    measurement.stats.maxMs,
  ]) {
    expect(Number.isFinite(value)).toBe(true);
    expect(value).toBeGreaterThan(0);
  }
  expect(measurement.stats.p95Ms).toBeGreaterThanOrEqual(
    measurement.stats.medianMs,
  );
};

test("point-cloud fragment KNN reports warmed fixed-viewport frame timing", async ({
  page,
}) => {
  test.setTimeout(300_000);
  await page.setViewportSize(benchmarkViewport);
  await gotoStableApp(page, "/pointcloud-preview");

  const workloads = [buildRegularGridWorkload(), buildDenseWithinCapWorkload()];
  const canvas = page
    .getByTestId("pointcloud-preview-canvas")
    .locator("canvas");
  const results: Array<{
    readonly workload: BenchmarkWorkload["name"];
    readonly pointCount: number;
    readonly expectedMaxPointsPerCell: number;
    readonly baked: ModeMeasurement;
    readonly fragmentKnn: ModeMeasurement;
  }> = [];

  for (const workload of workloads) {
    await page.getByTestId("pointcloud-upload-input").setInputFiles({
      name: workload.fileName,
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(workload.json), "utf8"),
    });
    await expect(page.getByTestId("pointcloud-load-status")).toHaveText(
      "ready",
    );
    await expect(page.getByTestId("point-sample-count")).toHaveText(
      String(workload.pointCount),
    );
    await expect(page.getByTestId("mesh-color-mode-status")).toContainText(
      "Spatial-hash fragment KNN shading active",
    );

    const baked = await measureMode(page, canvas, "baked");
    const fragmentKnn = await measureMode(page, canvas, "fragment-knn");
    expectValidMeasurement(baked);
    expectValidMeasurement(fragmentKnn);
    results.push({
      workload: workload.name,
      pointCount: workload.pointCount,
      expectedMaxPointsPerCell: workload.expectedMaxPointsPerCell,
      baked,
      fragmentKnn,
    });
  }

  const canvasMetrics = await canvas.evaluate((element) => {
    const target = element as HTMLCanvasElement;
    const context = target.getContext("webgl2");
    if (context === null) {
      throw new Error("Point-cloud preview WebGL2 context is unavailable.");
    }
    return {
      width: target.width,
      height: target.height,
      renderer: String(context.getParameter(context.RENDERER)),
    };
  });
  expect(canvasMetrics.width).toBeGreaterThan(0);
  expect(canvasMetrics.height).toBeGreaterThan(0);

  console.log(
    "POINTCLOUD_FRAGMENT_KNN_BENCHMARK",
    JSON.stringify({
      schemaVersion: 1,
      viewport: benchmarkViewport,
      canvas: canvasMetrics,
      measurement: {
        warmupFrameCount,
        trialCount,
        framesPerTrial,
      },
      workloads: results,
    }),
  );
});
