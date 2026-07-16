import { expect, test, type Locator, type Page } from "@playwright/test";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { gotoStableApp } from "./helpers/appPage";

const validUploadJson = JSON.stringify({
  pc_xyz: [
    [0.9, 0, -0.2],
    [0.75, -0.25, 0],
    [0.75, 0.25, 0],
  ],
  pc_rgb: [
    [0.43, 0.54, 0.42],
    [0.19, 0.31, 0.44],
    [0.28, 0.4, 0.49],
  ],
  m_verts: [
    [-1, 0, 1],
    [1, 0, 1],
    [1, 0, -1],
    [-1, 0, -1],
  ],
  m_faces: [
    [0, 1, 2],
    [0, 2, 3],
  ],
});

const buildKernelPathGroup = (anchor: readonly [number, number, number]) =>
  Array.from({ length: 8 }, (_, pathIndex) => [
    anchor,
    [anchor[0] + (pathIndex + 1) * 0.08, anchor[1], anchor[2] + 0.2],
  ]);

const convolutionKernelUploadJson = JSON.stringify({
  pc_xyz: [
    [0.9, 0, -0.2],
    [0.75, -0.25, 0],
    [0.75, 0.25, 0],
  ],
  pc_rgb: [
    [0.43, 0.54, 0.42],
    [0.19, 0.31, 0.44],
    [0.28, 0.4, 0.49],
  ],
  m_verts: [
    [-1, 0, 1],
    [1, 0, 1],
    [1, 0, -1],
    [-1, 0, -1],
  ],
  m_faces: [
    [0, 1, 2],
    [0, 2, 3],
  ],
  level_0_paths: [
    buildKernelPathGroup([0, 0, 0]),
    buildKernelPathGroup([0.5, 0, 0.25]),
  ],
  level_1_paths: [buildKernelPathGroup([0.25, 0, 0.5])],
});

const alternateUploadJson = JSON.stringify({
  pc_xyz: [
    [-0.9, 0, 0.2],
    [-0.75, -0.25, 0],
    [-0.75, 0.25, 0],
    [-0.55, 0, -0.35],
    [-0.35, 0.15, 0.25],
  ],
  pc_rgb: [
    [0.9, 0.2, 0.15],
    [0.75, 0.35, 0.1],
    [0.55, 0.4, 0.2],
    [0.35, 0.45, 0.35],
    [0.15, 0.5, 0.55],
  ],
  m_verts: [
    [-1, 0, 1],
    [1, 0, 1],
    [1, 0, -1],
    [-1, 0, -1],
  ],
  m_faces: [
    [0, 1, 2],
    [0, 2, 3],
  ],
});

const appendedUploadJson = JSON.stringify({
  pc_xyz: [
    [0, 0, 0],
    [0.1, 0.2, 0.3],
    [0.2, 0.3, 0.4],
    [0.3, 0.4, 0.5],
    [0.4, 0.5, 0.6],
    [0.5, 0.6, 0.7],
    [0.6, 0.7, 0.8],
  ],
  pc_rgb: [
    [0.1, 0.1, 0.9],
    [0.2, 0.2, 0.8],
    [0.3, 0.3, 0.7],
    [0.4, 0.4, 0.6],
    [0.5, 0.5, 0.5],
    [0.6, 0.6, 0.4],
    [0.7, 0.7, 0.3],
  ],
  m_verts: [
    [-1, 0, 1],
    [1, 0, 1],
    [1, 0, -1],
    [-1, 0, -1],
  ],
  m_faces: [
    [0, 1, 2],
    [0, 2, 3],
  ],
});

const saturatedPointUploadJson = JSON.stringify({
  pc_xyz: [
    [-0.4, 0, 0],
    [0, 0, 0],
    [0.4, 0, 0],
  ],
  pc_rgb: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
  m_verts: [
    [-1, -0.5, -0.5],
    [1, -0.5, -0.5],
    [1, 0.5, -0.5],
    [-1, 0.5, -0.5],
  ],
  m_faces: [
    [0, 1, 2],
    [0, 2, 3],
  ],
});

const fragmentKnnMesh = {
  m_verts: [
    [-1, -1, 0],
    [1, -1, 0],
    [1, 1, 0],
    [-1, 1, 0],
  ],
  m_faces: [
    [0, 1, 2],
    [0, 2, 3],
  ],
} as const;

const weightedFragmentKnnPoints = [
  [1, 0, 0],
  [2, 0, 0],
  [4, 0, 0],
] as const;

const weightedFragmentKnnColors = [
  [0.25, 0.04, 0.16],
  [0.36, 0.49, 0.09],
  [0.64, 0.16, 0.81],
] as const;

const weightedFragmentKnnUploadJson = JSON.stringify({
  pc_xyz: weightedFragmentKnnPoints,
  pc_rgb: weightedFragmentKnnColors,
  ...fragmentKnnMesh,
});

const translatedFragmentKnnMin = -6_131_334;
const translatedFragmentKnnMax = 2_051_648.625;
const translatedFragmentKnnTarget = -8_013.668_945_312_5;
const translatedFragmentKnnExactColor = [0.95, 0.05, 0.1] as const;
const translatedFragmentKnnNearColor = [0.05, 0.2, 0.95] as const;
const translatedFragmentKnnFillerCount = 1_166;
const translatedFragmentKnnPoints: readonly (readonly [
  number,
  number,
  number,
])[] = [
  [translatedFragmentKnnTarget, 0, 0],
  [translatedFragmentKnnTarget + 0.03125, 0, 0],
  [translatedFragmentKnnTarget + 0.0625, 0, 0],
  [translatedFragmentKnnTarget + 0.09375, 0, 0],
  ...Array.from(
    { length: translatedFragmentKnnFillerCount },
    (_, index) =>
      [
        Math.fround(
          translatedFragmentKnnMin +
            ((translatedFragmentKnnMax - translatedFragmentKnnMin) * index) /
              (translatedFragmentKnnFillerCount - 1),
        ),
        0,
        0,
      ] as const,
  ),
];
const translatedFragmentKnnColors: readonly (readonly [
  number,
  number,
  number,
])[] = [
  translatedFragmentKnnExactColor,
  translatedFragmentKnnNearColor,
  translatedFragmentKnnNearColor,
  translatedFragmentKnnNearColor,
  ...Array.from(
    { length: translatedFragmentKnnFillerCount },
    () => [0.1, 0.1, 0.1] as const,
  ),
];
const translatedFragmentKnnUploadJson = JSON.stringify({
  pc_xyz: translatedFragmentKnnPoints,
  pc_rgb: translatedFragmentKnnColors,
  m_verts: [
    [translatedFragmentKnnTarget - 1, -1, 0],
    [translatedFragmentKnnTarget + 1, -1, 0],
    [translatedFragmentKnnTarget + 1, 1, 0],
    [translatedFragmentKnnTarget - 1, 1, 0],
  ],
  m_faces: [
    [0, 1, 2],
    [0, 2, 3],
  ],
});

const largeUploadJson = JSON.stringify({
  pc_xyz: Array.from({ length: 513 }, (_, index) => [
    (index % 19) / 18,
    (Math.floor(index / 19) % 9) / 8,
    Math.floor(index / 171) / 3,
  ]),
  pc_rgb: Array.from({ length: 513 }, (_, index) => [
    (index % 7) / 6,
    (index % 11) / 10,
    (index % 13) / 12,
  ]),
  m_verts: [
    [-1, 0, 1],
    [1, 0, 1],
    [1, 0, -1],
    [-1, 0, -1],
  ],
  m_faces: [
    [0, 1, 2],
    [0, 2, 3],
  ],
});

const denseCellUploadJson = JSON.stringify({
  pc_xyz: Array.from({ length: 300 }, () => [0.5, 0.5, 0.5]),
  pc_rgb: Array.from({ length: 300 }, (_, index) => [
    (index % 7) / 6,
    (index % 11) / 10,
    (index % 13) / 12,
  ]),
  m_verts: [
    [-1, 0, 1],
    [1, 0, 1],
    [1, 0, -1],
    [-1, 0, -1],
  ],
  m_faces: [
    [0, 1, 2],
    [0, 2, 3],
  ],
});

const ablationExperimentFilename = ({
  contentSamplesPerFace,
  distanceMeasure,
  outputStep,
}: {
  readonly contentSamplesPerFace: number;
  readonly distanceMeasure: "EUCLIDEAN" | "SPECTRAL";
  readonly outputStep?: number | null;
}): string => {
  const outputStepSuffix =
    outputStep === undefined || outputStep === null ? "" : `_step${outputStep}`;
  return `1sw_0cw_0.1tv_L1_${contentSamplesPerFace}c-spf_192x256s-img_SIMPLE_AXIS_RAW_COLORS_MAX_${distanceMeasure}_4knn_0.7rf_2gr_1.5std-attn_300steps${outputStepSuffix}.json`;
};

const ablationCellTestId = (
  contentSamplesPerFace: number,
  distanceMeasure: "EUCLIDEAN" | "SPECTRAL",
): string =>
  `pointcloud-ablation-cell-contentSamplesPerFace=number:${contentSamplesPerFace}|distanceMeasure=string:${distanceMeasure}`;

const readPreviewBackgroundPixel = async (
  previewCanvas: Locator,
): Promise<readonly number[]> =>
  previewCanvas.locator("canvas").evaluate((canvas) => {
    const context =
      (canvas as HTMLCanvasElement).getContext("webgl2") ??
      (canvas as HTMLCanvasElement).getContext("webgl");
    if (context === null) {
      throw new Error("Point-cloud preview WebGL context is unavailable.");
    }

    const pixel = new Uint8Array(4);
    context.readPixels(1, 1, 1, 1, context.RGBA, context.UNSIGNED_BYTE, pixel);
    return Array.from(pixel);
  });

type FragmentKnnCanvasSample = {
  readonly pixel: readonly [number, number];
  readonly position: readonly [number, number, number];
};

const snapCameraToPositiveZ = async (page: Page): Promise<void> => {
  await page.getByTestId("snap-axis-pos-z").click();
  await expect
    .poll(async () => {
      const state = await readCameraState(page.getByTestId("camera-state"));
      if (state === null) {
        return false;
      }
      const offset = state.position.map(
        (value, index) => value - state.target[index],
      );
      return (
        offset[2] > 0 &&
        Math.abs(offset[0]) <= 1e-5 &&
        Math.abs(offset[1]) <= 1e-5
      );
    })
    .toBe(true);
};

const fragmentKnnCanvasSample = async (
  page: Page,
  previewCanvas: Locator,
  cameraDistance?: number,
): Promise<FragmentKnnCanvasSample> => {
  const { width, height } = await previewCanvas
    .locator("canvas")
    .evaluate((canvas) => ({
      width: (canvas as HTMLCanvasElement).width,
      height: (canvas as HTMLCanvasElement).height,
    }));
  const pixel = [Math.floor(width / 2), Math.floor(height / 2)] as const;
  const cameraState = await readCameraState(page.getByTestId("camera-state"));
  if (cameraState === null) {
    throw new Error("Point-cloud preview camera state is unavailable.");
  }

  const distance =
    cameraDistance ?? cameraState.position[2] - cameraState.target[2];
  const halfHeightAtTarget = distance * Math.tan((42 * Math.PI) / 360);
  const normalizedX = ((pixel[0] + 0.5) / width) * 2 - 1;
  const normalizedY = ((pixel[1] + 0.5) / height) * 2 - 1;
  return {
    pixel,
    position: [
      cameraState.target[0] +
        normalizedX * halfHeightAtTarget * (width / height),
      cameraState.target[1] + normalizedY * halfHeightAtTarget,
      cameraState.target[2],
    ],
  };
};

const readPreviewPixel = async (
  previewCanvas: Locator,
  pixel: readonly [number, number],
): Promise<readonly [number, number, number]> =>
  previewCanvas.locator("canvas").evaluate((canvas, [x, y]) => {
    const context =
      (canvas as HTMLCanvasElement).getContext("webgl2") ??
      (canvas as HTMLCanvasElement).getContext("webgl");
    if (context === null) {
      throw new Error("Point-cloud preview WebGL context is unavailable.");
    }

    const color = new Uint8Array(4);
    context.readPixels(x, y, 1, 1, context.RGBA, context.UNSIGNED_BYTE, color);
    return [color[0], color[1], color[2]] as const;
  }, pixel);

const encodeGammaChannel = (value: number): number =>
  value <= 0.0031308
    ? value * 12.92
    : 1.055 * Math.pow(Math.max(value, 0), 1 / 2.4) - 0.055;

const expectedFragmentKnnColor = ({
  position,
  points,
  colors,
  gammaDecodingEnabled,
  brightness,
}: {
  readonly position: readonly [number, number, number];
  readonly points: readonly (readonly [number, number, number])[];
  readonly colors: readonly (readonly [number, number, number])[];
  readonly gammaDecodingEnabled: boolean;
  readonly brightness: number;
}): readonly [number, number, number] => {
  const neighbors = points
    .map((point, index) => ({
      color: colors[index]!,
      squaredDistance:
        Math.pow(point[0] - position[0], 2) +
        Math.pow(point[1] - position[1], 2) +
        Math.pow(point[2] - position[2], 2),
    }))
    .sort((left, right) => left.squaredDistance - right.squaredDistance)
    .slice(0, 3);
  const adjustedColors = neighbors.map(({ color }) =>
    color.map((value) =>
      Math.min(
        1,
        (gammaDecodingEnabled ? value : encodeGammaChannel(value)) * brightness,
      ),
    ),
  );
  if (neighbors[0]!.squaredDistance <= 1e-12) {
    return adjustedColors[0] as [number, number, number];
  }

  const weights = neighbors.map(
    ({ squaredDistance }) => 1 / Math.max(squaredDistance, 1e-12),
  );
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  return [0, 1, 2].map(
    (channel) =>
      neighbors.reduce(
        (sum, _, index) =>
          sum + adjustedColors[index]![channel]! * weights[index]!,
        0,
      ) / totalWeight,
  ) as [number, number, number];
};

const expectPreviewPixelColor = async (
  previewCanvas: Locator,
  pixel: readonly [number, number],
  expectedColor: readonly [number, number, number],
): Promise<void> => {
  const expectedBytes = expectedColor.map((value) => Math.round(value * 255));
  await expect
    .poll(async () => {
      const actual = await readPreviewPixel(previewCanvas, pixel);
      return Math.max(
        ...actual.map((value, index) =>
          Math.abs(value - (expectedBytes[index] ?? 0)),
        ),
      );
    })
    .toBeLessThanOrEqual(3);
};

const readPreviewColorPixelCount = async (
  previewCanvas: Locator,
): Promise<number> =>
  previewCanvas.locator("canvas").evaluate((canvas) => {
    const target = canvas as HTMLCanvasElement;
    const context = target.getContext("webgl2") ?? target.getContext("webgl");
    if (context === null) {
      throw new Error("Point-cloud preview WebGL context is unavailable.");
    }

    const width = target.width;
    const height = target.height;
    const pixels = new Uint8Array(width * height * 4);
    context.readPixels(
      0,
      0,
      width,
      height,
      context.RGBA,
      context.UNSIGNED_BYTE,
      pixels,
    );

    let colorPixelCount = 0;
    for (let offset = 0; offset < pixels.length; offset += 4) {
      const red = pixels[offset] ?? 0;
      const green = pixels[offset + 1] ?? 0;
      const blue = pixels[offset + 2] ?? 0;
      const maxChannel = Math.max(red, green, blue);
      const minChannel = Math.min(red, green, blue);
      if (maxChannel >= 180 && maxChannel - minChannel >= 100) {
        colorPixelCount += 1;
      }
    }
    return colorPixelCount;
  });

const readCameraState = async (
  cameraState: Locator,
): Promise<{
  readonly position: readonly [number, number, number];
  readonly target: readonly [number, number, number];
} | null> => {
  const text = await cameraState.textContent();
  if (text === null || text === "unavailable") {
    return null;
  }
  return JSON.parse(text) as {
    readonly position: readonly [number, number, number];
    readonly target: readonly [number, number, number];
  };
};

const snapCameraToPositiveX = async (
  cameraState: Locator,
  snapButton: Locator,
): Promise<string> => {
  await expect.poll(() => readCameraState(cameraState)).not.toBeNull();
  await snapButton.click();
  await expect
    .poll(
      async () => {
        const state = await readCameraState(cameraState);
        if (state === null) {
          return false;
        }
        const offset = state.position.map(
          (value, index) => value - state.target[index],
        );
        return (
          offset[0] > 0 &&
          Math.abs(offset[1]) <= 1e-5 &&
          Math.abs(offset[2]) <= 1e-5
        );
      },
      { timeout: 10_000 },
    )
    .toBe(true);
  return (await cameraState.textContent())!;
};

const readStoredZipEntries = (archive: Buffer): ReadonlyMap<string, Buffer> => {
  const endSignature = 0x06054b50;
  let endOffset = archive.length - 22;
  while (endOffset >= 0 && archive.readUInt32LE(endOffset) !== endSignature) {
    endOffset -= 1;
  }
  expect(endOffset).toBeGreaterThanOrEqual(0);

  const entryCount = archive.readUInt16LE(endOffset + 10);
  let centralOffset = archive.readUInt32LE(endOffset + 16);
  const entries = new Map<string, Buffer>();
  for (let index = 0; index < entryCount; index += 1) {
    expect(archive.readUInt32LE(centralOffset)).toBe(0x02014b50);
    const compressionMethod = archive.readUInt16LE(centralOffset + 10);
    const compressedSize = archive.readUInt32LE(centralOffset + 20);
    const nameLength = archive.readUInt16LE(centralOffset + 28);
    const extraLength = archive.readUInt16LE(centralOffset + 30);
    const commentLength = archive.readUInt16LE(centralOffset + 32);
    const localOffset = archive.readUInt32LE(centralOffset + 42);
    const name = archive
      .subarray(centralOffset + 46, centralOffset + 46 + nameLength)
      .toString("utf8");

    expect(compressionMethod).toBe(0);
    expect(archive.readUInt32LE(localOffset)).toBe(0x04034b50);
    const localNameLength = archive.readUInt16LE(localOffset + 26);
    const localExtraLength = archive.readUInt16LE(localOffset + 28);
    const dataOffset = localOffset + 30 + localNameLength + localExtraLength;
    entries.set(
      name,
      archive.subarray(dataOffset, dataOffset + compressedSize),
    );
    centralOffset += 46 + nameLength + extraLength + commentLength;
  }
  return entries;
};

const readPngSize = (
  png: Buffer,
): { readonly width: number; readonly height: number } => {
  expect(png.subarray(0, 8)).toEqual(
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
  );
  return {
    width: png.readUInt32BE(16),
    height: png.readUInt32BE(20),
  };
};

test("point-cloud preview boots from the standalone route with the bundled demo", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  await expect(
    page.getByRole("heading", {
      name: /Point-Cloud Mesh Preview/i,
    }),
  ).toBeVisible();
  await expect(page.getByTestId("pointcloud-preview-canvas")).toBeVisible();
  await expect(page.getByTestId("pointcloud-source-label")).toHaveText(
    "Bundled medium example",
  );
  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");
  await expect(page.getByTestId("mesh-vertex-count")).toHaveText("64");
  await expect(page.getByTestId("mesh-face-count")).toHaveText("98");
  await expect(page.getByTestId("point-sample-count")).toHaveText("191");
  await expect(page.getByTestId("mesh-color-mode-select")).toHaveValue(
    "fragment-knn",
  );
  await expect(page.getByTestId("mesh-color-mode-status")).toContainText(
    /Fragment KNN shading active/i,
  );
  await expect(page.getByTestId("pointcloud-fps")).toContainText(/FPS/i);
  await expect(
    page.getByRole("button", { name: /Upload point-cloud mesh/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /Reload preview/i }),
  ).toBeVisible();
  await expect(page.getByTestId("screenshot-button")).toBeVisible();
  await expect(page.getByTestId("batch-screenshot-button")).toHaveCount(0);
  await expect(page.getByTestId("save-viewpoint-button")).toBeEnabled();
  await expect(page.getByTestId("swap-yz-button")).toBeVisible();
});

test("point-cloud preview hosts a filename-only ablation tab shell", async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 1600, height: 720 });
  await gotoStableApp(page, "/pointcloud-preview");

  await expect(page.getByTestId("pointcloud-preview-tab")).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(page.getByTestId("pointcloud-ablation-tab")).toHaveAttribute(
    "aria-selected",
    "false",
  );
  await expect(page.getByTestId("pointcloud-preview-canvas")).toBeVisible();
  await expect(page.getByTestId("screenshot-button")).toBeVisible();

  await page.getByTestId("pointcloud-ablation-tab").click();
  await expect(page.getByTestId("pointcloud-ablation-tab")).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(page.getByTestId("pointcloud-preview-canvas")).toBeHidden();
  await expect(page.getByTestId("pointcloud-ablation-empty")).toBeVisible();
  await expect(
    page.getByTestId("pointcloud-ablation-selected-count"),
  ).toHaveText("0");
  await expect(page.getByTestId("pointcloud-ablation-parsed-count")).toHaveText(
    "0",
  );
  await expect(
    page.getByTestId("pointcloud-ablation-unparsed-count"),
  ).toHaveText("0");

  const ablationFolderInput = page.getByTestId(
    "pointcloud-ablation-folder-input",
  );
  const ablationFileInput = page.getByTestId("pointcloud-ablation-file-input");
  const invalidAblationDir = testInfo.outputPath("invalid-ablation");
  await mkdir(invalidAblationDir, { recursive: true });
  await writeFile(join(invalidAblationDir, "notes.json"), "not json");
  await writeFile(join(invalidAblationDir, "README.json"), "not json");
  await ablationFolderInput.setInputFiles(invalidAblationDir);
  await expect(
    page.getByTestId("pointcloud-ablation-selected-count"),
  ).toHaveText("2");
  await expect(page.getByTestId("pointcloud-ablation-parsed-count")).toHaveText(
    "0",
  );
  await expect(
    page.getByTestId("pointcloud-ablation-unparsed-count"),
  ).toHaveText("2");
  await expect(
    page.getByTestId("pointcloud-ablation-no-summaries"),
  ).toBeVisible();

  const mixedAblationDir = testInfo.outputPath("mixed-ablation");
  await mkdir(mixedAblationDir, { recursive: true });
  await writeFile(
    join(
      mixedAblationDir,
      "1sw_2cw_0tv_L2_8c-spf_SIMPLE_AXIS_RAW_COLORS_MAX_EUCLIDEAN_4knn_0.7rf_2gr_3.0std-attn_120steps.json",
    ),
    "not json",
  );
  await writeFile(
    join(
      mixedAblationDir,
      "5sw_2cw_0tv_L1_4c-spf_192x256s-img_PCP_LOGIT_AVG_SPECTRAL_8knn_1.2rf_1gr_1.0std-attn_60steps_step30.json",
    ),
    "not json",
  );
  await writeFile(join(mixedAblationDir, "scratch.json"), "not json");
  await ablationFolderInput.setInputFiles(mixedAblationDir);
  await expect(
    page.getByTestId("pointcloud-ablation-selected-count"),
  ).toHaveText("3");
  await expect(page.getByTestId("pointcloud-ablation-parsed-count")).toHaveText(
    "2",
  );
  await expect(
    page.getByTestId("pointcloud-ablation-unparsed-count"),
  ).toHaveText("1");
  await expect(
    page.getByTestId("pointcloud-ablation-summary-styleWeight"),
  ).toContainText("1");
  await expect(
    page.getByTestId("pointcloud-ablation-summary-styleWeight"),
  ).toContainText("5");
  await expect(
    page.getByTestId("pointcloud-ablation-summary-tvMode"),
  ).toContainText("L1");
  await expect(
    page.getByTestId("pointcloud-ablation-summary-tvMode"),
  ).toContainText("L2");
  await expect(
    page.getByTestId("pointcloud-ablation-summary-styleResolution"),
  ).toContainText("192x256");
  await expect(
    page.getByTestId("pointcloud-ablation-summary-outputStep"),
  ).toContainText("step30");

  await ablationFileInput.setInputFiles([
    {
      name: "1sw_0cw_0.1tv_L1_2c-spf_192x256s-img_SIMPLE_AXIS_RAW_COLORS_MAX_EUCLIDEAN_4knn_0.7rf_2gr_1.5std-attn_300steps_step60.json",
      mimeType: "application/json",
      buffer: Buffer.from("not json", "utf8"),
    },
    {
      name: "1sw_0cw_0.1tv_L1_4c-spf_192x256s-img_SIMPLE_AXIS_RAW_COLORS_MAX_SPECTRAL_4knn_0.7rf_2gr_1.5std-attn_300steps_step120.json",
      mimeType: "application/json",
      buffer: Buffer.from("not json", "utf8"),
    },
  ]);
  await expect(
    page.getByTestId("pointcloud-ablation-selected-count"),
  ).toHaveText("2");
  await expect(page.getByTestId("pointcloud-ablation-parsed-count")).toHaveText(
    "2",
  );
  await expect(
    page.getByTestId("pointcloud-ablation-unparsed-count"),
  ).toHaveText("0");
  await expect(
    page.getByTestId("pointcloud-ablation-summary-contentSamplesPerFace"),
  ).toContainText("2");
  await expect(
    page.getByTestId("pointcloud-ablation-summary-contentSamplesPerFace"),
  ).toContainText("4");
  await expect(
    page.getByTestId("pointcloud-ablation-summary-outputStep"),
  ).toContainText("step60");
  await expect(
    page.getByTestId("pointcloud-ablation-summary-outputStep"),
  ).toContainText("step120");
  await expect
    .poll(async () =>
      page
        .getByTestId("pointcloud-ablation-scroll-region")
        .evaluate((element) => element.scrollHeight > element.clientHeight),
    )
    .toBe(true);
  await page
    .getByTestId("pointcloud-ablation-scroll-region")
    .evaluate((element) => {
      element.scrollTop = element.scrollHeight;
    });
  await expect
    .poll(async () =>
      page
        .getByTestId("pointcloud-ablation-scroll-region")
        .evaluate((element) => element.scrollTop),
    )
    .toBeGreaterThan(0);

  await page.getByTestId("pointcloud-preview-tab").click();
  await expect(page.getByTestId("pointcloud-preview-tab")).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(page.getByTestId("pointcloud-preview-canvas")).toBeVisible();
  await expect(
    page.getByRole("button", { name: /Upload point-cloud mesh/i }),
  ).toBeVisible();
  await expect(page.getByTestId("screenshot-button")).toBeVisible();
  await expect(page.getByTestId("save-viewpoint-button")).toBeEnabled();
});

test("point-cloud ablation matrix filters cells and previews a unique experiment", async ({
  page,
}, testInfo) => {
  await gotoStableApp(page, "/pointcloud-preview");

  const cameraStateBeforePreview = await snapCameraToPositiveX(
    page.getByTestId("camera-state"),
    page.getByTestId("snap-axis-pos-x"),
  );
  await page.getByTestId("save-viewpoint-button").click();
  await page.getByTestId("viewpoint-name-1").fill("Export view");

  await page.getByTestId("pointcloud-ablation-tab").click();
  await expect(page.getByTestId("pointcloud-ablation-empty")).toBeVisible();

  const matrixDir = testInfo.outputPath("ablation-matrix");
  const step60Dir = join(matrixDir, "step60");
  const step120aDir = join(matrixDir, "step120-a");
  const step120bDir = join(matrixDir, "step120-b");
  const spectralDir = join(matrixDir, "spectral");
  await mkdir(step60Dir, { recursive: true });
  await mkdir(step120aDir, { recursive: true });
  await mkdir(step120bDir, { recursive: true });
  await mkdir(spectralDir, { recursive: true });

  const uniqueStep60Filename = ablationExperimentFilename({
    contentSamplesPerFace: 2,
    distanceMeasure: "EUCLIDEAN",
    outputStep: 60,
  });
  const duplicateStep120Filename = ablationExperimentFilename({
    contentSamplesPerFace: 2,
    distanceMeasure: "EUCLIDEAN",
    outputStep: 120,
  });
  const spectralStep120Filename = ablationExperimentFilename({
    contentSamplesPerFace: 4,
    distanceMeasure: "SPECTRAL",
    outputStep: 120,
  });

  await writeFile(join(step60Dir, uniqueStep60Filename), validUploadJson);
  await writeFile(join(step120aDir, duplicateStep120Filename), "not json");
  await writeFile(
    join(step120bDir, duplicateStep120Filename),
    '{"m_verts":"broken"}',
  );
  await writeFile(join(spectralDir, spectralStep120Filename), "not json");

  await page
    .getByTestId("pointcloud-ablation-folder-input")
    .setInputFiles(matrixDir);

  await expect(
    page.getByTestId("pointcloud-ablation-selected-count"),
  ).toHaveText("4");
  await expect(page.getByTestId("pointcloud-ablation-parsed-count")).toHaveText(
    "4",
  );
  await expect(
    page.getByTestId("pointcloud-ablation-unparsed-count"),
  ).toHaveText("0");
  await expect(
    page.getByTestId("pointcloud-ablation-multi-view-ambiguous"),
  ).toContainText("1 duplicate setting is excluded");
  await expect(
    page
      .getByTestId("pointcloud-ablation-multi-view-setting-select")
      .locator("option"),
  ).toHaveCount(2);
  await expect(
    page.getByTestId("pointcloud-ablation-x-axis-select"),
  ).toHaveValue("contentSamplesPerFace");
  await expect(
    page.getByTestId("pointcloud-ablation-y-axis-select"),
  ).toHaveValue("distanceMeasure");
  const outputStepSelect = page.getByTestId(
    "pointcloud-ablation-fixed-outputStep-select",
  );
  await expect
    .poll(() =>
      outputStepSelect.evaluate((select) =>
        Array.from(
          (select as HTMLSelectElement).selectedOptions,
          (option) => option.value,
        ),
      ),
    )
    .toEqual(["number:120"]);

  const ambiguousCell = page.getByTestId(ablationCellTestId(2, "EUCLIDEAN"));
  await expect(ambiguousCell).toHaveAttribute("data-status", "ambiguous");
  await expect(ambiguousCell).toContainText("Ambiguous x2");
  await expect(ambiguousCell.getByRole("button")).toHaveCount(0);

  const missingCell = page.getByTestId(ablationCellTestId(4, "EUCLIDEAN"));
  await expect(missingCell).toHaveAttribute("data-status", "missing");
  await expect(missingCell).toContainText("Missing");
  await expect(missingCell.getByRole("button")).toHaveCount(0);

  const availableCell = page.getByTestId(ablationCellTestId(4, "SPECTRAL"));
  await expect(availableCell).toHaveAttribute("data-status", "available");
  await expect(
    availableCell.getByRole("button", { name: "Available" }),
  ).toBeVisible();
  await expect(
    page.getByTestId("pointcloud-ablation-export-viewpoint-select"),
  ).toHaveValue("1");
  await expect(
    page.getByTestId("pointcloud-ablation-export-blocked"),
  ).toContainText("Resolve 1 ambiguous cell before export.");
  await expect(
    page.getByTestId("pointcloud-ablation-export-button"),
  ).toBeDisabled();

  await outputStepSelect.selectOption("number:60");
  await expect(ambiguousCell).toHaveAttribute("data-status", "available");
  await expect(
    page.getByTestId(
      "pointcloud-ablation-column-contentSamplesPerFace-number:4",
    ),
  ).toHaveCount(0);
  await expect(
    page.getByTestId("pointcloud-ablation-row-distanceMeasure-string:SPECTRAL"),
  ).toHaveCount(0);
  await expect(
    page.getByTestId("pointcloud-ablation-export-button"),
  ).toBeEnabled();

  const downloadPromise = page.waitForEvent("download");
  await page.getByTestId("pointcloud-ablation-export-button").click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(
    /^pointcloud-ablation-grid-export-view-\d{8}-\d{6}\.png$/,
  );
  const downloadPath = await download.path();
  expect(downloadPath).not.toBeNull();
  const png = await readFile(downloadPath!);
  expect(readPngSize(png)).toEqual({ width: 580, height: 358 });
  await expect(page.getByTestId("pointcloud-ablation-tab")).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(
    page.getByTestId("pointcloud-ablation-export-button"),
  ).toBeEnabled();

  await ambiguousCell.getByRole("button", { name: "Available" }).click();
  await expect(page.getByTestId("pointcloud-preview-tab")).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(page.getByTestId("pointcloud-source-label")).toContainText(
    uniqueStep60Filename,
  );
  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");
  await expect(page.getByTestId("point-sample-count")).toHaveText("3");
  await expect(
    page.locator('[data-testid^="pointcloud-upload-row-"]'),
  ).toHaveCount(0);
  await expect(page.getByTestId("camera-state")).toHaveText(
    cameraStateBeforePreview,
  );
});

test("point-cloud ablation output-step fixed filter supports multiple selections", async ({
  page,
}, testInfo) => {
  await gotoStableApp(page, "/pointcloud-preview");
  await page.getByTestId("pointcloud-ablation-tab").click();

  const matrixDir = testInfo.outputPath("ablation-output-step-multiselect");
  await mkdir(matrixDir, { recursive: true });

  const filenames = [
    ablationExperimentFilename({
      contentSamplesPerFace: 2,
      distanceMeasure: "EUCLIDEAN",
      outputStep: 120,
    }),
    ablationExperimentFilename({
      contentSamplesPerFace: 4,
      distanceMeasure: "SPECTRAL",
      outputStep: 220,
    }),
    ablationExperimentFilename({
      contentSamplesPerFace: 2,
      distanceMeasure: "EUCLIDEAN",
      outputStep: 301,
    }),
    ablationExperimentFilename({
      contentSamplesPerFace: 4,
      distanceMeasure: "SPECTRAL",
      outputStep: 302,
    }),
    ablationExperimentFilename({
      contentSamplesPerFace: 2,
      distanceMeasure: "SPECTRAL",
      outputStep: 305,
    }),
    ablationExperimentFilename({
      contentSamplesPerFace: 4,
      distanceMeasure: "EUCLIDEAN",
      outputStep: 320,
    }),
  ];
  await Promise.all(
    filenames.map((filename) =>
      writeFile(join(matrixDir, filename), validUploadJson),
    ),
  );

  await page
    .getByTestId("pointcloud-ablation-folder-input")
    .setInputFiles(matrixDir);
  await expect(
    page.getByTestId("pointcloud-ablation-x-axis-select"),
  ).toHaveValue("contentSamplesPerFace");
  await expect(
    page.getByTestId("pointcloud-ablation-y-axis-select"),
  ).toHaveValue("distanceMeasure");

  const outputStepSelect = page.getByTestId(
    "pointcloud-ablation-fixed-outputStep-select",
  );
  await expect(outputStepSelect).toHaveAttribute("multiple", "");
  await expect
    .poll(() =>
      outputStepSelect.evaluate((select) =>
        Array.from(
          (select as HTMLSelectElement).selectedOptions,
          (option) => option.value,
        ),
      ),
    )
    .toEqual(["number:320"]);

  await outputStepSelect.selectOption([
    "number:301",
    "number:302",
    "number:305",
    "number:320",
  ]);
  await expect
    .poll(() =>
      outputStepSelect.evaluate((select) =>
        Array.from(
          (select as HTMLSelectElement).selectedOptions,
          (option) => option.value,
        ),
      ),
    )
    .toEqual(["number:301", "number:302", "number:305", "number:320"]);

  await expect(
    page.getByTestId(ablationCellTestId(2, "EUCLIDEAN")),
  ).toHaveAttribute("data-status", "available");
  await expect(
    page.getByTestId(ablationCellTestId(4, "SPECTRAL")),
  ).toHaveAttribute("data-status", "available");
  await expect(
    page.getByTestId(ablationCellTestId(2, "SPECTRAL")),
  ).toHaveAttribute("data-status", "available");
  await expect(
    page.getByTestId(ablationCellTestId(4, "EUCLIDEAN")),
  ).toHaveAttribute("data-status", "available");

  await outputStepSelect.selectOption(["number:120", "number:220"]);
  await expect(
    page.getByTestId(ablationCellTestId(2, "EUCLIDEAN")),
  ).toHaveAttribute("data-status", "available");
  await expect(
    page.getByTestId(ablationCellTestId(4, "SPECTRAL")),
  ).toHaveAttribute("data-status", "available");
  await expect(
    page.getByTestId(ablationCellTestId(2, "SPECTRAL")),
  ).toHaveAttribute("data-status", "missing");
  await expect(
    page.getByTestId(ablationCellTestId(4, "EUCLIDEAN")),
  ).toHaveAttribute("data-status", "missing");

  await outputStepSelect.selectOption([
    "number:301",
    "number:302",
    "number:305",
    "number:320",
  ]);
  await page.reload();
  await page.getByTestId("pointcloud-ablation-tab").click();
  await page
    .getByTestId("pointcloud-ablation-folder-input")
    .setInputFiles(matrixDir);
  await expect
    .poll(() =>
      page
        .getByTestId("pointcloud-ablation-fixed-outputStep-select")
        .evaluate((select) =>
          Array.from(
            (select as HTMLSelectElement).selectedOptions,
            (option) => option.value,
          ),
        ),
    )
    .toEqual(["number:301", "number:302", "number:305", "number:320"]);
});

test("point-cloud ablation exports one setting from multiple saved viewpoints", async ({
  page,
}, testInfo) => {
  await page.addInitScript(() => {
    const originalToBlob = HTMLCanvasElement.prototype.toBlob;
    HTMLCanvasElement.prototype.toBlob = function (callback, type, quality) {
      window.setTimeout(() => {
        originalToBlob.call(this, callback, type, quality);
      }, 500);
    };
  });
  await gotoStableApp(page, "/pointcloud-preview");

  await page.getByTestId("save-viewpoint-button").click();
  await page.getByTestId("viewpoint-name-1").fill("Default view");
  await snapCameraToPositiveX(
    page.getByTestId("camera-state"),
    page.getByTestId("snap-axis-pos-x"),
  );
  await page.getByTestId("save-viewpoint-button").click();
  await page.getByTestId("viewpoint-name-2").fill("Side view");

  const experimentDir = testInfo.outputPath("ablation-multi-view");
  await mkdir(experimentDir, { recursive: true });
  const filenames = [
    ablationExperimentFilename({
      contentSamplesPerFace: 2,
      distanceMeasure: "EUCLIDEAN",
      outputStep: 60,
    }),
    ablationExperimentFilename({
      contentSamplesPerFace: 4,
      distanceMeasure: "SPECTRAL",
      outputStep: 120,
    }),
  ];
  await Promise.all(
    filenames.map((filename) =>
      writeFile(join(experimentDir, filename), validUploadJson),
    ),
  );

  await page.getByTestId("pointcloud-ablation-tab").click();
  await page
    .getByTestId("pointcloud-ablation-folder-input")
    .setInputFiles(experimentDir);

  await expect(
    page.getByTestId("pointcloud-ablation-multi-view-export"),
  ).toBeVisible();
  const settingSelect = page.getByTestId(
    "pointcloud-ablation-multi-view-setting-select",
  );
  await expect(settingSelect.locator("option")).toHaveCount(2);
  const secondSettingValue = await settingSelect
    .locator("option")
    .nth(1)
    .getAttribute("value");
  expect(secondSettingValue).not.toBeNull();
  await settingSelect.selectOption(secondSettingValue!);
  await expect(settingSelect).toHaveValue(secondSettingValue!);

  await expect(
    page.getByTestId("pointcloud-ablation-multi-view-viewpoint-1"),
  ).toBeChecked();
  await expect(
    page.getByTestId("pointcloud-ablation-multi-view-viewpoint-2"),
  ).toBeChecked();
  await expect(
    page.getByTestId("pointcloud-ablation-multi-view-selected-count"),
  ).toHaveText("2 of 2 selected");

  await page.getByTestId("pointcloud-ablation-multi-view-select-all").click();
  await expect(
    page.getByTestId("pointcloud-ablation-multi-view-export-button"),
  ).toBeDisabled();
  await expect(
    page.getByTestId("pointcloud-ablation-multi-view-selection-empty"),
  ).toBeVisible();
  const firstViewpointCheckbox = page.getByTestId(
    "pointcloud-ablation-multi-view-viewpoint-1",
  );
  await firstViewpointCheckbox.click();
  await expect(firstViewpointCheckbox).toBeChecked();
  await expect(
    page.getByTestId("pointcloud-ablation-multi-view-selected-count"),
  ).toHaveText("1 of 2 selected");
  await expect(page.locator("#root > main")).toBeVisible();

  await firstViewpointCheckbox.click();
  await expect(
    page.getByTestId("pointcloud-ablation-multi-view-export-button"),
  ).toBeDisabled();
  await page.reload();
  await expect(page.getByTestId("camera-state")).not.toHaveText("unavailable");
  const cameraStateBeforeExport = await page
    .getByTestId("camera-state")
    .textContent();
  expect(cameraStateBeforeExport).not.toBeNull();
  await page.getByTestId("pointcloud-ablation-tab").click();
  await page
    .getByTestId("pointcloud-ablation-folder-input")
    .setInputFiles(experimentDir);
  await expect(
    page.getByTestId("pointcloud-ablation-multi-view-setting-select"),
  ).toHaveValue(secondSettingValue!);
  await expect(
    page.getByTestId("pointcloud-ablation-multi-view-viewpoint-1"),
  ).not.toBeChecked();
  const secondViewpointCheckbox = page.getByTestId(
    "pointcloud-ablation-multi-view-viewpoint-2",
  );
  await expect(secondViewpointCheckbox).not.toBeChecked();
  await secondViewpointCheckbox.click();
  await expect(secondViewpointCheckbox).toBeChecked();
  await expect(
    page.getByTestId("pointcloud-ablation-multi-view-selected-count"),
  ).toHaveText("1 of 2 selected");
  await expect(page.locator("#root > main")).toBeVisible();
  await expect(
    page.getByTestId("pointcloud-ablation-multi-view-export-button"),
  ).toBeEnabled();

  await page.getByTestId("pointcloud-ablation-multi-view-select-all").click();
  await expect(
    page.getByTestId("pointcloud-ablation-multi-view-selected-count"),
  ).toHaveText("2 of 2 selected");

  const downloadPromise = page.waitForEvent("download");
  await page
    .getByTestId("pointcloud-ablation-multi-view-export-button")
    .click();
  await expect(
    page.getByTestId("pointcloud-ablation-export-overlay"),
  ).toContainText("Exporting viewpoint sheet");
  await expect(page.getByTestId("pointcloud-preview-tab")).toBeDisabled();
  await expect(page.getByTestId("pointcloud-ablation-tab")).toBeDisabled();
  await expect(page.locator("main")).toHaveAttribute("inert", "");
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(
    /^pointcloud-ablation-viewpoints-.*-\d{8}-\d{6}\.png$/,
  );
  const downloadPath = await download.path();
  expect(downloadPath).not.toBeNull();
  const png = await readFile(downloadPath!);
  expect(readPngSize(png)).toEqual({ width: 724, height: 412 });
  await expect(page.getByTestId("pointcloud-ablation-tab")).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(
    page.getByTestId("pointcloud-ablation-multi-view-export-button"),
  ).toBeEnabled();

  await page.getByTestId("pointcloud-preview-tab").click();
  await expect(page.getByTestId("camera-state")).toHaveText(
    cameraStateBeforeExport!,
  );
});

test("point-cloud ablation matrix hides empty rows and columns", async ({
  page,
}, testInfo) => {
  await gotoStableApp(page, "/pointcloud-preview");
  await page.getByTestId("pointcloud-ablation-tab").click();

  const matrixDir = testInfo.outputPath("ablation-empty-axes");
  await mkdir(matrixDir, { recursive: true });

  const filenames = [
    ablationExperimentFilename({
      contentSamplesPerFace: 2,
      distanceMeasure: "EUCLIDEAN",
      outputStep: 60,
    }),
    ablationExperimentFilename({
      contentSamplesPerFace: 4,
      distanceMeasure: "EUCLIDEAN",
      outputStep: 60,
    }),
    ablationExperimentFilename({
      contentSamplesPerFace: 8,
      distanceMeasure: "SPECTRAL",
      outputStep: 120,
    }),
  ];
  await Promise.all(
    filenames.map((filename) =>
      writeFile(join(matrixDir, filename), validUploadJson),
    ),
  );

  await page
    .getByTestId("pointcloud-ablation-folder-input")
    .setInputFiles(matrixDir);

  await page
    .getByTestId("pointcloud-ablation-fixed-outputStep-select")
    .selectOption("number:60");

  await expect(
    page.getByTestId(
      "pointcloud-ablation-column-contentSamplesPerFace-number:2",
    ),
  ).toBeVisible();
  await expect(
    page.getByTestId(
      "pointcloud-ablation-column-contentSamplesPerFace-number:4",
    ),
  ).toBeVisible();
  await expect(
    page.getByTestId(
      "pointcloud-ablation-column-contentSamplesPerFace-number:8",
    ),
  ).toHaveCount(0);
  await expect(
    page.getByTestId(
      "pointcloud-ablation-row-distanceMeasure-string:EUCLIDEAN",
    ),
  ).toBeVisible();
  await expect(
    page.getByTestId("pointcloud-ablation-row-distanceMeasure-string:SPECTRAL"),
  ).toHaveCount(0);
  await expect(page.getByTestId(ablationCellTestId(8, "SPECTRAL"))).toHaveCount(
    0,
  );
});

test("point-cloud ablation options persist and fall back when unavailable", async ({
  page,
}, testInfo) => {
  await gotoStableApp(page, "/pointcloud-preview");
  await page.getByTestId("pointcloud-ablation-tab").click();

  const stickyDir = testInfo.outputPath("ablation-sticky-options");
  await mkdir(stickyDir, { recursive: true });
  await writeFile(
    join(
      stickyDir,
      ablationExperimentFilename({
        contentSamplesPerFace: 2,
        distanceMeasure: "EUCLIDEAN",
        outputStep: 60,
      }),
    ),
    validUploadJson,
  );
  await writeFile(
    join(
      stickyDir,
      ablationExperimentFilename({
        contentSamplesPerFace: 2,
        distanceMeasure: "SPECTRAL",
        outputStep: 120,
      }),
    ),
    validUploadJson,
  );
  await writeFile(
    join(
      stickyDir,
      ablationExperimentFilename({
        contentSamplesPerFace: 4,
        distanceMeasure: "SPECTRAL",
        outputStep: 120,
      }),
    ),
    validUploadJson,
  );

  await page
    .getByTestId("pointcloud-ablation-folder-input")
    .setInputFiles(stickyDir);
  await expect(
    page.getByTestId("pointcloud-ablation-x-axis-select"),
  ).toHaveValue("contentSamplesPerFace");
  await expect(
    page.getByTestId("pointcloud-ablation-y-axis-select"),
  ).toHaveValue("distanceMeasure");

  await page
    .getByTestId("pointcloud-ablation-y-axis-select")
    .selectOption("outputStep");
  await page
    .getByTestId("pointcloud-ablation-fixed-distanceMeasure-select")
    .selectOption("string:SPECTRAL");

  await page.reload();
  await page.getByTestId("pointcloud-ablation-tab").click();
  await page
    .getByTestId("pointcloud-ablation-folder-input")
    .setInputFiles(stickyDir);
  await expect(
    page.getByTestId("pointcloud-ablation-x-axis-select"),
  ).toHaveValue("contentSamplesPerFace");
  await expect(
    page.getByTestId("pointcloud-ablation-y-axis-select"),
  ).toHaveValue("outputStep");
  await expect(
    page.getByTestId("pointcloud-ablation-fixed-distanceMeasure-select"),
  ).toHaveValue("string:SPECTRAL");

  const fallbackDir = testInfo.outputPath("ablation-sticky-fallback");
  await mkdir(fallbackDir, { recursive: true });
  await writeFile(
    join(
      fallbackDir,
      ablationExperimentFilename({
        contentSamplesPerFace: 2,
        distanceMeasure: "EUCLIDEAN",
        outputStep: null,
      }),
    ),
    validUploadJson,
  );
  await writeFile(
    join(
      fallbackDir,
      ablationExperimentFilename({
        contentSamplesPerFace: 4,
        distanceMeasure: "SPECTRAL",
        outputStep: null,
      }),
    ),
    validUploadJson,
  );

  await page
    .getByTestId("pointcloud-ablation-folder-input")
    .setInputFiles(fallbackDir);
  await expect(
    page.getByTestId("pointcloud-ablation-x-axis-select"),
  ).toHaveValue("contentSamplesPerFace");
  await expect(
    page.getByTestId("pointcloud-ablation-y-axis-select"),
  ).toHaveValue("distanceMeasure");
  await expect(
    page.getByTestId("pointcloud-ablation-fixed-distanceMeasure-select"),
  ).toHaveCount(0);
});

test("point-cloud preview reports upload errors and recovers on a valid upload", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  const fileInput = page.getByTestId("pointcloud-upload-input");
  await fileInput.setInputFiles({
    name: "invalid-pointcloud.json",
    mimeType: "application/json",
    buffer: Buffer.from('{"m_verts":"broken"}', "utf8"),
  });
  await expect(
    page.getByText(/must contain m_verts, m_faces, pc_xyz, and pc_rgb/i),
  ).toBeVisible();
  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("error");

  await fileInput.setInputFiles({
    name: "tiny-upload.json",
    mimeType: "application/json",
    buffer: Buffer.from(validUploadJson, "utf8"),
  });
  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");
  await expect(page.getByTestId("pointcloud-source-label")).toHaveText(
    "tiny-upload.json",
  );
  await expect(page.getByTestId("mesh-vertex-count")).toHaveText("4");
  await expect(page.getByTestId("point-sample-count")).toHaveText("3");
  await expect(
    page.getByText(/must contain m_verts, m_faces, pc_xyz, and pc_rgb/i),
  ).toHaveCount(0);
});

test("point-cloud preview exposes kernel render mode for convolution uploads", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  await expect(page.getByTestId("render-mode-kernels-button")).toHaveCount(0);
  await expect(page.getByTestId("toggle-kernel-directions-button")).toHaveCount(
    0,
  );

  const fileInput = page.getByTestId("pointcloud-upload-input");
  await fileInput.setInputFiles({
    name: "tiny-kernels.json",
    mimeType: "application/json",
    buffer: Buffer.from(convolutionKernelUploadJson, "utf8"),
  });

  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");
  await expect(page.getByTestId("render-mode-kernels-button")).toBeVisible();
  await expect(page.getByTestId("toggle-kernel-directions-button")).toHaveCount(
    0,
  );
  await page.getByTestId("render-mode-kernels-button").click();
  await expect(page.getByTestId("render-mode-kernels-button")).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.getByTestId("kernel-level-select")).toHaveValue("0");
  const directionToggle = page.getByTestId("toggle-kernel-directions-button");
  const directionSelect = page.getByTestId("kernel-direction-index-select");
  const canvas = page
    .getByTestId("pointcloud-preview-canvas")
    .locator("canvas");
  await expect(directionToggle).toHaveAttribute("aria-pressed", "false");
  await expect(directionSelect).toHaveValue("0");
  await expect(directionSelect.locator("option")).toHaveText([
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
  ]);
  await expect(page.getByTestId("mesh-color-mode-status")).toContainText(
    /Kernel preview active.*2 anchors/i,
  );

  const directionsHidden = await canvas.screenshot();
  await directionToggle.click();
  await expect(directionToggle).toHaveAttribute("aria-pressed", "true");
  await expect
    .poll(async () =>
      Buffer.compare(directionsHidden, await canvas.screenshot()),
    )
    .not.toBe(0);

  const directionZero = await canvas.screenshot();
  await directionSelect.selectOption("7");
  await expect(directionSelect).toHaveValue("7");
  await expect
    .poll(async () => Buffer.compare(directionZero, await canvas.screenshot()))
    .not.toBe(0);

  await directionToggle.click();
  await expect(directionToggle).toHaveAttribute("aria-pressed", "false");
  await expect
    .poll(async () =>
      Buffer.compare(directionsHidden, await canvas.screenshot()),
    )
    .toBe(0);

  await page.getByTestId("kernel-level-select").selectOption("1");
  await expect(page.getByTestId("kernel-level-select")).toHaveValue("1");
  await expect(page.getByTestId("mesh-color-mode-status")).toContainText(
    /Kernel preview active.*1 anchors/i,
  );

  await page.getByTestId("render-mode-surface-button").click();
  await expect(page.getByTestId("kernel-level-select")).toHaveCount(0);
  await expect(directionToggle).toHaveCount(0);
  await expect(directionSelect).toHaveCount(0);

  await fileInput.setInputFiles({
    name: "legacy-upload.json",
    mimeType: "application/json",
    buffer: Buffer.from(validUploadJson, "utf8"),
  });
  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");
  await expect(page.getByTestId("render-mode-kernels-button")).toHaveCount(0);
});

test("point-cloud preview queues multiple uploads and lazily switches between them", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  const fileInput = page.getByTestId("pointcloud-upload-input");
  const uploadRows = page.locator('[data-testid^="pointcloud-upload-row-"]');

  await fileInput.setInputFiles([
    {
      name: "tiny-a.json",
      mimeType: "application/json",
      buffer: Buffer.from(validUploadJson, "utf8"),
    },
    {
      name: "tiny-b.json",
      mimeType: "application/json",
      buffer: Buffer.from(alternateUploadJson, "utf8"),
    },
  ]);
  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");
  await expect(page.getByTestId("pointcloud-source-label")).toHaveText(
    "tiny-a.json",
  );
  await expect(page.getByTestId("point-sample-count")).toHaveText("3");
  await expect(uploadRows).toHaveCount(2);
  await expect(page.getByTestId("pointcloud-upload-status-1")).toHaveText(
    "ready",
  );
  await expect(page.getByTestId("pointcloud-upload-status-2")).toHaveText(
    "queued",
  );

  await fileInput.setInputFiles([
    {
      name: "append-upload.json",
      mimeType: "application/json",
      buffer: Buffer.from(appendedUploadJson, "utf8"),
    },
    {
      name: "invalid-queued.json",
      mimeType: "application/json",
      buffer: Buffer.from('{"m_verts":"broken"}', "utf8"),
    },
  ]);
  await expect(page.getByTestId("pointcloud-source-label")).toHaveText(
    "append-upload.json",
  );
  await expect(page.getByTestId("point-sample-count")).toHaveText("7");
  await expect(uploadRows).toHaveCount(4);
  await expect(page.getByTestId("pointcloud-upload-status-4")).toHaveText(
    "queued",
  );

  await page.getByTestId("pointcloud-upload-select-2").click();
  await expect(page.getByTestId("pointcloud-source-label")).toHaveText(
    "tiny-b.json",
  );
  await expect(page.getByTestId("point-sample-count")).toHaveText("5");

  await page.getByTestId("pointcloud-upload-remove-2").click();
  await expect(page.getByTestId("pointcloud-source-label")).toHaveText(
    "append-upload.json",
  );
  await expect(page.getByTestId("point-sample-count")).toHaveText("7");
  await expect(uploadRows).toHaveCount(3);

  await page.getByTestId("pointcloud-upload-select-4").click();
  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("error");
  await expect(page.getByTestId("pointcloud-source-label")).toHaveText(
    "invalid-queued.json",
  );
  await expect(page.getByTestId("pointcloud-upload-status-4")).toHaveText(
    "error",
  );
  await expect(page.getByTestId("pointcloud-upload-row-4")).toBeVisible();
  await expect(
    page.getByText(/must contain m_verts, m_faces, pc_xyz, and pc_rgb/i),
  ).toBeVisible();

  await page.getByTestId("pointcloud-upload-remove-4").click();
  await expect(page.getByTestId("pointcloud-source-label")).toHaveText(
    "append-upload.json",
  );
  await expect(page.getByTestId("point-sample-count")).toHaveText("7");

  await page.getByTestId("pointcloud-upload-remove-3").click();
  await expect(page.getByTestId("pointcloud-source-label")).toHaveText(
    "tiny-a.json",
  );
  await expect(page.getByTestId("point-sample-count")).toHaveText("3");

  await page.getByTestId("pointcloud-upload-remove-1").click();
  await expect(page.getByTestId("pointcloud-source-label")).toHaveText(
    "Bundled medium example",
  );
  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");
  await expect(uploadRows).toHaveCount(0);
});

test("point-cloud preview keeps fragment shading active for larger uploads via spatial hashing", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  const fileInput = page.getByTestId("pointcloud-upload-input");
  await fileInput.setInputFiles({
    name: "large-upload.json",
    mimeType: "application/json",
    buffer: Buffer.from(largeUploadJson, "utf8"),
  });

  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");
  await expect(page.getByTestId("point-sample-count")).toHaveText("513");
  await expect(page.getByTestId("mesh-color-mode-select")).toHaveValue(
    "fragment-knn",
  );
  await expect(page.getByTestId("mesh-color-mode-status")).toContainText(
    /Spatial-hash fragment KNN shading active/i,
  );
});

test("point-cloud preview renders point spheres with point colors", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  await page.getByTestId("pointcloud-upload-input").setInputFiles({
    name: "saturated-points.json",
    mimeType: "application/json",
    buffer: Buffer.from(saturatedPointUploadJson, "utf8"),
  });
  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");

  await page.getByTestId("toggle-mesh-button").click();
  await page.getByTestId("toggle-points-button").click();
  await page.getByTestId("toggle-point-spheres-button").click();
  await page.getByTestId("point-size-slider").fill("0.12");

  const previewCanvas = page.getByTestId("pointcloud-preview-canvas");
  await expect
    .poll(() => readPreviewColorPixelCount(previewCanvas), { timeout: 10_000 })
    .toBeGreaterThan(100);
});

test("point-cloud fragment KNN renders the inverse-squared blend and survives repeated display updates", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1600, height: 900 });
  await gotoStableApp(page, "/pointcloud-preview");

  await page.getByTestId("pointcloud-upload-input").setInputFiles({
    name: "weighted-fragment-knn.json",
    mimeType: "application/json",
    buffer: Buffer.from(weightedFragmentKnnUploadJson, "utf8"),
  });
  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");
  await expect(page.getByTestId("mesh-color-mode-status")).toContainText(
    /Spatial-hash fragment KNN shading active/i,
  );

  await snapCameraToPositiveZ(page);
  const previewCanvas = page.getByTestId("pointcloud-preview-canvas");
  const sample = await fragmentKnnCanvasSample(page, previewCanvas);
  const gammaCheckbox = page.getByLabel("Disable gamma decoding");
  const brightnessSlider = page.getByLabel(/^Brightness:/);
  const expectedColor = (gammaDecodingEnabled: boolean, brightness: number) =>
    expectedFragmentKnnColor({
      position: sample.position,
      points: weightedFragmentKnnPoints,
      colors: weightedFragmentKnnColors,
      gammaDecodingEnabled,
      brightness,
    });

  await expectPreviewPixelColor(
    previewCanvas,
    sample.pixel,
    expectedColor(true, 1),
  );

  await gammaCheckbox.check();
  await expectPreviewPixelColor(
    previewCanvas,
    sample.pixel,
    expectedColor(false, 1),
  );

  await brightnessSlider.fill("0.5");
  await expectPreviewPixelColor(
    previewCanvas,
    sample.pixel,
    expectedColor(false, 0.5),
  );

  await gammaCheckbox.uncheck();
  await expectPreviewPixelColor(
    previewCanvas,
    sample.pixel,
    expectedColor(true, 0.5),
  );

  await brightnessSlider.fill("1.25");
  await expectPreviewPixelColor(
    previewCanvas,
    sample.pixel,
    expectedColor(true, 1.25),
  );

  await gammaCheckbox.check();
  await expectPreviewPixelColor(
    previewCanvas,
    sample.pixel,
    expectedColor(false, 1.25),
  );

  await brightnessSlider.fill("1");
  await gammaCheckbox.uncheck();
  await expectPreviewPixelColor(
    previewCanvas,
    sample.pixel,
    expectedColor(true, 1),
  );
});

test("point-cloud fragment KNN returns the exact sample color on an exact hit", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1600, height: 900 });
  await gotoStableApp(page, "/pointcloud-preview");

  const fileInput = page.getByTestId("pointcloud-upload-input");
  const previewCanvas = page.getByTestId("pointcloud-preview-canvas");
  await expect
    .poll(() => readCameraState(page.getByTestId("camera-state")))
    .not.toBeNull();
  const cameraState = await readCameraState(page.getByTestId("camera-state"));
  if (cameraState === null) {
    throw new Error("Point-cloud preview camera state is unavailable.");
  }
  const cameraDistance = Math.hypot(
    cameraState.position[0] - cameraState.target[0],
    cameraState.position[1] - cameraState.target[1],
    cameraState.position[2] - cameraState.target[2],
  );
  const sample = await fragmentKnnCanvasSample(
    page,
    previewCanvas,
    cameraDistance,
  );
  const exactColor = [0.12, 0.46, 0.78] as const;
  const points = [sample.position, [2, 0, 0], [4, 0, 0]] as const;
  const colors = [exactColor, [1, 0, 0], [0, 1, 0]] as const;
  await fileInput.setInputFiles({
    name: "exact-hit-fragment-knn.json",
    mimeType: "application/json",
    buffer: Buffer.from(
      JSON.stringify({
        pc_xyz: points,
        pc_rgb: colors,
        ...fragmentKnnMesh,
      }),
      "utf8",
    ),
  });
  await expect(page.getByTestId("pointcloud-source-label")).toHaveText(
    "exact-hit-fragment-knn.json",
  );
  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");
  await expect(page.getByTestId("mesh-color-mode-status")).toContainText(
    /Spatial-hash fragment KNN shading active/i,
  );
  await snapCameraToPositiveZ(page);

  await expectPreviewPixelColor(previewCanvas, sample.pixel, exactColor);
});

test("point-cloud fragment KNN keeps translated boundary cells conservative", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1600, height: 900 });
  await gotoStableApp(page, "/pointcloud-preview");

  await page.getByTestId("pointcloud-upload-input").setInputFiles({
    name: "translated-boundary-fragment-knn.json",
    mimeType: "application/json",
    buffer: Buffer.from(translatedFragmentKnnUploadJson, "utf8"),
  });
  await expect(page.getByTestId("pointcloud-source-label")).toHaveText(
    "translated-boundary-fragment-knn.json",
  );
  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");
  await expect(page.getByTestId("point-sample-count")).toHaveText("1170");
  await expect(page.getByTestId("mesh-color-mode-status")).toContainText(
    /1170 points across 147 cells/i,
  );

  await page.getByRole("button", { name: "Reset camera" }).click();
  await expect
    .poll(async () => {
      const state = await readCameraState(page.getByTestId("camera-state"));
      return state === null
        ? Number.POSITIVE_INFINITY
        : Math.abs(state.target[0] - translatedFragmentKnnTarget);
    })
    .toBeLessThanOrEqual(1e-3);
  await snapCameraToPositiveZ(page);

  const previewCanvas = page.getByTestId("pointcloud-preview-canvas");
  const sample = await fragmentKnnCanvasSample(page, previewCanvas);
  expect(
    Math.abs(sample.position[0] - translatedFragmentKnnTarget),
  ).toBeLessThan(0.02);

  const gridDimension = 147;
  const gridExtent = translatedFragmentKnnMax - translatedFragmentKnnMin;
  const exactPointCell = Math.floor(
    ((translatedFragmentKnnTarget - translatedFragmentKnnMin) / gridExtent) *
      gridDimension,
  );
  const shaderCellSize = Math.fround(Math.fround(gridExtent) / gridDimension);
  const shaderQueryCell = Math.floor(
    Math.fround(
      Math.fround(sample.position[0] - translatedFragmentKnnMin) /
        shaderCellSize,
    ),
  );
  expect(exactPointCell).toBe(109);
  expect(shaderQueryCell).toBe(110);

  const unpaddedNeighborMin = Math.fround(
    Math.fround(translatedFragmentKnnMin) +
      Math.fround(Math.fround(exactPointCell) * shaderCellSize),
  );
  const unpaddedNeighborMax = Math.fround(unpaddedNeighborMin + shaderCellSize);
  const unpaddedMinimumSquaredDistance = Math.pow(
    sample.position[0] - unpaddedNeighborMax,
    2,
  );
  const centerCellThirdSquaredDistance =
    Math.pow(translatedFragmentKnnPoints[3]![0] - sample.position[0], 2) +
    Math.pow(sample.position[1], 2) +
    Math.pow(sample.position[2], 2);
  expect(unpaddedMinimumSquaredDistance).toBeGreaterThan(
    centerCellThirdSquaredDistance,
  );

  const expectedColor = expectedFragmentKnnColor({
    position: sample.position,
    points: translatedFragmentKnnPoints,
    colors: translatedFragmentKnnColors,
    gammaDecodingEnabled: true,
    brightness: 1,
  });
  expect(expectedColor[0]).toBeGreaterThan(0.9);

  await expectPreviewPixelColor(previewCanvas, sample.pixel, expectedColor);
});

test("point-cloud preview falls back to baked colours when one spatial-hash cell is too dense", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  const fileInput = page.getByTestId("pointcloud-upload-input");
  await fileInput.setInputFiles({
    name: "dense-cell-upload.json",
    mimeType: "application/json",
    buffer: Buffer.from(denseCellUploadJson, "utf8"),
  });

  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");
  await expect(page.getByTestId("point-sample-count")).toHaveText("300");
  await expect(page.getByTestId("mesh-color-mode-select")).toHaveValue(
    "fragment-knn",
  );
  await expect(page.getByTestId("mesh-color-mode-status")).toContainText(
    /dense cell/i,
  );
});

test("point-cloud preview switches between the available background colours", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  const backgroundSelect = page.getByTestId("background-color-select");
  const previewCanvas = page.getByTestId("pointcloud-preview-canvas");

  await expect(backgroundSelect).toHaveValue("white");
  await expect(backgroundSelect.locator("option")).toHaveText([
    "Default",
    "Black",
    "White",
  ]);
  await expect
    .poll(() => readPreviewBackgroundPixel(previewCanvas))
    .toEqual([255, 255, 255, 255]);

  await backgroundSelect.selectOption("black");
  await expect
    .poll(() => readPreviewBackgroundPixel(previewCanvas))
    .toEqual([0, 0, 0, 255]);

  await backgroundSelect.selectOption("default");
  await expect
    .poll(() => readPreviewBackgroundPixel(previewCanvas))
    .toEqual([9, 17, 31, 255]);
});

test("point-cloud preview toggles the ground plane axis", async ({ page }) => {
  await gotoStableApp(page, "/pointcloud-preview");

  await page.getByTestId("toggle-mesh-button").click();
  await page.getByTestId("toggle-points-button").click();

  const groundPlaneCheckbox = page.getByTestId("ground-plane-axis-checkbox");
  const canvas = page
    .getByTestId("pointcloud-preview-canvas")
    .locator("canvas");

  await expect(groundPlaneCheckbox).not.toBeChecked();
  await groundPlaneCheckbox.check();
  await expect(groundPlaneCheckbox).toBeChecked();
  const visibleGroundPlane = await canvas.screenshot();

  await groundPlaneCheckbox.uncheck();
  await expect(groundPlaneCheckbox).not.toBeChecked();
  await expect
    .poll(async () =>
      Buffer.compare(visibleGroundPlane, await canvas.screenshot()),
    )
    .not.toBe(0);

  await groundPlaneCheckbox.check();
  await expect(groundPlaneCheckbox).toBeChecked();
  await expect
    .poll(async () =>
      Buffer.compare(visibleGroundPlane, await canvas.screenshot()),
    )
    .toBe(0);
});

test("point-cloud preview disables dependent controls and saves viewpoints", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  await expect(page.getByTestId("point-size-slider")).toBeDisabled();
  await expect(page.getByTestId("toggle-point-spheres-button")).toBeDisabled();
  await expect(page.getByTestId("toggle-point-spheres-button")).toHaveAttribute(
    "aria-pressed",
    "false",
  );
  await page.getByTestId("toggle-points-button").click();
  await expect(page.getByTestId("point-size-slider")).toBeEnabled();
  await expect(page.getByTestId("toggle-point-spheres-button")).toBeEnabled();
  await page.getByTestId("toggle-point-spheres-button").click();
  await expect(page.getByTestId("toggle-point-spheres-button")).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await page.getByTestId("toggle-points-button").click();
  await expect(page.getByTestId("point-size-slider")).toBeDisabled();
  await expect(page.getByTestId("toggle-point-spheres-button")).toBeDisabled();

  await page.getByTestId("toggle-mesh-button").click();
  await expect(page.getByTestId("toggle-wireframe-button")).toBeDisabled();
  await expect(page.getByTestId("toggle-solid-mesh-button")).toBeDisabled();
  await expect(page.getByTestId("toggle-solid-mesh-button")).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await page.getByTestId("toggle-mesh-button").click();
  await page.getByTestId("toggle-solid-mesh-button").click();
  await expect(page.getByTestId("toggle-solid-mesh-button")).toHaveAttribute(
    "aria-pressed",
    "false",
  );

  await page.getByTestId("save-viewpoint-button").click();
  await expect(page.getByTestId("viewpoint-go-1")).toBeVisible();
  await page.getByTestId("viewpoint-name-1").fill("Top view");

  await page.getByTestId("swap-yz-button").click();
  await expect(page.getByTestId("swap-yz-button")).toHaveText(/Y\/Z swapped/i);
  await page.getByTestId("viewpoint-update-1").click();
  await page.getByTestId("swap-yz-button").click();
  await expect(page.getByTestId("swap-yz-button")).toHaveText(/Flip Y and Z/i);
  await page.getByTestId("viewpoint-go-1").click();
  await expect(page.getByTestId("swap-yz-button")).toHaveText(/Y\/Z swapped/i);

  const fileInput = page.getByTestId("pointcloud-upload-input");
  await fileInput.setInputFiles({
    name: "tiny-upload.json",
    mimeType: "application/json",
    buffer: Buffer.from(validUploadJson, "utf8"),
  });
  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");
  await expect(page.getByTestId("viewpoint-go-1")).toBeVisible();
  await expect(page.getByTestId("viewpoint-name-1")).toHaveValue("Top view");

  await page.reload();
  await expect(page.getByTestId("viewpoint-go-1")).toBeVisible();
  await expect(page.getByTestId("viewpoint-name-1")).toHaveValue("Top view");

  await page.getByTestId("viewpoint-delete-1").click();
  await expect(page.getByTestId("viewpoint-go-1")).toHaveCount(0);

  await page.reload();
  await expect(page.getByTestId("viewpoint-go-1")).toHaveCount(0);
});

test("point-cloud preview shows and copies saved viewpoint camera details", async ({
  context,
  page,
}) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"], {
    origin: "http://127.0.0.1:4173",
  });
  await gotoStableApp(page, "/pointcloud-preview");
  await expect(page.getByTestId("save-viewpoint-button")).toBeEnabled();

  const cameraState = JSON.parse(
    (await page.getByTestId("camera-state").textContent()) ?? "null",
  ) as {
    position: [number, number, number];
    target: [number, number, number];
  };
  const formatTuple = (tuple: [number, number, number]): string =>
    `(${tuple.map((value) => value.toFixed(1)).join(", ")})`;
  const expectedCameraText =
    `position = ${formatTuple(cameraState.position)}\n` +
    `focal_point = ${formatTuple(cameraState.target)}`;

  await page.getByTestId("save-viewpoint-button").click();
  const infoButton = page.getByTestId("viewpoint-info-1");
  await expect(infoButton).toHaveAttribute("aria-expanded", "false");

  await infoButton.click();
  await expect(infoButton).toHaveAttribute("aria-expanded", "true");
  await expect(page.getByTestId("viewpoint-camera-details-1")).toHaveText(
    expectedCameraText,
  );

  await page.getByTestId("viewpoint-copy-1").click();
  await expect(page.getByTestId("viewpoint-copy-1")).toContainText("Copied");
  await expect
    .poll(() => page.evaluate(() => navigator.clipboard.readText()))
    .toBe(expectedCameraText);

  await page.getByText("Saved viewpoints", { exact: true }).click();
  await expect(page.getByTestId("viewpoint-info-popover-1")).toHaveCount(0);
  await expect(infoButton).toHaveAttribute("aria-expanded", "false");
});

test("point-cloud preview can download a screenshot", async ({ page }) => {
  await gotoStableApp(page, "/pointcloud-preview");

  const fileInput = page.getByTestId("pointcloud-upload-input");
  await fileInput.setInputFiles({
    name: "tiny-upload.json",
    mimeType: "application/json",
    buffer: Buffer.from(validUploadJson, "utf8"),
  });
  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");

  const downloadPromise = page.waitForEvent("download");
  await page.getByTestId("screenshot-button").click();
  const download = await downloadPromise;

  expect(download.suggestedFilename()).toMatch(
    /^tiny-upload-screenshot-\d{8}-\d{6}\.png$/,
  );
});

test("point-cloud preview disables batch screenshots without saved viewpoints", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  await page.getByTestId("pointcloud-upload-input").setInputFiles([
    {
      name: "tiny-a.json",
      mimeType: "application/json",
      buffer: Buffer.from(validUploadJson, "utf8"),
    },
    {
      name: "tiny-b.json",
      mimeType: "application/json",
      buffer: Buffer.from(alternateUploadJson, "utf8"),
    },
  ]);
  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");

  await page.getByTestId("batch-screenshot-button").click();
  await expect(page.getByTestId("batch-screenshot-modal")).toBeVisible();
  await expect(page.getByTestId("batch-screenshot-empty")).toBeVisible();
  await expect(page.getByTestId("batch-screenshot-download")).toBeDisabled();
});

test("point-cloud preview stacks panels and keeps batch actions clickable on narrow screens", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await gotoStableApp(page, "/pointcloud-preview");

  await page.getByTestId("pointcloud-upload-input").setInputFiles([
    {
      name: "tiny-a.json",
      mimeType: "application/json",
      buffer: Buffer.from(validUploadJson, "utf8"),
    },
    {
      name: "tiny-b.json",
      mimeType: "application/json",
      buffer: Buffer.from(alternateUploadJson, "utf8"),
    },
  ]);
  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");

  const dataPanelBox = await page
    .getByTestId("pointcloud-left-panel")
    .boundingBox();
  const previewBox = await page
    .getByTestId("pointcloud-preview-canvas")
    .boundingBox();
  const controlsBox = await page
    .getByTestId("pointcloud-right-panel")
    .boundingBox();
  expect(dataPanelBox).not.toBeNull();
  expect(previewBox).not.toBeNull();
  expect(controlsBox).not.toBeNull();
  expect(previewBox!.y).toBeGreaterThan(dataPanelBox!.y + dataPanelBox!.height);
  expect(controlsBox!.y).toBeGreaterThan(previewBox!.y + previewBox!.height);

  await page.getByTestId("batch-screenshot-button").click();
  await expect(page.getByTestId("batch-screenshot-modal")).toBeVisible();
});

test("point-cloud preview downloads every mesh and selected viewpoint in one ZIP", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  await page.getByTestId("save-viewpoint-button").click();
  await page.getByTestId("viewpoint-name-1").fill("Front view");
  await page.getByTestId("swap-yz-button").click();
  await page.getByTestId("save-viewpoint-button").click();
  await page.getByTestId("viewpoint-name-2").fill("Swapped view");
  await page.getByTestId("swap-yz-button").click();

  await page.getByTestId("pointcloud-upload-input").setInputFiles([
    {
      name: "duplicate.json",
      mimeType: "application/json",
      buffer: Buffer.from(validUploadJson, "utf8"),
    },
    {
      name: "duplicate.json",
      mimeType: "application/json",
      buffer: Buffer.from(alternateUploadJson, "utf8"),
    },
  ]);
  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");
  await expect(page.getByTestId("pointcloud-source-label")).toHaveText(
    "duplicate.json",
  );
  const cameraStateBeforeBatch = await snapCameraToPositiveX(
    page.getByTestId("camera-state"),
    page.getByTestId("snap-axis-pos-x"),
  );

  await page.getByTestId("batch-screenshot-button").click();
  await expect(page.getByTestId("batch-screenshot-download")).toBeDisabled();
  await page.getByTestId("batch-screenshot-select-all").click();
  await expect(page.getByTestId("batch-screenshot-download")).toHaveText(
    "Download 4 screenshots as ZIP",
  );

  const downloadPromise = page.waitForEvent("download");
  await page.getByTestId("batch-screenshot-download").click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(
    /^pointcloud-batch-screenshots-\d{8}-\d{6}\.zip$/,
  );

  const downloadPath = await download.path();
  expect(downloadPath).not.toBeNull();
  const archive = await readFile(downloadPath!);
  const zipEntries = readStoredZipEntries(archive);
  expect(zipEntries.size).toBe(4);
  const entryNames = [...zipEntries.keys()];
  expect(entryNames).toEqual(
    expect.arrayContaining([
      expect.stringMatching(
        /^duplicate-upload-1-view-1-front-view-\d{8}-\d{6}\.png$/,
      ),
      expect.stringMatching(
        /^duplicate-upload-1-view-2-swapped-view-\d{8}-\d{6}\.png$/,
      ),
      expect.stringMatching(
        /^duplicate-upload-2-view-1-front-view-\d{8}-\d{6}\.png$/,
      ),
      expect.stringMatching(
        /^duplicate-upload-2-view-2-swapped-view-\d{8}-\d{6}\.png$/,
      ),
    ]),
  );
  for (const png of zipEntries.values()) {
    expect(png.subarray(0, 8)).toEqual(
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    );
  }

  await expect(page.getByTestId("batch-screenshot-modal")).toHaveCount(0);
  await expect(page.getByTestId("pointcloud-source-label")).toHaveText(
    "duplicate.json",
  );
  await expect(page.getByTestId("swap-yz-button")).toHaveText(/Flip Y and Z/i);
  await expect(page.getByTestId("camera-state")).toHaveText(
    cameraStateBeforeBatch!,
  );
});

test("point-cloud batch screenshots restore kernel render settings", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  await page.getByTestId("pointcloud-upload-input").setInputFiles([
    {
      name: "kernels-a.json",
      mimeType: "application/json",
      buffer: Buffer.from(convolutionKernelUploadJson, "utf8"),
    },
    {
      name: "kernels-b.json",
      mimeType: "application/json",
      buffer: Buffer.from(convolutionKernelUploadJson, "utf8"),
    },
  ]);
  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");
  await page.getByTestId("render-mode-kernels-button").click();
  await page.getByTestId("kernel-level-select").selectOption("1");
  await expect(page.getByTestId("mesh-color-mode-status")).toContainText(
    /Kernel preview active.*1 anchors/i,
  );

  await page.getByTestId("save-viewpoint-button").click();
  await page.getByTestId("viewpoint-name-1").fill("Kernel view");
  await page.getByTestId("batch-screenshot-button").click();
  await page.getByTestId("batch-screenshot-select-all").click();

  const downloadPromise = page.waitForEvent("download");
  await page.getByTestId("batch-screenshot-download").click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(
    /^pointcloud-batch-screenshots-\d{8}-\d{6}\.zip$/,
  );

  await expect(page.getByTestId("batch-screenshot-modal")).toHaveCount(0);
  await expect(page.getByTestId("pointcloud-source-label")).toHaveText(
    "kernels-a.json",
  );
  await expect(page.getByTestId("render-mode-kernels-button")).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.getByTestId("kernel-level-select")).toHaveValue("1");
  await expect(page.getByTestId("mesh-color-mode-status")).toContainText(
    /Kernel preview active.*1 anchors/i,
  );
});

test("point-cloud batch screenshot reports malformed queued meshes and restores the preview", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  await page.getByTestId("swap-yz-button").click();
  await page.getByTestId("save-viewpoint-button").click();
  await page.getByTestId("viewpoint-name-1").fill("Swapped view");

  await page.getByTestId("pointcloud-upload-input").setInputFiles([
    {
      name: "valid.json",
      mimeType: "application/json",
      buffer: Buffer.from(validUploadJson, "utf8"),
    },
    {
      name: "broken.json",
      mimeType: "application/json",
      buffer: Buffer.from('{"m_verts":"broken"}', "utf8"),
    },
  ]);
  await expect(page.getByTestId("pointcloud-source-label")).toHaveText(
    "valid.json",
  );
  await expect(page.getByTestId("swap-yz-button")).toHaveText(/Y\/Z swapped/i);
  const cameraStateBeforeBatch = await snapCameraToPositiveX(
    page.getByTestId("camera-state"),
    page.getByTestId("snap-axis-pos-x"),
  );

  await page.getByTestId("batch-screenshot-button").click();
  await page.getByTestId("batch-viewpoint-1").check();
  await page.getByTestId("batch-screenshot-download").click();

  await expect(page.getByTestId("batch-screenshot-error")).toContainText(
    /broken\.json.*must contain m_verts, m_faces, pc_xyz, and pc_rgb/i,
  );
  await expect(page.getByTestId("pointcloud-source-label")).toHaveText(
    "valid.json",
  );
  await expect(page.getByTestId("pointcloud-upload-status-2")).toHaveText(
    "error",
  );
  await expect(page.getByTestId("swap-yz-button")).toHaveText(/Y\/Z swapped/i);
  await expect(page.getByTestId("camera-state")).toHaveText(
    cameraStateBeforeBatch!,
  );
  await expect(page.getByTestId("batch-screenshot-download")).toBeEnabled();
});

test("point-cloud preview right-side cards can collapse and expand", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  const cameraCardToggle = page
    .getByTestId("camera-card")
    .getByRole("button", { name: /Camera and orientation/i });
  await expect(cameraCardToggle).toHaveAttribute("aria-expanded", "true");
  await expect(page.getByTestId("save-viewpoint-button")).toBeVisible();

  await cameraCardToggle.click();
  await expect(cameraCardToggle).toHaveAttribute("aria-expanded", "false");
  await expect(page.getByTestId("save-viewpoint-button")).toBeHidden();

  await cameraCardToggle.click();
  await expect(cameraCardToggle).toHaveAttribute("aria-expanded", "true");
  await expect(page.getByTestId("save-viewpoint-button")).toBeVisible();
});
