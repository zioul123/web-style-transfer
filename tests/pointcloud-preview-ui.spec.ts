import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";
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

test("point-cloud preview reports upload errors and recovers on a valid upload", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  const fileInput = page.locator('input[type="file"]');
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

test("point-cloud preview queues multiple uploads and lazily switches between them", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  const fileInput = page.locator('input[type="file"]');
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

  const fileInput = page.locator('input[type="file"]');
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

test("point-cloud preview falls back to baked colours when one spatial-hash cell is too dense", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  const fileInput = page.locator('input[type="file"]');
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

test("point-cloud preview disables dependent controls and saves viewpoints", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  await page.getByTestId("toggle-points-button").click();
  await expect(page.getByTestId("point-size-slider")).toBeDisabled();

  await page.getByTestId("toggle-mesh-button").click();
  await expect(page.getByTestId("toggle-wireframe-button")).toBeDisabled();

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

  const fileInput = page.locator('input[type="file"]');
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

test("point-cloud preview can download a screenshot", async ({ page }) => {
  await gotoStableApp(page, "/pointcloud-preview");

  const fileInput = page.locator('input[type="file"]');
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

  await page.locator('input[type="file"]').setInputFiles([
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

  await page.locator('input[type="file"]').setInputFiles([
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
  const cameraStateBeforeSnap = await page
    .getByTestId("camera-state")
    .textContent();
  await page.getByTestId("snap-axis-pos-x").click();
  await expect
    .poll(() => page.getByTestId("camera-state").textContent())
    .not.toBe(cameraStateBeforeSnap);
  const cameraStateBeforeBatch = await page
    .getByTestId("camera-state")
    .textContent();
  expect(cameraStateBeforeBatch).not.toBe("unavailable");

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

test("point-cloud batch screenshot reports malformed queued meshes and restores the preview", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  await page.getByTestId("swap-yz-button").click();
  await page.getByTestId("save-viewpoint-button").click();
  await page.getByTestId("viewpoint-name-1").fill("Swapped view");

  await page.locator('input[type="file"]').setInputFiles([
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
  const cameraStateBeforeSnap = await page
    .getByTestId("camera-state")
    .textContent();
  await page.getByTestId("snap-axis-pos-x").click();
  await expect
    .poll(() => page.getByTestId("camera-state").textContent())
    .not.toBe(cameraStateBeforeSnap);
  const cameraStateBeforeBatch = await page
    .getByTestId("camera-state")
    .textContent();
  expect(cameraStateBeforeBatch).not.toBe("unavailable");

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
