import { expect, test } from "@playwright/test";
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
