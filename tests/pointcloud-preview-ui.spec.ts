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

test("point-cloud preview boots from the standalone route with the bundled demo", async ({
  page,
}) => {
  await gotoStableApp(page, "/pointcloud-preview");

  await expect(
    page.getByRole("heading", {
      name: /Browser preview for mesh-aligned point clouds/i,
    }),
  ).toBeVisible();
  await expect(page.getByTestId("pointcloud-preview-canvas")).toBeVisible();
  await expect(page.getByTestId("pointcloud-source-label")).toHaveText(
    "Bundled tiny example",
  );
  await expect(page.getByTestId("pointcloud-load-status")).toHaveText("ready");
  await expect(page.getByTestId("mesh-vertex-count")).toHaveText("4");
  await expect(page.getByTestId("mesh-face-count")).toHaveText("2");
  await expect(page.getByTestId("point-sample-count")).toHaveText("3");
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
