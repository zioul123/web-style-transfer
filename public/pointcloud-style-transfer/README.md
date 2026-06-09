# Point-cloud preview examples

This directory stores committed example exports for the browser-side
`/pointcloud-preview` route.

## Files

- `pc-and-mesh-tiny-example.json`: the smallest deterministic example used by
  parser and interpolation tests.
- `pc-and-mesh-medium-example.json`: the default demo loaded by the preview
  route in the browser.

The browser route loads these files through `assetUrl()` so the paths continue
to work under the repository's Vite `BASE_URL` and GitHub Pages deployment.

## JSON format

Each file uses the shared mesh-plus-point-cloud JSON shape consumed by
`src/features/pointcloud-preview/loadPointCloudMesh.ts` and mirrored by
`python-reference/pointcloud-style-transfer/data-storage.py`:

```json
{
  "m_verts": [[0, 0, 0]],
  "m_faces": [[0, 0, 0]],
  "pc_xyz": [[0, 0, 0]],
  "pc_rgb": [[1, 1, 1]]
}
```

Rules:

- `m_verts` contains mesh vertex positions as `[x, y, z]`.
- `m_faces` contains triangle indices into `m_verts`.
- `pc_xyz` contains aligned point-sample positions in the same coordinate space
  as the mesh.
- `pc_rgb` contains per-point colours in `[r, g, b]` float form, one entry per
  `pc_xyz` sample.

The preview route builds both exact CPU-side 3-nearest-neighbour inspection
state and a spatial-hash layout for fragment shading from this data without
changing the on-disk JSON schema.
