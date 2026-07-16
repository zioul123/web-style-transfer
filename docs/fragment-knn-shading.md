# Fragment KNN Mesh Shading

This document is the technical reference for fragment-space point-cloud
colouring in the standalone point-cloud preview. It describes both the original
implementation and the optimized implementation now owned by
`buildFragmentKnnFragmentShader` in
`src/features/pointcloud-preview/PointCloudPreviewScene.tsx`.

For the chronological account of the optimization work, see
[Optimizing fragment-space KNN shading](blog/optimizing-fragment-knn-shading.md).

## What the shader does

The point-cloud preview can colour a triangle mesh in two ways:

- **Baked vertex colours** compute a colour for each mesh vertex on the CPU and
  let rasterization interpolate those colours across each triangle.
- **Fragment KNN colours** compute a colour independently at every rendered
  fragment from nearby point-cloud samples.

Fragment-space shading preserves local detail that can disappear when a large
triangle has only three precomputed vertex colours. For a fragment at position
`q`, the shader selects up to three point samples and computes an
inverse-squared-distance blend:

```text
weight_i = 1 / max(squaredDistance(q, point_i), 1e-16)
colour   = sum(displayColour_i * weight_i) / sum(weight_i)
```

If the nearest squared distance is at most `1e-16`, the shader returns that
sample's display colour directly. This avoids an unstable division and makes an
exact sample hit visually exact.

## Important semantic boundary

The renderer performs a **local spatial-hash search**, not exact global KNN. It
searches the query cell and the immediately adjacent cells: a fixed `3 x 3 x 3`
domain of at most 27 cells. This is the same domain used before and after the
optimization.

The hover inspector has a different job. It uses the CPU KD-tree to return an
exact global three-neighbour result for inspection. Keeping that distinction
explicit avoids treating the renderer's bounded approximation as a general
KNN implementation.

Equal-distance points may be selected in a different order because the
optimized shader visits the centre cell first. Non-tied results, the search
domain, display conversion, exact-hit rule, and inverse-squared weighting are
preserved.

## Data preparation

`buildSpatialHashGrid3d` in
`src/features/pointcloud-preview/math/spatialHash3d.ts` prepares the data on the
CPU:

1. Compute the point-cloud bounds and active axes.
2. Choose grid dimensions targeting approximately eight points per cell.
3. Count the points assigned to each cell.
4. Convert those counts into prefix offsets.
5. Reorder positions and colours so every cell owns one contiguous range.

The reordered position and colour arrays remain aligned. If cell `c` has start
`s` and count `n`, its points are stored at `[s, s + n)` in both arrays.

The preview uploads three textures:

| Texture         | Format    | Per-entry contents           |
| --------------- | --------- | ---------------------------- |
| Point positions | `RGBA32F` | `xyz` position plus padding  |
| Point colours   | `RGBA32F` | `rgb` colour plus padding    |
| Cell ranges     | `RG32UI`  | unsigned `start` and `count` |

Texture rows are capped at 2,048 texels. A one-row fast path handles the common
case without integer division; larger arrays use integer row and column
calculation.

The shader's inner point loop still needs a compile-time upper bound. If the
spatial hash reports more than 256 points in any cell, the UI keeps the selected
mode visible but safely renders the baked-colour fallback instead.

## The original implementation

The original shader was a deliberately direct implementation of the algorithm:

1. Convert the fragment position into a clamped grid coordinate.
2. Visit the surrounding cells with three nested loops over `x`, `y`, and `z`.
3. Read two entries from a floating-point cell-offset texture to recover each
   cell's start and end.
4. For every point in every valid cell:
   - read its position;
   - read its colour;
   - calculate squared distance;
   - apply gamma and brightness display conversion;
   - insertion-sort the distance and converted colour into the best three.
5. Blend the three retained colours.

Each texture read used normalized UV coordinates. Converting a linear index
into UVs required floating-point division, `floor`, and normalization by the
texture width and height.

### Why it was written that way

The design was reasonable for a correctness-first first version:

- A spatial hash reduced the problem from scanning the entire point cloud per
  fragment to scanning a small, bounded neighbourhood.
- Textures were the simplest Three.js-compatible way to make variable-length
  point data available in a fragment shader.
- Fixed loop bounds were predictable for GLSL compilation and matched the
  existing dense-cell safety cap.
- Carrying colours alongside distances made the top-three insertion code easy
  to understand and kept the final blend small.
- Recreating a material from React dependencies guaranteed that control values
  reached the shader without requiring a separate uniform synchronization
  path.

That version established the visual behaviour and fallback rules with a small
amount of route-local code. Its problem was not the high-level algorithm; it
was the amount of unnecessary work performed inside the hottest loop.

## Original cost profile

Let `F` be the number of covered mesh fragments and `C` the number of candidate
points in the valid neighbouring cells. The main search cost is proportional to
`F * C`.

The original implementation paid, for every candidate:

- one position texture fetch;
- one colour texture fetch;
- linear-index-to-normalized-UV arithmetic for each fetch;
- a distance calculation;
- display colour conversion, including the potentially expensive `pow` path;
- top-three insertion comparisons and movement of both distance and colour.

It also read two offset texels per visited cell. Outside the shader, brightness,
gamma, and wireframe changes recreated the `ShaderMaterial`; the broad cleanup
effect could then dispose textures that the replacement material still reused.

## The optimized search

The current shader separates **finding neighbours** from **shading with their
colours**.

### 1. Integer texel access

The material now uses GLSL 3 through WebGL2. `texelFetch` addresses exact texels
with integer coordinates, so lookups do not construct normalized UVs. Cell
metadata is one unsigned `start/count` texel rather than two floating-point
prefix-offset texels.

### 2. Distance and index only in the candidate loop

The candidate loop fetches a position, computes its distance, and insertion
sorts only the point index and squared distance. Colour textures and display
conversion are untouched until the final neighbours are known.

After the search, the shader fetches and adjusts at most three colours. This is
the largest structural saving because rejected candidates no longer pay for a
colour read or gamma conversion.

### 3. Centre-first traversal

The 27 offsets are stored in a fixed array ordered from the query cell outward:
centre, face neighbours, edge neighbours, then corner neighbours. Processing
the centre first usually establishes a useful third-nearest distance early.

### 4. Conservative cell pruning

Once three neighbours have been found, the shader computes the minimum possible
squared distance from the fragment to a candidate cell's axis-aligned bounding
box. If that lower bound is greater than the current third-nearest squared
distance, no point in the cell can enter the result and the cell is skipped.

```text
if minimumDistanceSquared(fragment, cellAabb) > thirdNearestDistanceSquared:
    skip cell
```

The comparison is strictly `>` rather than `>=`, so an equal-distance point is
not rejected by the bound.

This pruning does not reduce the 27-cell domain. It proves at runtime that a
particular cell cannot affect the answer already found.

### 5. Float-safe cell bounds

CPU cell assignment uses JavaScript number arithmetic while GLSL reconstructs
cell boundaries from 32-bit float uniforms. At large translated coordinates,
rounding can put a stored point just outside the shader's reconstructed cell
box. An unpadded AABB lower bound could then incorrectly prune the point.

`conservativeCellBoundary` expands each reconstructed boundary outward using a
scale-aware IEEE-754 error bound. The bound accounts for conversion of the grid
minimum, cell size, and integer coordinate, followed by multiplication,
addition, and padding evaluation. It uses float unit roundoff
`u = 2^-24` and the inverse-rounding factor `u / (1 - u)`.

The expansion is intentionally conservative: a slightly larger box may miss a
pruning opportunity, but it must not falsely prove that a real candidate is too
far away.

### 6. Degenerate-grid fast path

If all grid dimensions are one, only the centre cell is valid. The shader visits
that cell once and avoids 26 guaranteed-invalid iterations and their AABB work.

## Material and resource lifecycle

Shader arithmetic was only part of the cost. The original material depended on
brightness, gamma, and wireframe React values, so routine control changes
recompiled or replaced material state.

The current scene:

- creates textures when the dataset or colour mode changes;
- creates the material when its texture/grid structure changes;
- updates brightness and gamma uniforms in place;
- mutates wireframe state in place;
- disposes each geometry, texture, and material in an effect scoped to that
  resource's identity;
- avoids rebuilding baked mesh colours for brightness changes while fragment
  mode is active.

This keeps hot UI updates away from shader compilation and makes resource
ownership explicit.

## Correctness and performance evidence

The browser tests in `tests/pointcloud-preview-ui.spec.ts` cover:

- a deterministic inverse-squared three-colour blend;
- repeated gamma and brightness updates;
- the exact-hit colour path;
- a 1,170-point translated multi-cell case that reproduces CPU/GPU boundary
  disagreement and validates conservative pruning;
- fragment mode on an existing larger upload;
- baked fallback when a cell exceeds the 256-point cap.

The optional benchmark in
`benchmarks/pointcloud-fragment-knn-performance.spec.ts` uses a fixed 1,600 by
900 browser viewport and approximately 700 by 606 canvas. It warms each mode,
records three 12-frame trials, synchronizes with `gl.finish()`, and reports
median and p95 frame intervals for a regular grid and a dense single-cell case.

The recorded implementation run produced:

| Workload                             | Original median / p95 | Optimized median range | Optimized p95 range |
| ------------------------------------ | --------------------: | ---------------------: | ------------------: |
| 1,728-point regular grid, max 8/cell |       44.25 / 63.7 ms |         16.15-19.55 ms |        17.7-22.0 ms |
| 128 coincident points in one cell    |        49.5 / 61.8 ms |          29.8-35.95 ms |        30.9-38.9 ms |

These figures are observational, not a portable performance guarantee. The
benchmark includes browser scheduling and the configured Chromium/SwiftShader
environment, and it is sensitive to thermal and host load. Its durable value is
the fixed workload and repeatable method; physical-GPU and cross-browser
measurements remain useful follow-up work.

## Files to read next

- `src/features/pointcloud-preview/PointCloudPreviewScene.tsx`: shader,
  textures, material, uniforms, and cleanup.
- `src/features/pointcloud-preview/math/spatialHash3d.ts`: CPU grid building and
  sorted cell ranges.
- `src/features/pointcloud-preview/math/interpolation.ts`: CPU interpolation
  semantics.
- `tests/pointcloud-preview-data.spec.ts`: spatial-hash tuple preservation and
  CPU interpolation coverage.
- `tests/pointcloud-preview-ui.spec.ts`: rendered-pixel correctness and
  fallback coverage.
- `benchmarks/pointcloud-fragment-knn-performance.spec.ts`: repeatable timing
  harness.
