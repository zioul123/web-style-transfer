# Web Style Transfer Architecture Overview

This document describes the current module boundaries and runtime data flow for the WebGPU style-transfer app.

## System shape

The app has three main layers:

1. **Main-thread UI and orchestration**
   - React app shell in `src/App.tsx`.
   - Style-transfer controller, model-pack loading, cache status, and image/tensor conversion in `src/features/style-transfer/`.
   - Image benchmark route in `src/BenchmarkApp.tsx`.
   - Point-cloud preview route in `src/PointCloudPreviewApp.tsx` and `src/features/pointcloud-preview/`.
2. **Typed worker protocol**
   - Public type exports through `src/types.ts`.
   - Split protocol definitions in `src/types/worker-protocol/`.
3. **Worker-side WebGPU compute**
   - Worker entrypoint in `src/styleTransfer.worker.ts`.
   - Message routing in `src/ml/worker/main-thread-protocol/`.
   - Runtime helpers, kernels, and optimization pipelines under `src/ml/worker/`.

The project is intentionally message-driven. The main thread posts a typed request, the worker routes it by discriminant, compute runs on the worker-owned WebGPU device, and the worker posts back a typed response.

## Main-thread app

### `src/App.tsx`

`App.tsx` is now mostly presentation:

- Renders content/style/output cards.
- Renders run controls and advanced options.
- Displays worker/GPU/model-cache status.
- Delegates state and side effects to `useStyleTransferController`.
- Switches model-pack options based on whether the app is running locally or from hosted assets.

It still owns UI-specific labels and some option presentation helpers. That is appropriate for now.

### `src/features/style-transfer/hooks/useStyleTransferController.ts`

The controller hook owns most main-thread orchestration:

- Creates and manages the style-transfer worker.
- Pings the worker and initializes WebGPU.
- Loads default images and uploaded images.
- Converts DOM images to NCHW float tensors and output tensors back to PNG data URLs.
- Chooses resolution presets, including automatic portrait/landscape matching.
- Loads selected VGG19 model packs and tracks IndexedDB cache status.
- Builds `run-style-transfer` requests.
- Maintains chunked run state, optimizer session IDs, losses, iteration count, timing text, and preview output.
- Clears worker-side sessions and model cache when relevant controls change.

This is the main seam for future UI/component extraction.

### `src/features/style-transfer/modelPacks.ts` and `modelCache.ts`

`modelPacks.ts` resolves manifest/shard URLs, parses VGG19 model packs into the worker payload shape, and writes successful downloads into IndexedDB.

`modelCache.ts` wraps IndexedDB operations for model-pack records and exposes cache status/clear helpers. If IndexedDB is unavailable, cache calls degrade to empty/no-op results.

### `src/BenchmarkApp.tsx`

The benchmark route is selected when the path after the Vite base starts with `/benchmark`. It provides:

- Kernel-lab variants for experimental optimization flags and speed comparisons.
- Model-pack comparison and acceptance helpers.
- First-pool optimization benchmarks.

Some benchmark flows require optional fixtures or model packs that are not committed by default.

### `src/PointCloudPreviewApp.tsx` and `src/features/pointcloud-preview/`

The point-cloud preview route is selected when the path after the Vite base
starts with `/pointcloud-preview`. It is intentionally isolated from the worker
protocol and the WebGPU style-transfer pipeline so point-cloud inspection does
not couple preview-only behavior back into the optimization stack.

The route currently provides:

- loading of committed demo assets or a session-local queue of uploaded
  mesh-plus-point-cloud JSON exports, with inactive uploads kept as `File`
  references until selected;
- JSON validation, typed-array conversion, bounds calculation, and precomputed
  baked vertex colours in `loadPointCloudMesh.ts`;
- exact CPU-side 3-nearest-neighbour hit inspection via the shared
  `src/ml/geometry/kdTree3d.ts` helper and feature-local
  `math/interpolation.ts`;
- fragment-space mesh colouring backed by a feature-local spatial hash in
  `math/spatialHash3d.ts`, with fallback to baked vertex colours when a dense
  cell would exceed current shader bounds;
- browser-only view state such as mesh/point toggles, gamma and brightness
  controls, screenshots, and saved viewpoints persisted in local storage across
  datasets on this route.

`PointCloudPreviewScene.tsx` owns the R3F canvas, point/mesh materials,
fragment-shader textures, hit overlays, camera commands, and FPS sampling.
`PointCloudPreviewApp.tsx` composes focused route-local controllers and
presentation components:

- `usePointCloudPreviewController.ts` owns view settings, hit selection, FPS,
  and camera-command state;
- `usePointCloudAssetsController.ts` owns bundled and uploaded asset loading,
  the lazy `File` queue, and stale-request protection;
- `useSavedViewpointsController.ts` owns saved-camera persistence and
  mutations;
- `usePointCloudScreenshotsController.ts` owns current-canvas downloads and
  synchronized batch capture/restoration;
- the `PointCloudPreview*Panel.tsx`, `PointCloudPreviewViewport.tsx`, and modal
  components own the route presentation.

## Worker protocol

The public import surface is `src/types.ts`, which re-exports the split protocol files in `src/types/worker-protocol/`.

The protocol is organized as:

- `core.ts`: tensor shapes, feature-matrix and point-cloud metadata payloads,
  operands, and tensor response payloads.
- `tensor-ops.ts`: op-level request variants used by parity tests and
  debugging, including readback-heavy point-cloud feature, convolution, and
  surface-pool routes.
- `pipelines.ts`: first-pool/full-style-transfer pipeline requests, optimizer flags, and run stats.
- `messages.ts`: full worker request/response unions.

Important message families:

- Infrastructure: `ping`, `init-webgpu`, `tensor-roundtrip`.
- Tensor ops: primitive, VGG-layer, loss, and backward parity routes.
- Pipelines: `run-first-pool-optimizer`, `run-style-transfer`, `clear-style-transfer-session`.

The discriminated unions keep request validation explicit and make Playwright tests able to exercise individual operations without running the full app.

## Worker bootstrap and routing

### `src/styleTransfer.worker.ts`

The worker entrypoint is intentionally tiny. It imports and mounts the message router.

### `src/ml/worker/main-thread-protocol/messageRouter.ts`

The message router handles high-level request dispatch:

- `ping`
- `init-webgpu`
- `tensor-roundtrip`
- pipeline requests
- style-transfer session clearing
- delegation of all `tensor-op` requests to `tensorOpRouter`

Pipeline errors are caught and converted into typed failed responses.

### `src/ml/worker/main-thread-protocol/tensorOpRouter.ts`

The tensor-op router owns operation-level dispatch for tests and debug routes. It converts protocol tensors/operands into runtime tensors or buffers, invokes the relevant op runner, and returns JSON-friendly arrays/scalars.

These routes are intentionally readback-heavy because they are correctness/debug endpoints. The full optimization pipelines use GPU-resident buffers more aggressively.

### Response and initialization helpers

- `responses.ts` centralizes typed worker response posting.
- `initWebGpu.ts` obtains the adapter/device and resets runtime caches when needed.

## Runtime infrastructure

Runtime helpers live in `src/ml/worker/runtime/`:

- `deviceState.ts`: worker-global WebGPU device state.
- `bufferKernels.ts`: upload, readback, owned-buffer helpers, and reusable buffer utilities.
- `bufferPool.ts`: reusable GPU buffer pool for allocation-sensitive paths.
- `computeContext.ts`: shared compute helpers such as unary dispatch.
- `computePipelineCache.ts`: cached compute pipeline creation.
- `gpuFlags.ts`: common WebGPU usage/map flag constants.
- `operands.ts`: protocol operand parsing.
- `optimizationContext.ts`: per-run optimization runtime context and cleanup lifecycle.
- `shaderRunner.ts`: lower-level shader execution helpers for primitive routes.
- `tensorShapes.ts`: tensor-shape arithmetic and validation helpers.

A recurring pattern in the worker is explicit ownership: buffers that are created for a step are either released through the optimization context or returned as owned outputs to the caller.

## Operation modules

Worker ops are grouped by operator family under `src/ml/worker/ops/`:

- `convolution/`: conv2d forward, fused conv+ReLU, and input-gradient backward variants.
- `relu/`: ReLU forward/backward.
- `pooling/`: max-pool forward/backward.
- `normalization/`: ImageNet normalization forward/backward.
- `pointcloud/`: point-feature exp/normalization, point-cloud convolution, and
  surface-pool helpers for point-cloud parity routes and GPU-resident pipeline
  experiments.
- `gram/`: Gram matrix forward/backward.
- `loss/`: MSE, content loss, style loss, and weighted scalar sum helpers.

Most families split shader source generation from run helpers. Many run helpers expose both readback routes for tests and buffer-returning routes for pipeline execution.

## Optimization pipelines

### First-pool pipeline

The first-pool optimizer is a smaller benchmark and parity target. It optimizes through a truncated VGG path ending at the first max-pool and is useful for fast iteration on forward/backward/update behavior.

### Full style-transfer pipeline

The full pipeline under `src/ml/worker/pipelines/optimization/style-transfer/` runs the fixed VGG19 style-transfer graph:

1. Uploads content, style, and current input image tensors.
2. Parses supplied VGG19 conv weights.
3. Builds persistent target context for content features and style Gram matrices when possible.
4. Runs forward passes through the configured VGG19 layer plan.
5. Computes content/style losses.
6. Backpropagates gradients from tap layers to the input tensor.
7. Applies the selected optimizer update.
8. Clamps pixels and records timing/kernel stats.
9. Reads back losses according to `lossReadbackInterval` and returns final/output values for preview.

The worker can keep optimizer state by `sessionId`. `clear-style-transfer-session` disposes state when the main thread resets or changes incompatible run settings.

### Input optimizers

`input-optimizer/` provides a common interface over GPU vector operations. Current modes are:

- `sgd`
- `adam`
- `lbfgs`

The optimizer updates buffers representing image pixels, not VGG19 weights.

## VGG19 layer constants

`src/ml/constants/vgg19.ts` centralizes the torch `vgg19.features` ReLU tap indices used by the app and tests. Pipeline layer schedules live in `src/ml/worker/pipelines/optimization/layerSchedules.ts` and `style-transfer/vgg19Plan.ts`.

Keeping UI tap constants and worker schedules aligned is important. If future work changes tap layers or truncation depth, update constants, pipeline plans, tests, and fixture exporters together.

## Model-pack loading

Manifests are parsed by `src/ml/worker/models/vgg19/weights.ts`. The parser validates supported formats, shard byte lengths, checksums, tensor byte ranges, and tensor shapes, then decodes weights to `Float32Array` for the compute pipeline.

The app resolves model URLs through `src/shared/assetUrls.ts`:

- `assetUrl()` respects Vite `BASE_URL`.
- `vgg19ModelUrl()` uses `VITE_VGG19_MODEL_BASE_URL` when configured and otherwise falls back to public assets.

## End-to-end app flow

1. React renders defaults and starts the controller hook.
2. The controller creates the worker and initializes WebGPU.
3. The controller loads selected model-pack manifests/shards, using IndexedDB when possible.
4. Content/style images are converted to NCHW float arrays at the selected resolution.
5. The controller sends `run-style-transfer` chunks to the worker.
6. The worker executes the fixed graph and optimizer updates.
7. The worker returns losses, timing stats, and updated output values.
8. The controller renders the output PNG preview and schedules the next chunk if the run is still active.

## Point-cloud preview flow

1. `RouteApp` selects `PointCloudPreviewApp` when the URL path matches
   `/pointcloud-preview` after the Vite base.
2. The route loads the committed medium example or parses an uploaded JSON
   export with `loadPointCloudMesh.ts`.
3. The loader validates the four-array schema, builds typed arrays, computes
   bounds, precomputes baked mesh colours, and derives both k-d tree and
   spatial-hash lookup structures.
4. `PointCloudPreviewScene.tsx` renders the mesh and aligned point cloud,
   switching between baked colours and spatial-hash fragment-KNN shading as the
   dataset allows.
5. Mesh hover hits use the CPU k-d tree path for exact nearest-3 inspection,
   while route-local React state tracks controls, screenshots, and saved
   viewpoints in browser storage.

## Testing architecture

Playwright is used for integration, worker, and WebGPU parity coverage. The suite includes:

- App boot and worker/WebGPU initialization checks.
- Point-cloud preview route boot, upload, fallback, screenshot, and viewpoint
  persistence checks.
- Tensor primitive parity tests.
- VGG first-pool forward and optimization checks.
- Full phase-3 forward/loss parity checks when fixtures are present.
- Backward parity checks with committed phase-4 fixture data.
- Full style-transfer endpoint checks when optional fixtures/model packs are present.
- LBFGS utility tests in the default correctness suite.

Performance-oriented benchmark specs, including pack-acceptance threshold helpers and kernel-lab smoke checks, live under `benchmarks/` and run with `npm run benchmark` instead of default CI. The tests are designed to skip optional large-fixture paths rather than fail a fresh checkout.

## Known follow-ups

- Move image conversion and worker orchestration into smaller feature modules if the controller hook continues to grow.
- Keep model-pack UI options in sync with what is actually hosted for non-local deployments.
- Continue promoting successful kernel-lab flags into default pipeline behavior only after broad device validation.
- Add a consolidated model-pack export guide or script that covers all supported formats.
