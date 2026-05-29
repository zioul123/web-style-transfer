# WebGPU Gatys Style Transfer Plan and Status

This document records the current browser-first implementation plan and the state of the port as of May 29, 2026.

## Goal

Port a minimal Gatys neural style transfer pipeline to the browser with WebGPU while keeping numerical behavior anchored to a PyTorch reference implementation.

The pipeline optimizes the output image pixels. VGG19 weights are fixed; the browser does not train or update model parameters.

## Reference algorithm

The Python reference performs this loop:

1. Load content and style images.
2. Resize and normalize images with ImageNet channel statistics.
3. Run fixed pretrained VGG19 feature layers.
4. Capture content target features at `relu4_2`.
5. Capture style target Gram matrices at `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`, and `relu5_1`.
6. Initialize the optimized image from the content image.
7. Iteratively minimize weighted content and style losses.
8. Clamp pixels to `[0, 1]` after each update.

The WebGPU implementation follows the same shape: fixed VGG19 weights, manual forward/backward for the required operators, and input-image-only optimization.

## Required operation surface

The browser runtime intentionally implements only the operations required by this fixed graph:

- Tensor validation and shape helpers for 4D NCHW tensors.
- Elementwise add/sub/mul/div and scalar broadcasting for parity tests and utility kernels.
- Clamp and MSE reductions.
- VGG-compatible convolution, fused convolution+ReLU, ReLU, max-pool, and normalization.
- CHW flattening and Gram matrix computation.
- Content/style loss computation.
- Manual backward kernels for content loss, style/Gram path, ReLU, max-pool, convolution input gradients, and normalization.
- Input optimizers: SGD, Adam, and LBFGS-style updates.

## Architecture principles

- **Worker owns compute.** The UI thread handles controls and previews; the worker owns WebGPU device state and dispatches kernels.
- **Typed message protocol.** Main thread and tests communicate through discriminated worker request/response unions.
- **No generic autograd.** The graph is fixed, so backward routing is handwritten and caches only tensors needed for input gradients.
- **GPU-resident hot path.** Full optimization keeps intermediate tensors on GPU where possible and reads back only scalar losses and output snapshots.
- **Fixtures are the oracle.** PyTorch exporters generate deterministic data for layer, loss, backward, and optimizer parity tests.
- **Model packs are manifest-backed.** Runtime loading supports fp32/fp16 and quantized pack formats through a common manifest parser, even though only some packs are committed.

## Phase status

| Phase                                     | Status                                      | Current notes                                                                                                                                         |
| ----------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phase 0: React/Vite/worker/WebGPU init    | Complete                                    | App boots, worker ping/init protocol is implemented, and Playwright boot checks exist.                                                                |
| Phase 1: Tensor primitives and simple ops | Complete                                    | Tensor roundtrip, elementwise ops, clamp, scalar broadcasting, and MSE parity are covered.                                                            |
| Phase 2: VGG forward ops                  | Complete                                    | Normalization, convolution, fused conv+ReLU, ReLU, and max-pool are implemented and tested with first-pool fixtures.                                  |
| Phase 3: Gram matrices and losses         | Complete                                    | Style/content losses are implemented and tested against phase-3 fixtures when present.                                                                |
| Phase 4: Manual backward                  | Complete for the fixed input-gradient graph | Backward kernels and PyTorch parity fixtures exist for the required loss/layer path.                                                                  |
| Phase 5: Optimizer loop                   | Implemented                                 | First-pool and full VGG-style optimization endpoints run in the worker with GPU-resident buffers and optimizer options.                               |
| Phase 6: Product-minimum UI               | Implemented, needs QA                       | The app has image upload/defaults, controls, model-pack selection/cache status, progress telemetry, previews, and download-oriented output rendering. |
| Phase 7: Benchmarking and tuning          | In progress                                 | `/benchmark` exposes first-pool, full-style, pack-acceptance, and kernel-lab workflows; optimization flags are still experimental.                    |

## Current worker endpoints

Infrastructure messages:

- `ping`
- `init-webgpu`
- `tensor-roundtrip`

Pipeline messages:

- `run-first-pool-optimizer`
- `run-style-transfer`
- `clear-style-transfer-session`

Tensor-op messages cover primitive parity/debug routes including:

- `add`, `sub`, `mul`, `div`, `clamp`, `mse`
- `normalize-forward`, `normalize-backward`
- `conv2d-forward`, `conv2d-relu-forward`, `conv2d-backward-input`
- `relu-forward`, `relu-backward`
- `maxpool2d-forward`, `maxpool2d-backward`
- `reshape-chw-flatten`, `gram-matrix`, `gram-backward`
- `content-loss`, `content-loss-backward`
- `style-loss`, `style-loss-backward`

## Model-pack strategy

The app loads VGG19 weights from `public/vgg19-models` unless `VITE_VGG19_MODEL_BASE_URL` points elsewhere. The manifest parser validates shard sizes and SHA-256 checksums and decodes:

- `fp32-le`
- `fp16-le`
- `int8-per-channel-le`
- `int8log-per-channel-le`
- `int4-experimental-le`
- `int4log-experimental-le`

Committed packs are currently limited to `int8-per-channel` and `int4log-experimental`. Optional pack names remain in the UI and benchmark code so locally generated or externally hosted packs can be compared without code changes.

## Testing and fixtures

Core command:

```bash
npm test
```

Build command required before completing changes:

```bash
npm run build
```

Fixture exporters:

```bash
python python-reference/export_vgg19_first_pool.py
python python-reference/export_vgg19_phase3_full_pass.py
python python-reference/export_phase4_backprop_fixtures.py
python python-reference/export_lbfgs_fixtures.py
```

Some Playwright specs skip when optional large fixtures or full model packs are missing. The committed small fixtures cover first-pool, phase-4 backprop, and LBFGS checks; full phase-3/full-style-transfer tests need the large generated phase-3 fixture and, for fp32-specific paths, a local or hosted fp32 model pack.

## Remaining follow-ups

1. **Performance tuning**
   - Continue reducing allocations/readbacks in benchmarked paths.
   - Promote proven kernel flags to defaults only after cross-device validation.
   - Revisit tiled/vectorized convolution and optimized pool backward variants.

2. **Model-pack workflow**
   - Add a single documented export command for all supported pack formats.
   - Decide which packs are committed versus externally hosted.
   - Keep UI options aligned with hosted/committed availability.

3. **Full-style QA**
   - Establish visual acceptance baselines at common resolutions.
   - Validate defaults on Chrome/Edge across integrated and discrete GPUs.
   - Tune optimizer defaults separately for quantized packs if needed.

4. **Main-thread cleanup**
   - Continue moving image conversion, worker orchestration, and model loading out of app presentation code.
   - Split UI components further once control behavior stabilizes.

5. **Protocol and docs hygiene**
   - Keep `src/types/worker-protocol/*` as the source of truth for worker contracts.
   - Update fixture READMEs whenever exporter outputs or skip behavior changes.
