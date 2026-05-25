# WebGPU Gatys Style Transfer: Minimal Browser-First Plan

## 1) What the Python reference actually does

The current reference implementation is the classical Gatys optimization loop over the **input image pixels** (not model weights):

1. Load content + style images and resize.
2. Run both through pretrained VGG19 feature extractor.
3. Capture one content target feature map (`relu4_2`) and five style targets (`relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`, `relu5_1`).
4. Compute style targets via Gram matrices.
5. Initialize output image from content.
6. Iteratively optimize output image with weighted sum:
   - content MSE between output/content features
   - style MSE between output/style Gram matrices
7. Clamp optimized pixels to `[0, 1]` after each step.

This means we do **inference + input optimization**, with fixed network weights and no training loop.

## 2) Minimal operation set required in-browser

For a lean WebGPU implementation, the operation surface can be heavily constrained to what this script uses:

### Tensor/data primitives

- `float32` tensor storage (optionally `float16` later for speed/memory).
- 4D NCHW logical layout: `[1, C, H, W]`.
- Tensor reshape/view (`[B,C,H,W] -> [C, H*W]` for Gram).
- Basic elementwise ops: add, subtract, multiply, divide.
- Scalar-tensor ops.
- Clamp `[0,1]` on image tensor.

### Neural network forward ops

- 2D convolution (stride 1, padding 1, kernel 3x3, bias add).
- ReLU.
- MaxPool2d (kernel 2, stride 2).
- Input normalization per channel: `(x - mean) / std`.

### Loss ops

- Mean squared error (MSE).
- Matrix multiply for Gram matrix: `G = F * F^T`.
- Gram normalization by `(B*C*H*W)`.
- Weighted sum of losses.

### Gradient/backprop ops (only what is needed)

- Backprop through:
  - MSE
  - Gram construction (`matmul` + normalization)
  - ReLU
  - MaxPool (argmax mask for backward)
  - Conv2d (input gradient only; no weight/bias gradient needed)
  - Normalize layer
- Optimizer update for input image only (start with SGD/Adam; LBFGS optional).

## 3) What can be removed vs PyTorch

Not needed for MVP:

- Generic autodiff graph engine for arbitrary models.
- Parameter gradient computation for VGG weights.
- Training mode features (dropout, batchnorm updates).
- Dynamic model mutation at runtime.
- Broad tensor API parity with PyTorch.
- Multi-batch support (`B=1` is enough initially).

## 4) Lightweight "ML backend" architecture in frontend

## 4.1 Suggested project structure (frontend-first, backend-ready)

```txt
src/
  app/                     # React app shell, routes, UI state
  features/style-transfer/
    ui/                    # upload controls, presets, progress, preview
    hooks/                 # useStyleTransferController
    workers/
      styleTransfer.worker.ts
  ml/
    core/                  # tensor wrapper, scheduler, shape checks
    webgpu/                # adapter, buffer pools, shader pipeline utils
    ops/                   # conv, relu, pool, mse, matmul, gram, clamp
    graph/                 # fixed execution plan for VGG feature taps
    autograd-lite/         # minimal manual backward for required ops
    optim/                 # sgd/adam over input image tensor
    models/
      vgg19/               # static weights, layer config, tap definitions
  render/
    r3f/                   # optional image plane / transition visualization
  shared/
    types/
    utils/
```

Future backend compatibility:

- Keep model config + weights loading behind an interface (`ModelProvider`).
- Keep image IO + optimization API shape stable so a server path can be swapped later.

## 4.2 Runtime design

- React UI thread handles controls and previews.
- Heavy ML compute runs in a Web Worker (no UI jank).
- Worker owns WebGPU device and compute pipelines.
- Progress snapshots emitted every N steps to UI.

## 4.3 Weight strategy

- Export VGG19 feature weights once (offline script) to compact binary shards.
- Bundle only required feature extractor layers (through deepest tapped layer).
- Keep layer metadata JSON: channels, kernel dims, padding/stride.

## 5) Backprop approach: minimal + manual

Since graph is fixed, manual backward is practical and lean:

1. During forward, cache only tensors needed for backward (activations, pool indices/masks, feature maps used by taps).
2. Compute scalar losses and seed gradients at tap points.
3. Backprop from taps to input with op-specific gradient kernels.
4. Update input image tensor.

This avoids building a generic dynamic tape system. A tiny "autograd-lite" registry can map each fixed op node to:

- forward kernel
- backward kernel
- required cache metadata

## 6) Incremental tasks with verification at each step

## Phase 0 — Scaffolding

1. Initialize React + TypeScript + Vite.
2. Add WebGPU capability check + fallback message.
3. Add Worker wiring and message protocol.

Verification:

- App boots, worker responds to ping, WebGPU adapter/device creation passes.

## Phase 1 — Core tensor + simple ops

1. Implement tensor buffers + shape/stride metadata.
2. Implement elementwise ops + clamp + MSE.
3. Implement CPU reference versions for each op (test oracle).

Verification:

- Unit tests compare GPU vs CPU within tolerance.
- Deterministic small tensors (e.g. 4x4) snapshot tests.

## Phase 2 — Conv/ReLU/Pool forward only

1. Implement conv2d forward kernel for VGG-compatible params.
2. Implement ReLU and maxpool forward.
3. Implement normalization layer.

Verification:

- Layer-by-layer parity tests against exported PyTorch reference tensors.
- End-to-end forward through truncated VGG on one test image.

## Phase 3 — Gram + losses

1. Implement reshape + matmul Gram computation.
2. Implement content/style loss accumulation with per-layer weights.

Verification:

- Compare per-layer style/content loss values vs Python for fixed seeds and image.

## Phase 4 — Manual backward (input gradients only)

1. Add backward kernels for MSE, Gram path, ReLU, pool, conv, normalize.
2. Implement activation caches and gradient routing from tapped layers.

Verification:

- Finite-difference gradient checks on tiny tensors.
- Compare input gradient stats/direction with PyTorch autograd baseline.

## Phase 5 — Input optimizer loop

1. Implement SGD first (optional Adam next).
2. Iterative optimization loop in worker with periodic preview readback.
3. Clamp after each step.

Verification:

- Loss decreases over steps.
- Output visually reflects style transfer at 64px/128px test settings.

## Phase 6 — Product minimum UI

1. Upload style/content images.
2. Basic controls: steps, style weight, content weight, resolution preset.
3. Show progress and final download.

Verification:

- Manual UX pass across Chrome/Edge WebGPU-enabled builds.
- Memory use stable across repeated runs.

## 7) Recommended first MVP constraints

To stay lean and hit working results quickly:

- Fixed resolution preset(s): 128x128 first.
- Fixed VGG taps same as Python reference.
- Batch size 1 only.
- Optimizer = SGD or Adam only initially.
- Float32 first; add FP16 optimization later.

## 8) Optional use of React Three Fiber

R3F is not required for core ML, but can be valuable for:

- side-by-side textured planes (content/style/output)
- animated transition between iterations
- zoom/pan canvas interaction

Keep this strictly presentation-side; do not couple with compute core.

---

## Implementation status snapshot (updated May 17, 2026)

### Phase 0

- ✅ React + TypeScript + Vite scaffolding is in place.
- ✅ Worker wiring and ping/init protocol are implemented.
- ✅ Playwright phase-0 validation is passing in Chromium+SwiftShader environments.

### Phase 1

- ✅ Core tensor shape/data primitives are implemented.
- ✅ CPU reference ops are implemented for `add`, `sub`, `mul`, `div`, `clamp`, `mse`.
- ✅ Worker WebGPU kernels are implemented for `add`, `sub`, `mul`, `div`, `clamp`, `mse`.
- ✅ Scalar operand support is implemented for binary ops (`tensor ⊕ scalar` and `scalar ⊕ tensor`) with explicit erroring for unsupported `scalar ⊕ scalar` binary requests.
- ✅ MSE request validation enforces tensor-tensor operands only.
- ✅ Clamp supports both tensor and scalar inputs.
- ✅ Playwright parity tests are passing for tensor roundtrip, exhaustive tensor/scalar op coverage, and GPU MSE reduction parity across lengths that exercise multi-pass reduction logic (`64`, `64 * 5`, `64 * 5 + 1`, `64 * 5 + 32`, `64 * 5 + 63`).
- ✅ **Phase 1 is complete**.

### Notes for future agents

- Current worker architecture is intentionally minimal and message-oriented to keep parity testing deterministic.
- Shader code generation has been separated into helper functions to make future extraction into dedicated shader modules straightforward.
- For scalar binary ops, the current implementation uses shader-mode specialization with scalar broadcast reads (`b[0]`) rather than CPU-side scalar expansion.
- The next high-leverage work item is proceeding into Phase 2 conv/relu/pool forward ops.
- Phase 2 forward kernels currently favor straightforward correctness-first implementations; obvious future optimization targets include shared-memory tiled conv, vectorized loads/stores, and fused conv+bias+relu passes.

---

## Phase 4 implementation notes (manual backward, fixed graph)

### What is now in place

- Worker protocol includes explicit backward ops:
  - `relu-backward`
  - `maxpool2d-backward`
  - `normalize-backward`
  - `conv2d-backward-input`
  - `content-loss-backward`
  - `gram-backward`
  - `style-loss-backward`
- Backward parity fixture generation exists in Python and exports deterministic tiny tensors from PyTorch.
- Playwright spec validates each backward path against fixture expected gradients.

### Forward/backward mechanism details

1. **Forward execution** remains message-based and op-granular in `styleTransfer.worker.ts`.
2. **Backward seeds** are generated from scalar losses:
   - content loss seed: `2 * (input - target) / N`
   - style loss seed: MSE gradient in Gram space.
3. **Gram backward** maps Gram-space gradients back to feature-space gradients with explicit normalization by `B*C*H*W`.
4. **Layer backward routing** applies local Jacobian rules for ReLU, MaxPool, Normalize, and Conv2d input gradients.
5. **Input-only optimization intent** is preserved: no weight/bias updates are part of this phase.

### GPU and alignment caveat

When encoding uniforms for backward kernels in WGSL, ensure 16-byte alignment for uniform structs. A prior mismatch can cause hangs/timeouts during test runs, especially in SwiftShader environments.

### Suggested follow-up cleanup

- Move backward kernels into dedicated shader modules for easier profiling.
- Add small shared helper(s) for shape compatibility checks to keep handler logic compact.
- Add a full phase-4 end-to-end fixture/spec that verifies chained gradient routing from multiple taps in one pass.

## Implementation update (May 19, 2026)

### Added optimization mode

- `fusedOps`: fused conv+ReLU path in worker forward execution.

The older CPU-resident full pipeline and super-fused block scheduling path have since been removed; the full style-transfer endpoint now uses the GPU-resident pipeline as the single production path.

### Readability and transfer-focused cleanup

- Conv layer metadata is cached once per run (`weight`, `shape`, `bias`) to avoid repeated `Float32Array` conversion/allocation in forward/backward loops.
- Backward routing in fused mode uses stored ReLU outputs directly to keep logic consistent with fused forward.

### Remaining easy optimization target

The worker currently performs per-op GPU->CPU readback in `runUnaryShader`. This keeps correctness straightforward but creates serialization overhead. The next pragmatic performance milestone is introducing GPU-resident intermediate buffers and only reading back at required boundaries (loss scalar extraction and final/output snapshots).
