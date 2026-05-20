# Web Style Transfer

## Introduction

This project ports a minimal Gatys style transfer pipeline to the browser with WebGPU. The Python reference remains in `python-reference/` and is used as the parity baseline for operation-level and loss-level verification.

## Running Python Version

From the root folder, first install dependencies:

```bash
pip install -r requirements.txt
```

Then, run the script with:

```bash
python python-reference/style-transfer.py
```

This generates `./expt` outputs from style transfer.

## Phase 0 Web App (React + TypeScript + Vite + Tailwind)

Install dependencies:

```bash
npm install
```

Run the app:

```bash
npm run dev
```

Build production bundle:

```bash
npm run build
```

## WebGPU/Worker verification with Playwright (SwiftShader)

This environment has no physical GPU. Use Chromium + SwiftShader to validate worker wiring and WebGPU adapter/device initialization behavior.

Install browser + OS dependencies:

```bash
npx playwright install chromium
npx playwright install-deps chromium
```

Run phase 0 browser tests:

```bash
npm test
```

## WebGPU Port Planning

See `docs/webgpu-style-transfer-plan.md` for the full phased implementation plan.

## Current ML Worker Architecture (Forward + Backward)

The browser ML runtime is intentionally worker-first and message-driven:

- `src/styleTransfer.worker.ts` owns WebGPU device lifecycle and kernel dispatch.
- The app sends typed `tensor-op` requests (`src/types.ts`) and receives scalar/vector results.
- Each op is validated, executed on GPU where supported, and returned in plain JSON-friendly arrays.

### Forward path currently implemented

- Tensor elementwise ops: `add`, `sub`, `mul`, `div`, `clamp`, `mse`.
- VGG feature ops: `normalize-forward`, `conv2d-forward`, `relu-forward`, `maxpool2d-forward`.
- Style-transfer loss prep: `reshape-chw-flatten`, `gram-matrix`, `content-loss`, `style-loss`.

### Backward path currently implemented (phase 4 groundwork)

- Loss/Gram backward on GPU in worker:
  - `content-loss-backward`
  - `gram-backward`
  - `style-loss-backward` (gram + mse chain)
- Layer backward reference ops exposed via worker routes:
  - `relu-backward`, `maxpool2d-backward`, `normalize-backward`, `conv2d-backward-input`

Important design choice: gradients are computed for the **input image tensor only**. VGG weights remain fixed.

### Parity and fixture strategy

- PyTorch fixture generator: `python-reference/export_phase4_backprop_fixtures.py`
- Generated fixture artifact: `public/phase4-backprop/phase4_backprop_fixture.json`
- Backward parity test: `tests/phase4.spec.ts`

This keeps implementation deterministic and reviewable while preserving a direct numerical baseline against PyTorch.


### Fused execution modes (Phase 6+)

`run-style-transfer` now supports two optional performance flags:

- `fusedOps`: uses fused conv+ReLU in forward and compatible ReLU backward gating.
- `superFusedOps`: runs conv/relu layers in VGG-stage block scheduling (between pool boundaries) while still materializing required ReLU taps for style/content losses.

These can be toggled from the UI (`Use fused conv+relu`, `Use super fused blocks`) for benchmarking.

### Current transfer bottleneck notes

Most heavy math is on GPU, but each op still maps results back to CPU (`runUnaryShader`) before the next op. This is the dominant remaining overhead. Recent cleanup reduces avoidable CPU allocations (pre-cached conv weights/biases), but the next major win is a GPU-resident execution chain (buffer reuse + deferred readback at chunk boundaries).
