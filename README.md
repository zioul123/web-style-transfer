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

## Deploying to GitHub Pages

This app can be hosted as a static Vite build on GitHub Pages. The included workflow at `.github/workflows/deploy-pages.yml` builds the app on pushes to `main` and publishes the `dist/` artifact.

### Repository settings

1. Push this branch to GitHub and merge it into `main`.
2. In the GitHub repository, open **Settings → Pages**.
3. Set **Build and deployment → Source** to **GitHub Actions**.
4. Run the **Deploy GitHub Pages** workflow from the **Actions** tab, or push to `main`.

The workflow sets `BASE_PATH=/${{ github.event.repository.name }}/` during `npm run build`, so Vite generates URLs that work under `https://<owner>.github.io/<repo>/`.

### VGG19 model pack hosting

By default, runtime model fetches use Vite's public asset base and resolve to `<base>/vgg19-models/...`. If the generated VGG19 model packs are committed under `public/vgg19-models`, they will be included in the Pages artifact automatically.

If you do not want to include the large model shards in the Pages artifact, host them from GitHub raw files instead and set `VITE_VGG19_MODEL_BASE_URL` while building. Use a raw URL, not the human-facing `github.com/<owner>/<repo>/blob/...` URL, for example:

```bash
VITE_VGG19_MODEL_BASE_URL=https://raw.githubusercontent.com/<owner>/<repo>/main/public/vgg19-models npm run build
```

To use this in GitHub Actions, uncomment and update the `VITE_VGG19_MODEL_BASE_URL` line in `.github/workflows/deploy-pages.yml`.

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

### GPU-resident style-transfer execution

`run-style-transfer` uses the GPU-resident optimization path. Intermediate tensors remain on GPU where possible, with readback limited to scalar loss reporting and final/output snapshots.

### Current transfer bottleneck notes

The optimization pipelines keep intermediate tensors GPU-resident and only read back scalar losses plus final/output snapshots. The remaining readback-heavy paths are the op-level worker routes used by parity tests and debugging; those intentionally return JSON-friendly arrays to the main thread.
