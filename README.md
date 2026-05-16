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
