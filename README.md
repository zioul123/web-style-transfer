# Web Style Transfer

## Introduction

The purpose of this app is to perform style transfer between images within the browser environment, using WebGPU. As an entrypoint, we have a working python implementation that is as trimmed down as possible, and should be used as reference for all the required operations needed in our WebGPU implementation.

## Running Python Version

From the root folder, first install dependencies:

```bash
pip install -r requirements.txt
```

Then, run the script with:

```bash
python python-reference/style-transfer.py
```

This will generate a folder `./expt` with the outputs from style transfer.

## Web App Setup

Install JavaScript dependencies from the repository root:

```bash
npm install
```

Run the local dev server:

```bash
npm run dev
```

Build the web app for production (also useful as a quick type/build verification):

```bash
npm run build
```

## Testing Guide

This repository uses **two** JavaScript test layers:

- **Unit tests (Vitest)**
  - Located in files that end in `*.test.ts` (example: `src/ml/ops/ops.test.ts`).
  - Purpose: fast correctness checks for pure logic/math and module behavior.
  - Command:

    ```bash
    npm run test:unit
    ```

- **E2E/browser tests (Playwright)**
  - Located in files that end in `*.spec.ts` under `tests/` (example: `tests/webgpu-ops.spec.ts`).
  - Purpose: validate real browser execution paths (including WebGPU behavior).
  - One-time browser install requirement:

    ```bash
    npx playwright install chromium
    ```

  - On Linux/container environments, install required shared libraries if browser launch fails:

    ```bash
    npx playwright install-deps chromium
    ```

  - Run all Playwright tests:

    ```bash
    npm run test:e2e
    ```

  - Run the dedicated SwiftShader WebGPU parity test:

    ```bash
    npm run test:e2e:swiftshader
    ```

### Which test command should I run?

- Use `npm run test:unit` during quick iteration on math/core logic.
- Use `npm run test:e2e` when changing browser/runtime/WebGPU behavior.
- Use `npm run test:e2e:swiftshader` when debugging GPU kernel parity specifically.
- Use `npm test` to run both suites together before opening or updating a PR.

## Repository Orientation (for developers and future AI sessions)

- `docs/webgpu-style-transfer-plan.md`: implementation phases and verification milestones.
- `src/ml/core/`: tensor core abstractions.
- `src/ml/ops/cpu.ts`: CPU fallback/oracle operations.
- `src/ml/ops/gpu.ts`: WebGPU-backed operation implementations and fallback metadata.
- `src/ml/webgpu/adapter.ts`: adapter/device capability detection utilities.
- `src/features/style-transfer/workers/`: worker protocol, worker implementation, and worker client.
- `src/app/App.tsx`: basic app shell and runtime status display.

## WebGPU Port Planning

See `docs/webgpu-style-transfer-plan.md` for a step-by-step breakdown of the minimal operations, lightweight manual backprop plan, and incremental verification tasks for the frontend WebGPU implementation.
