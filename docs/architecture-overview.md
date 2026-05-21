# Web Style Transfer Architecture Overview

This document explains the current architecture of the application end-to-end, from React UI input to WebGPU worker execution, and calls out areas where responsibilities appear out-of-place and could be reorganized in follow-up refactors.

## 1. System shape at a glance

The app is a **browser-first style transfer system** with three major layers:

1. **UI + orchestration on main thread** (`src/App.tsx`)
   - owns controls, image loading, and iterative run loop state.
2. **Typed request/response protocol** (`src/types.ts`)
   - defines all worker messages for op-level tests and pipeline-level optimization runs.
3. **Worker-side compute stack** (`src/styleTransfer.worker.ts` + `src/ml/worker/**`)
   - owns WebGPU device lifecycle, message routing, kernels, runtime helpers, and optimization pipelines.

Data flow is intentionally message-driven:

- Main thread posts `WorkerRequest` messages.
- Worker routes by `type`, runs an op/pipeline, and posts `WorkerResponse`.
- Main thread updates iteration state and may enqueue another run chunk.

---

## 2. Main thread architecture

## 2.1 App shell and control plane (`src/App.tsx`)

`App.tsx` currently combines UI, worker lifecycle, payload construction, and utility conversion logic:

- Creates and manages a single Worker instance (`styleTransfer.worker.ts`).
- Pings worker and initializes WebGPU.
- Loads VGG artifacts from `public/vgg19-phase3-full-pass`.
- Converts uploaded images to NCHW tensor data.
- Sends `run-style-transfer` requests with optimizer settings.
- Receives chunked results (`losses`, `finalValues`, `stats`) and updates preview image + run telemetry.

Important local constants and knobs include:

- Style/content tap indices (`STYLE_LAYER_INDICES`, `CONTENT_LAYER_INDEX`).
- Resolution presets.
- Optimizer mode and hyperparameters (`sgd`, `adam`, `lbfgs`).
- Fusion flags (`fusedOps`, `superFusedOps`).

### Out-of-place note

`App.tsx` contains substantial logic that would better live in feature modules:

- Image ↔ tensor conversion helpers.
- Worker request dispatching/retry/chunk loop behavior.
- Weight-loading concerns.

A future `features/style-transfer/` split (controller hook + data adapters + UI components) would reduce coupling and improve testability.

## 2.2 Main-thread ML exports (`src/ml/index.ts`, `src/ml/ops/cpu.ts`, `src/ml/tensor.ts`)

The main-thread ML module exports:

- `createTensor` validation wrapper.
- CPU reference ops used as correctness oracles in tests.

This gives the project a shared tensor shape contract across UI/tests/worker.

---

## 3. Protocol layer (`src/types.ts`)

`src/types.ts` is the contract between threads and tests.

It defines:

- Tensor data model (`TensorShape`, `WorkerTensor`, `WorkerTensorOperand`).
- Op-level request variants (`tensor-op` union for conv/relu/pool/gram/loss/grad operations).
- Pipeline-level request variants:
  - `run-first-pool-optimizer`
  - `run-style-transfer`
- Response unions for success/failure and scalar/vector payload forms.
- Runtime stats payload (`WorkerRunStats`).

### Strengths

- Strong discriminated unions prevent broad/unsafe envelopes.
- Enables test suite to validate ops independently from full optimization runs.

### Out-of-place note

This file is becoming large and spans multiple concerns (primitive ops, pipeline endpoints, stats types). It could be split into:

- `types/worker-protocol/core.ts`
- `types/worker-protocol/tensor-ops.ts`
- `types/worker-protocol/pipelines.ts`

---

## 4. Worker bootstrap and routing

## 4.1 Worker entrypoint (`src/styleTransfer.worker.ts`)

Entrypoint is intentionally minimal:

- imports `mountMessageRouter`
- mounts listener

This is clean and aligns with the current refactor goal.

## 4.2 Message router (`src/ml/worker/main-thread-protocol/messageRouter.ts`)

Router responsibilities:

- Parses `WorkerRequest` by discriminant.
- Handles worker infra messages (`ping`, `init-webgpu`, roundtrip).
- Dispatches pipeline requests directly to:
  - `runFirstPoolOptimizer`
  - `runStyleTransfer`
- Handles a broad `tensor-op` branch for layer/loss primitives and backward ops.

### Strengths

- Centralized, explicit dispatch.
- Keeps protocol behavior in one place.

### Out-of-place note

`tensor-op` branch is now very large and includes repeated marshal/convert boilerplate. This is a candidate for extraction into dedicated handlers (e.g., `tensorOpRouter.ts`) and possibly op-family subrouters.

## 4.3 Response helpers (`responses.ts`) and WebGPU init (`initWebGpu.ts`)

- `responses.ts` provides typed helper emitters for common message payloads.
- `initWebGpu.ts` handles adapter/device init and clears buffer pool on reset.

Out-of-place note: `messageRouter.ts` still sometimes uses direct `postResponse` while helpers exist for error/result pathways; consistency could be improved by using the helper surface uniformly.

---

## 5. Worker runtime infrastructure (`src/ml/worker/runtime/*`)

## 5.1 Device state (`deviceState.ts`)

Owns singleton-ish worker GPU device state with set/get/clear accessors.

## 5.2 Shader runner (`shaderRunner.ts`)

Generic utility layer for dispatching compute kernels against buffers and returning mapped readbacks. Includes scalar/tensor binary op helpers and clamp.

## 5.3 Buffer pool (`bufferPool.ts`)

Reusable GPU buffer pool to reduce allocation churn. Explicit acquire/release lifecycle.

## 5.4 Operand helpers (`operands.ts`)

Parses protocol operands into tensor/scalar values for op runners.

## 5.5 GPU usage constants (`gpuFlags.ts`)

Centralized usage/mapping constants for consistent buffer creation flags.

### Architectural observation

Runtime primitives are fairly reusable and reasonably separated, but some API signatures are still verbose (many repeated usage-flag args). A small `PipelineContext` or runtime facade could reduce repetitive call signatures across routers/pipelines.

---

## 6. Worker operation modules (`src/ml/worker/ops/*`)

Ops are split by domain and pair `.run.ts` wrappers with shader sources in `.shader.ts` files.

- `convolution/`: conv forward/backward-input, fused conv+relu, and super-fused variants.
- `relu/`: forward/backward.
- `pooling/`: maxpool forward/backward.
- `normalization/`: forward/backward.
- `gram/`: gram forward/backward.
- `loss/`: mse, content loss backward, style loss backward.

This structure matches the phased implementation plan and supports isolated parity testing.

### Out-of-place note

The naming convention mixes conceptual level in a few places (`conv2d.superfused.ts` vs pipeline-level super-fusion control). It works, but documenting whether fusion is an **op implementation detail** or **pipeline scheduling policy** would reduce ambiguity.

---

## 7. Optimization pipelines (`src/ml/worker/pipelines/optimization/*`)

## 7.1 First-pool optimizer (`firstPoolOptimizer.ts`)

Contains the targeted optimization route for the early VGG slice. It performs:

- normalization
- conv/relu/pool sequence
- style/content losses at configured points
- backward pass for image gradient
- optimizer update + clamp loop

Also exports a shared `runUnary` function used by the message router’s op paths.

### Out-of-place note

`runUnary` is a low-level runtime helper and not conceptually owned by the first-pool pipeline. It likely belongs in `runtime/` (or a worker compute facade), with both pipelines and router importing it from there.

## 7.2 Style transfer pipeline (`styleTransferPipeline.ts`)

Contains the full VGG-style optimization orchestration:

- constructs conv layer cache from payload weights
- runs baseline/fused/super-fused forward depending on flags
- computes style + content losses
- backpropagates through required ops
- applies SGD/Adam/LBFGS update
- clamps and reports timing stats

This is the heaviest compute coordinator in the codebase.

## 7.3 Layer schedules (`layerSchedules.ts`)

Centralized constants:

- ReLU layer indices
- pool layer indices
- super-fused block boundaries

Good separation of schedule policy from execution logic.

### Out-of-place note

Some related layer metadata still appears in other modules (e.g., UI tap indices in `App.tsx`). Consolidating these into one shared schedule/config module (possibly split by “pipeline schedule” vs “UI defaults”) would avoid drift.

---

## 8. End-to-end execution sequence

1. App loads weights and user images.
2. App converts content/style to normalized tensor value arrays.
3. App initializes worker and WebGPU device (`init-webgpu`).
4. App sends `run-style-transfer` with chunk step count.
5. Worker router forwards to `runStyleTransfer`.
6. Pipeline executes forward/loss/backward/update loop for `steps`.
7. Worker returns losses + updated image tensor + timing stats.
8. App renders updated preview and may enqueue next chunk while running.

---

## 9. Testing architecture

Playwright e2e suite validates progressive phases:

- app boot / worker availability
- primitive op parity and reduction behavior
- VGG forward parity
- loss parity
- backward parity
- endpoint optimization behaviors (including fused and super-fused paths)

Fixture generation scripts in `python-reference/` provide deterministic baselines for parity checks.

---

## 10. Architectural fit review (May 21, 2026)

The two highest-priority refactors from the previous version of this document have been addressed in code:

1. **`runUnary` ownership moved to runtime** ✅
   - `runUnary` now lives in `src/ml/worker/runtime/computeContext.ts`.
   - Both optimization pipelines and `tensorOpRouter` import it from runtime.
   - `firstPoolOptimizer.ts` no longer owns cross-cutting runtime helpers.

2. **`tensor-op` routing extracted from `messageRouter.ts`** ✅
   - `src/ml/worker/main-thread-protocol/tensorOpRouter.ts` now owns tensor-op dispatch.
   - `messageRouter.ts` delegates tensor-op requests to this dedicated router.
   - The top-level router is materially smaller and easier to scan.

Additional previously-suggested refactors are also now implemented:

- **Protocol type sprawl**: resolved by splitting into `src/types/worker-protocol/{core,messages,pipelines,tensor-ops}.ts`, with `src/types.ts` as a barrel export.
- **UI-heavy orchestration in `App.tsx`**: largely addressed by `src/features/style-transfer/hooks/useStyleTransferController.ts`, with `App.tsx` now mainly composing UI and controller state.

Still-open architecture concerns are now narrower and are listed as concrete follow-up tasks in section 11.

---

## 11. Suggested follow-up refactor tasks (remaining confusion points)

### Issue: Style/content tap indices are duplicated across modules and currently offset by one between UI/controller defaults and canonical VGG constants.

:::task-stub{title="Unify VGG tap index definitions and eliminate offset drift"}
Create a single source of truth for style/content tap layer indices under `src/ml/constants/vgg19.ts`, and make `src/features/style-transfer/hooks/useStyleTransferController.ts` consume that source directly.

Steps:
1. Locate all tap-index definitions and usages:
   - `src/ml/constants/vgg19.ts` (`VGG19_STYLE_LAYER_INDICES`, `VGG19_CONTENT_LAYER_INDEX`)
   - `src/features/style-transfer/hooks/useStyleTransferController.ts` (`STYLE_LAYER_INDICES`, `CONTENT_LAYER_INDEX`)
   - Any payload builders that pass `styleLayerIndices` / `contentLayerIndex`.
2. Decide and document one canonical indexing scheme (e.g., pre-ReLU conv index, post-ReLU feature index, or schedule index) and encode that meaning in type names/comments.
3. Replace local literals in the controller with imported constants or a small mapping helper that makes any intentional transform explicit.
4. Add validation/assertion at request-build time so impossible index combinations fail fast with a clear error.
5. Update architecture/docs where tap-index semantics are described so future contributors do not reintroduce off-by-one conversions.
:::

### Issue: Worker request/response posting patterns are still mixed between raw `postResponse` calls and response helper wrappers, which makes error-path behavior harder to reason about.

:::task-stub{title="Standardize worker response emission through shared helper APIs"}
Normalize message emission in `src/ml/worker/main-thread-protocol/` so success/error pathways use a consistent helper surface.

Steps:
1. Audit `messageRouter.ts` and `tensorOpRouter.ts` for direct `postResponse(...)` calls and categorize them by response type (`error`, `tensor-op-result`, pipeline result).
2. Expand `responses.ts` only as needed to cover missing cases with explicit, narrow helper signatures.
3. Replace ad-hoc response construction with helper calls, preserving existing payload shapes.
4. Ensure the default/unknown request path and exception path share a uniform error envelope.
5. Add a small protocol-focused test (or extend existing worker protocol tests) that verifies representative success + failure responses still match `WorkerResponse` unions.
:::

### Issue: Runtime op execution still relies on per-op GPU→CPU readback (`runUnaryShader`) in many paths, making control flow and performance tradeoffs hard to follow.

:::task-stub{title="Introduce explicit GPU-resident execution boundaries in worker runtime"}
Refactor runtime execution to model buffer-resident intermediate tensors and explicit readback boundaries, improving both readability and future performance work.

Steps:
1. In `src/ml/worker/runtime/`, define a minimal intermediate representation for GPU-resident tensor handles (buffer + shape metadata + lifecycle ownership).
2. Add helper APIs in `computeContext.ts` for:
   - op-to-op chaining without immediate readback,
   - explicit scalar readback points (loss extraction),
   - explicit image snapshot readback points (preview/final output).
3. Migrate one constrained path first (e.g., first-pool optimizer or a short style-transfer segment) to prove API clarity before broad rollout.
4. Update buffer-pool ownership/release rules to prevent leaks when handles survive across multiple ops.
5. Document the new boundary model in `docs/architecture-overview.md` and/or runtime module docs so contributors can reason about when data is on GPU vs CPU.
:::
