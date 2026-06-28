# Architecture

This is the short architectural contract for planning changes. See
`docs/architecture-overview.md` for a fuller module-by-module description and
`docs/code-map.md` for concrete entry points.

## System Boundaries

The application has three primary layers and one optional local backend:

1. **React main thread:** renders the app, handles user input and previews,
   selects model packs, and orchestrates worker requests.
2. **Typed worker protocol:** defines the complete request/response contract
   shared by the UI, worker, and tests.
3. **Worker-owned WebGPU compute:** owns device initialization, GPU resources,
   kernels, fixed VGG19 forward/backward execution, and input optimizers.
4. **Local FastAPI backend:** optional PyTorch/torchvision execution for image
   style-transfer chunks. It owns server-side VGG19 weights and optimizer
   sessions; the browser falls back to the WebGPU worker when it is unavailable.

Do not move expensive model execution onto the main thread. Do not bypass the
typed protocol to couple React code directly to worker internals.

## Runtime Flow

1. `src/main.tsx` mounts `src/RouteApp.tsx`.
2. `RouteApp` selects the main UI, `/benchmark`, or `/pointcloud-preview`
   route.
3. `useStyleTransferController` loads images and model packs, initializes the
   worker, probes the optional FastAPI backend, and builds typed requests.
4. `src/styleTransfer.worker.ts` mounts the worker message router.
5. The router validates request discriminants and delegates to an operation or
   optimization pipeline.
6. Worker runtime helpers allocate, reuse, read back, and release GPU buffers.
7. In auto backend mode, the controller sends JSON tensor chunks to FastAPI
   when health checks pass; otherwise it sends the existing worker request.
8. The selected backend returns losses, output values, status, and timing data.
9. The controller converts output tensors to a preview and updates the UI.

## Ownership Rules

- React components own presentation; the controller hook owns main-thread side
  effects and run state.
- `src/types/worker-protocol/` is the source of truth for worker contracts.
- Message routers dispatch and translate errors; they should not contain
  numerical kernels.
- `backend/style_transfer_backend/` owns local FastAPI routing, request schemas,
  and PyTorch execution. It must not depend on browser model-pack payloads.
- Runtime modules own reusable WebGPU infrastructure and buffer lifecycle.
- Operation modules own one operator family and commonly separate WGSL shader
  generation from dispatch helpers.
- Pipeline modules compose operations for the fixed optimization graph.
- Python reference scripts and fixtures are numerical oracles, not runtime
  dependencies.

## Change Routing

| Change                           | Start Here                              | Usually Also Inspect                                 |
| -------------------------------- | --------------------------------------- | ---------------------------------------------------- |
| UI/control behavior              | `src/features/style-transfer/`          | `src/App.tsx`, UI tests                              |
| Optional FastAPI backend         | `backend/style_transfer_backend/`       | backend tests, controller adapter, README            |
| Point-cloud preview route        | `src/features/pointcloud-preview/`      | `RouteApp.tsx`, `PointCloudPreviewApp.tsx`, UI tests |
| Worker request/response          | `src/types/worker-protocol/`            | routers, responses, worker tests, docs               |
| GPU helper or resource lifecycle | `src/ml/worker/runtime/`                | all callers, parity tests                            |
| Kernel behavior                  | `src/ml/worker/ops/<family>/`           | tensor-op router, fixtures, parity tests             |
| Full optimization behavior       | `src/ml/worker/pipelines/optimization/` | protocol, controller, runtime tests                  |
| Model-pack format/loading        | `modelPacks.ts`, `weights.ts`           | cache, public README, acceptance tests               |
| Performance flag                 | benchmark and pipeline settings         | correctness parity and benchmark evidence            |
| Route/deployment behavior        | `RouteApp.tsx`, `vite.config.ts`        | Pages workflow, app boot tests                       |

## Invariants

- VGG19 weights are fixed; only image pixels are optimized.
- Main-thread and worker messages remain discriminated and exhaustively handled.
- Tensor shapes, shard sizes, checksums, and supported formats stay validated.
- FastAPI requests use validated JSON tensors and server-owned VGG19 weights,
  and browser clients only send those tensors to loopback backend URLs after a
  matching FastAPI/PyTorch health response.
- GPU buffers have explicit ownership and cleanup.
- Correctness changes are anchored to deterministic fixtures or focused tests.
- Performance optimizations must preserve baseline semantics and include
  benchmark context.
- Optional large fixtures may be absent in a clean checkout; tests should skip
  only where the existing fixture policy permits.
