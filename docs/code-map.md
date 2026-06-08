# Code Map

Use this map to narrow retrieval before editing. Start with the smallest row
that matches the task, then follow imports and tests only as needed.

## Main Source Directories

| Path                                      | Contents                                                             |
| ----------------------------------------- | -------------------------------------------------------------------- |
| `src/features/pointcloud-preview/`        | Point-cloud mesh preview page, loaders, math, and R3F scene          |
| `src/features/style-transfer/components/` | Main UI panels and preview/control components                        |
| `src/features/style-transfer/hooks/`      | Main-thread style-transfer orchestration                             |
| `src/features/style-transfer/benchmark/`  | Model-pack benchmark and acceptance helpers                          |
| `src/shared/`                             | Cross-feature browser asset URL helpers                              |
| `src/types/worker-protocol/`              | Worker request, response, tensor, op, and pipeline types             |
| `src/ml/constants/`                       | Shared VGG19 layer/tap constants                                     |
| `src/ml/ops/`                             | CPU reference helpers used by tests and small parity paths           |
| `src/ml/worker/main-thread-protocol/`     | Worker message and tensor-op routing                                 |
| `src/ml/worker/runtime/`                  | WebGPU device, buffers, shapes, caches, and dispatch infrastructure  |
| `src/ml/worker/ops/`                      | Convolution, ReLU, pooling, normalization, Gram, and loss operations |
| `src/ml/worker/pipelines/optimization/`   | Optimizers and fixed optimization pipelines                          |
| `src/ml/worker/models/vgg19/`             | Model-pack validation and weight decoding                            |

## Important Entry Points

| File                                                              | Role                                                                         |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `src/main.tsx`                                                    | React bootstrap                                                              |
| `src/RouteApp.tsx`                                                | Main versus `/benchmark` route selection                                     |
| `src/App.tsx`                                                     | Main application composition                                                 |
| `src/BenchmarkApp.tsx`                                            | Benchmark UI and worker-driven benchmark flows                               |
| `src/PointCloudPreviewApp.tsx`                                    | Standalone `/pointcloud-preview` route shell                                 |
| `src/styleTransfer.worker.ts`                                     | Worker bootstrap                                                             |
| `src/features/style-transfer/hooks/useStyleTransferController.ts` | Main-thread state, image conversion, model loading, and worker orchestration |
| `src/ml/worker/main-thread-protocol/messageRouter.ts`             | Top-level worker request dispatch                                            |
| `src/ml/worker/main-thread-protocol/tensorOpRouter.ts`            | Operation-level parity/debug dispatch                                        |
| `src/ml/worker/pipelines/optimization/styleTransferPipeline.ts`   | Full pipeline wrapper and session clearing                                   |
| `src/ml/worker/pipelines/optimization/style-transfer/pipeline.ts` | Full style-transfer execution                                                |
| `src/types.ts`                                                    | Public re-export surface for worker protocol types                           |

## Important Modules

- `src/features/style-transfer/modelPacks.ts`: pack names, URL loading, parsing,
  and cache integration.
- `src/features/pointcloud-preview/loadPointCloudMesh.ts`: JSON validation,
  typed-array conversion, bounds, and precomputed mesh vertex colours.
- `src/features/pointcloud-preview/math/kdTree3d.ts`: immutable 3D k-d tree and
  nearest-neighbour search for point-cloud samples.
- `src/features/style-transfer/modelCache.ts`: IndexedDB model-pack persistence.
- `src/features/style-transfer/kernelOptimizationSettingsStorage.ts`:
  persisted experimental kernel settings.
- `src/shared/assetUrls.ts`: Vite-base-aware asset and optional external model
  pack URLs.
- `src/ml/constants/vgg19.ts`: style/content tap indices.
- `src/ml/worker/models/vgg19/weights.ts`: manifest validation, checksum checks,
  and format decoding.
- `src/ml/worker/runtime/bufferKernels.ts`: uploads, readbacks, owned outputs,
  and reusable buffer helpers.
- `src/ml/worker/runtime/optimizationContext.ts`: per-run resource lifecycle.
- `src/ml/worker/runtime/computePipelineCache.ts`: cached WebGPU pipelines.
- `src/ml/worker/pipelines/optimization/trackedOps.ts`: pipeline operations that
  retain backward-pass state.
- `src/ml/worker/pipelines/optimization/input-optimizer/`: SGD, Adam, LBFGS,
  and CPU/GPU vector operations.
- `src/ml/worker/pipelines/optimization/layerSchedules.ts` and
  `style-transfer/vgg19Plan.ts`: fixed VGG execution order and taps.

## Tests And Fixtures

| Path                                   | Purpose                                                      |
| -------------------------------------- | ------------------------------------------------------------ |
| `tests/*.spec.ts`                      | Default correctness, app, worker, parity, and pipeline suite |
| `tests/helpers/workerClient.ts`        | Typed browser worker client and response guards              |
| `tests/helpers/browserWorkerClient.ts` | Lower-level browser worker helpers                           |
| `tests/helpers/tensorAssertions.ts`    | Numerical comparison helpers                                 |
| `tests/helpers/fixtures.ts`            | Fixture loading helpers                                      |
| `tests/helpers/fullPassArtifacts.ts`   | Full-pass fixture/model-pack selection                       |
| `tests/helpers/appPage.ts`             | Stable app navigation helpers                                |
| `benchmarks/*.spec.ts`                 | Optional kernel, first-pool, and pack performance checks     |
| `public/vgg19-first-pool/`             | Committed first-pool fixtures                                |
| `public/phase4-backprop/`              | Committed backward fixture                                   |
| `public/lbfgs/`                        | Committed optimizer fixture                                  |
| `public/vgg19-phase3-full-pass/`       | Compact tracked fixture plus ignored optional outputs        |
| `public/pointcloud-style-transfer/`    | Tiny committed mesh + point-cloud preview example            |
| `python-reference/`                    | PyTorch fixture exporters and reference implementation       |

Add a focused Playwright spec near the closest existing behavior. Reuse test
helpers and type guards instead of creating a second worker client or numerical
assertion layer.

## Configuration

- `package.json` and `package-lock.json`: npm scripts and locked dependencies.
- `tsconfig.json`, `tsconfig.app.json`, `tsconfig.node.json`: strict TypeScript
  build configuration.
- `eslint.config.js`: TypeScript and React lint rules.
- `vite.config.ts`: React/Tailwind plugins, Vite base path, and Pages SPA
  fallback generation.
- `playwright.config.ts`: default Chromium/WebGPU test suite.
- `playwright.benchmark.config.ts`: optional benchmark suite.
- `.github/workflows/ci.yml`: push-to-main verification.
- `.github/workflows/agent-review.yml`: pull-request verification.
- `.github/workflows/deploy-pages.yml`: GitHub Pages build and deployment.
- `requirements.txt`: Python reference dependencies.

## Agent Tooling

- `AGENTS.md`: compact always-loaded invariants, routing, and completion rules.
- `.agents/skills/repo-change/`: progressively disclosed implementation
  workflow, context routing, delegation, and artifact contract.
- `.agents/skills/python-reference/`: fixture/exporter routing loaded only for
  Python reference and numerical parity tasks.
- `.agents/skills/repo-review/`: independent final-diff review procedure.
- `.codex/agents/repo-explorer.toml`: read-only broad-retrieval role.
- `.codex/agents/repo-actor.toml`: bounded implementation role using compact
  task artifacts.
- `.codex/agents/repo-reviewer.toml`: read-only final-review role.
- `.agent-templates/`: committed task artifact schemas.
- `.agent-artifacts/<task-id>/`: ignored task-local state and handoff contract.
- `scripts/agent-task.sh`: initialize and validate task-scoped artifacts.
- `scripts/agent-pr-summary.sh`: draft touched-file and PR summary artifacts.
- `scripts/agent-workflow-check.sh`: validate tracked skills, agents, scripts,
  and the root instruction context budget.
- `scripts/agent-check.sh`: workflow validation plus format, lint, build, and
  default Playwright checks.

## Existing Utilities And Patterns

- Import protocol types through `src/types.ts` unless a module already uses a
  more focused internal import.
- Use discriminated unions and exhaustive `switch` statements for protocol
  variants.
- Use response type guards from the protocol/test helpers before accessing
  variant-specific fields.
- Respect `assetUrl()` and `vgg19ModelUrl()` for base paths and external hosting.
- Use runtime buffer ownership/pool/context helpers for GPU allocations.
- Follow each op family's existing split between `*.shader.ts` and `*.run.ts`.
- Keep readback-heavy debug routes separate from GPU-resident pipeline paths.
- Use `assetUrl()` for preview JSON examples so route loading respects Vite
  `BASE_URL`.
- Use explicit React state generic types and narrow literal unions.
- Tests may skip optional large assets, but committed-fixture paths should
  remain deterministic.

## Documentation

- `README.md`: user and contributor setup, commands, testing, and deployment.
- `docs/architecture.md`: short architectural contract.
- `docs/architecture-overview.md`: detailed current data flow and module roles.
- `docs/webgpu-style-transfer-plan.md`: project phase status and follow-ups.
- `docs/change-policy.md`: scope, approval, docs, test, and risk policy.
- `docs/review-rubric.md`: strict final-diff review checklist.
- `.agents/skills/repo-change/references/context-routing.md`: conditional
  documentation and source retrieval.
- `public/vgg19-models/README.md`: pack layout and manifest schema.
- `python-reference/vgg19-phase3-full-pass-README.md`: large fixture workflow.
