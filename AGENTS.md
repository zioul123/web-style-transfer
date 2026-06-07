# Repository Agent Guide

This file is the operating contract for coding agents working in this
repository. For every non-trivial task, use the staged workflow below and keep
task-specific context in `.agent-artifacts/` rather than in a growing chat
history.

## Required Context

Before planning a change, read:

1. `README.md`
2. `docs/webgpu-style-transfer-plan.md`
3. `docs/architecture.md`
4. `docs/code-map.md`
5. `docs/change-policy.md`
6. `docs/review-rubric.md`

Read only the source files, tests, and focused documentation needed for the
current task after that baseline. Do not scan generated assets or large model
packs unless the task requires them.

## Project Overview

Web Style Transfer is a browser-native Gatys neural style-transfer application
built with React 19, TypeScript, Vite, Tailwind CSS, Web Workers, and WebGPU.
The React main thread owns controls, model loading, and previews. A typed worker
protocol connects it to worker-owned WebGPU runtime code, kernels, and fixed
VGG19 optimization pipelines. Playwright exercises the app, worker protocol,
GPU operations, and PyTorch parity fixtures.

The code is correctness-first. Keep the main-thread/worker boundary, typed
protocol, explicit GPU buffer ownership, and fixture-based numerical parity
intact.

## Important Directories

| Path                                    | Responsibility                                                                          |
| --------------------------------------- | --------------------------------------------------------------------------------------- |
| `src/features/style-transfer/`          | Main-thread controller, UI components, model-pack loading, cache, and benchmark helpers |
| `src/types/worker-protocol/`            | Source of truth for typed worker request and response contracts                         |
| `src/ml/worker/main-thread-protocol/`   | Worker request routing and response helpers                                             |
| `src/ml/worker/runtime/`                | WebGPU device, buffers, pipeline cache, shapes, and runtime ownership helpers           |
| `src/ml/worker/ops/`                    | WebGPU operation runners and WGSL shader generators                                     |
| `src/ml/worker/pipelines/optimization/` | First-pool and full style-transfer optimization pipelines                               |
| `tests/`                                | Default Playwright integration and parity suite                                         |
| `benchmarks/`                           | Optional performance and kernel-lab Playwright suite                                    |
| `python-reference/`                     | PyTorch reference implementation and fixture exporters                                  |
| `public/`                               | Runtime model packs and committed test fixtures                                         |
| `docs/`                                 | Architecture, plan, policy, review, and code-map documentation                          |
| `.agent-templates/`                     | Committed templates for staged agent artifacts                                          |
| `.agent-artifacts/`                     | Ignored, task-specific handoff artifacts; only `.gitkeep` is committed                  |

See `docs/code-map.md` for entry points and task-to-module guidance.

## Environment And Commands

Use Node.js 22. Before running npm commands locally:

```bash
nvm use 22
```

Use the repository virtual environment before Python commands:

```bash
source .venv/bin/activate
```

Common commands:

```bash
npm ci
npm run dev
npm run format
npm run format:check
npm run lint
npm run build
npm test
npm run benchmark
./scripts/agent-check.sh
./scripts/agent-pr-summary.sh
```

`npm run build` is the TypeScript typecheck plus production Vite build. The
default test suite is Playwright-based; there is no separate unit-test command.
`npm run benchmark` is optional and should be selected for performance-sensitive
changes, not run as a routine correctness check.

If Chromium or its Linux libraries are missing:

```bash
npx playwright install chromium
npx playwright install-deps chromium
```

Some parity tests require optional fixtures. Generate the phase-3 fixture only
when the task needs it:

```bash
source .venv/bin/activate
python python-reference/export_vgg19_phase3_full_pass.py
```

Do not commit large generated fixtures or model packs unless they were already
tracked or the task explicitly requires them.

## Change Workflow

For non-trivial work, follow these stages in order. One agent may perform all
stages, or separate agents may consume the artifacts as handoff contracts.

1. **Refine:** Create `.agent-artifacts/task.md` from
   `.agent-templates/task-template.md`. Turn the rough request into testable
   acceptance criteria and state assumptions and exclusions.
2. **Plan:** Create `.agent-artifacts/plan.md` from
   `.agent-templates/plan-template.md`. Identify likely files/functions, steps,
   checks, docs, risks, and rollback before editing.
3. **Retrieve:** Read only the relevant code and record the useful result in
   `.agent-artifacts/context-map.md`. Include existing abstractions and tests to
   reuse; do not paste large source blocks.
4. **Implement:** Make the smallest safe change that satisfies the acceptance
   criteria. Prefer existing patterns, helpers, components, protocol types, and
   runtime ownership utilities.
5. **Record scope:** Maintain `.agent-artifacts/touched-files.md`, including a
   reason and category for every changed file.
6. **Verify:** Run focused checks during development, then
   `./scripts/agent-check.sh` before completion when feasible. Always run
   `npm run build` before concluding a code or configuration change.
7. **Review:** Inspect the final diff against `docs/review-rubric.md` and write
   `.agent-artifacts/review.md`. The review must consider task, plan, context
   map, touched files, diff, and check results.
8. **Retry:** If checks or review fail, write a short scoped retry plan in the
   review artifact, correct only the identified issues, rerun affected checks,
   and review the new diff again.
9. **Summarize:** Create `.agent-artifacts/pr-summary.md` from the template or
   `./scripts/agent-pr-summary.sh`. Replace all placeholders and include tests,
   risks/hazards, rollback notes, and follow-up work.
10. **PR upkeep:** If an attached pull request exists, update its description
    with the final PR summary.

For a truly trivial documentation or typo change, the artifacts may be shorter,
but scope, verification, and review expectations still apply.

## Artifact Workflow

- Commit templates in `.agent-templates/`.
- Do not commit generated files under `.agent-artifacts/`; only `.gitkeep` is
  tracked.
- Treat artifacts as compact state, not transcripts. Summarize decisions,
  symbols, constraints, and command outcomes.
- Keep artifacts current when the plan or scope changes.
- Never place secrets, credentials, user data, full model weights, large fixture
  payloads, or copied source files in artifacts.
- A handoff should be possible from the artifacts plus the referenced files,
  without reconstructing the entire conversation.

## Small-Change Policy

Follow `docs/change-policy.md`.

- Change the fewest files and smallest behavioral surface needed.
- Avoid unrelated renames, formatting churn, dependency updates, generated
  output, and opportunistic refactors.
- Reuse existing abstractions before introducing a helper or component.
- Keep UI, protocol, worker routing, runtime, ops, and pipeline responsibilities
  within their established boundaries.
- Ask for explicit human approval before broad rewrites, new dependencies,
  destructive data changes, security-sensitive behavior, migrations, or public
  contract changes not already requested.

## TypeScript Style

- Use explicit types for React state, for example
  `useState<string>("value")`.
- Prefer narrow discriminated unions over objects with optional fields.
- Prefer type guards and exhaustiveness checks over type casts.
- Preserve strict request/response typing across the worker boundary.
- Add concise comments only where ownership, numerical behavior, or a
  non-obvious constraint would otherwise be difficult to understand.

## Documentation Update Policy

Update documentation in the same change when behavior, architecture, worker
protocol/API, setup, commands, fixtures, model-pack formats, deployment, or
developer workflow changes.

- Update `docs/architecture.md` or `docs/architecture-overview.md` for boundary
  and data-flow changes.
- Update `docs/code-map.md` when important entry points or ownership move.
- Update `docs/webgpu-style-transfer-plan.md` when phase status or planned work
  changes.
- Update fixture/model-pack READMEs when their formats or generation steps
  change.
- Update `README.md` for user-facing setup, commands, or behavior.

State why no docs were needed when a change appears externally visible but does
not require documentation.

## Review Policy

Use `docs/review-rubric.md`. A passing review requires:

- acceptance criteria are satisfied;
- the diff is no larger than necessary;
- existing utilities and components are reused;
- types and error handling are not weakened;
- architectural boundaries are preserved;
- relevant tests and docs are present;
- checks are recorded, including justified skips;
- security, privacy, performance, generated-artifact, and rollback risks are
  assessed;
- the PR summary explicitly names risks/hazards, even when none are known.

The reviewer should report findings before suggestions. Any required finding
means `fail` until corrected and re-reviewed.

## Definition Of Done

A non-trivial task is complete only when:

- task, plan, context map, and touched-file artifacts reflect the final scope;
- implementation is minimally scoped and follows existing patterns;
- relevant tests were added or updated;
- documentation was updated when required;
- focused checks pass;
- `npm run build` passes;
- the strict review passes;
- the PR summary contains changes, checks, risks/hazards, rollback notes, and
  follow-up work;
- generated fixtures, build output, and agent artifacts are not accidentally
  staged.

## Safety Rules

- Do not revert or overwrite unrelated user changes.
- Do not run destructive Git commands, delete data, clear caches, or regenerate
  large assets unless the task requires it and approval is clear.
- Do not commit or push unless explicitly asked.
- Do not install or upgrade dependencies unless absolutely necessary and
  approved.
- Do not bypass checksum, shape, protocol, or error validation to make tests
  pass.
- Do not weaken WebGPU buffer cleanup or ownership rules.
- Do not expose local paths, secrets, model data, uploaded images, or other
  private information.
- Treat auth, persistence migrations, data deletion, deployment permissions,
  external asset hosting, and public worker protocol changes as high-risk.
- Stop and request clarification when a safe, minimally scoped interpretation
  cannot be inferred from the repository and task.
