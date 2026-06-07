# Repository Agent Contract

Keep this file compact because it is loaded for every task. Detailed procedures
live in repo skills and are loaded only when relevant.

## Project Invariants

Web Style Transfer is a correctness-first React/TypeScript/WebGPU application.

- The React main thread owns controls, model loading, and previews.
- `src/types/worker-protocol/` is the typed main-thread/worker contract.
- The worker owns WebGPU execution, buffers, kernels, and optimization.
- GPU buffer ownership and cleanup remain explicit.
- Numerical behavior is anchored to deterministic PyTorch fixtures.
- Do not bypass checksum, shape, protocol, validation, or error handling.

## Context Routing

- Start repository exploration with `docs/code-map.md`.
- Read source, tests, and focused docs needed for the task; do not preload every
  project document or generated asset.
- Use `$repo-change` for non-trivial code, config, test, CI, workflow, or
  multi-file documentation changes.
- Use `$python-reference` only for PyTorch reference, fixture, quantization, or
  numerical parity work.
- Use `$repo-review` for independent final-diff review.
- Architecture and risk policies are in `docs/architecture.md` and
  `docs/change-policy.md`; load them when the changed surface requires them.

## Delegation

- Keep bounded single-area work in one agent.
- For uncertain or cross-area retrieval, spawn the read-only `repo-explorer`
  with fresh context.
- For broad implementation, pass task, plan, and the compact context map to a
  fresh `repo-actor` instead of forwarding raw exploration history.
- For non-trivial code/config changes, use the read-only `repo-reviewer` after
  checks and before completion.
- Parallelize independent read-heavy work. Avoid overlapping writers.

## Environment

Use Node.js 22 before npm commands:

```bash
nvm use 22
```

Activate the repository environment before Python commands:

```bash
source .venv/bin/activate
```

Common checks:

```bash
npm run build
npm test
./scripts/agent-check.sh
```

`npm run benchmark` is optional and reserved for performance-sensitive work.

## Change Rules

- Make the smallest change that satisfies explicit acceptance criteria.
- Reuse existing helpers, components, protocol types, test utilities, and
  runtime ownership patterns.
- Preserve strict discriminated unions, type guards, and exhaustive routing.
- Add focused tests for behavior changes.
- Update the narrowest relevant documentation in the same change. State why
  docs are unnecessary when that decision is not obvious.
- Ask before broad rewrites, dependency changes, destructive data changes,
  migrations, security-sensitive behavior, or unrequested public-contract
  changes.
- Do not commit or push unless explicitly asked.
- Do not revert unrelated user changes or commit generated model assets unless
  requested.

## Completion

For non-trivial work:

1. Use task-scoped artifacts under `.agent-artifacts/<task-id>/`.
2. Keep task, plan, context, and touched-file state current.
3. Run focused checks and always run `npm run build` for code/config changes.
4. Run `./scripts/agent-check.sh` when feasible.
5. Pass independent review and resolve required findings.
6. Complete risks, rollback, checks, and follow-up work in the PR summary.
7. Update an attached PR description when applicable.
