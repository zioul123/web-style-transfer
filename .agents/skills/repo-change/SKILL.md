---
name: repo-change
description: Plan, implement, verify, review, and summarize non-trivial changes in the Web Style Transfer repository. Use for code, configuration, test, CI, architecture, protocol, workflow, or multi-file documentation changes that need task-scoped artifacts and repository checks. Do not use for a trivial typo or explanation-only request.
---

# Repository Change

Use focused retrieval and keep durable task state under
`.agent-artifacts/<task-id>/`.

## Start

1. Choose a short lowercase task ID containing letters, digits, and hyphens.
2. Run `./scripts/agent-task.sh init <task-id>`.
3. Refine the request in `task.md` before editing.
4. Read `docs/code-map.md`, then only the files and conditional guidance needed
   for the task. Use [context-routing.md](references/context-routing.md).
5. Record distilled findings in `context-map.md`; do not paste source blocks or
   raw search logs.

## Decide Whether To Delegate

Keep bounded work in one agent. Use fresh-context subagents when the task has
uncertain retrieval, crosses multiple ownership areas, or benefits from an
independent final review.

- Spawn `repo-explorer` read-only for broad or uncertain retrieval. Give it the
  refined task, not the parent conversation. Record only its useful summary in
  `context-map.md`.
- For broad tasks, spawn `repo-actor` with fresh context and give it `task.md`,
  `plan.md`, and `context-map.md`, rather than raw exploration output.
- After non-trivial code or configuration changes, spawn `repo-reviewer`
  read-only with the task directory and final diff. Use `$repo-review`.

See [delegation.md](references/delegation.md) for thresholds and handoff
contracts.

## Implement

1. Complete `plan.md`, including checks, documentation impact, risk, and
   rollback.
2. Make the smallest change that satisfies the acceptance criteria.
3. Preserve main-thread/worker boundaries, strict protocol typing, explicit GPU
   ownership, validation, and fixture-backed numerical behavior.
4. Update relevant documentation in the same change. If none is needed, state
   why in `plan.md` and `review.md`.
5. Maintain `touched-files.md` as scope changes.

Use `$python-reference` only for PyTorch reference, fixture, quantization, or
numerical parity work.

## Verify And Review

1. Run focused checks while implementing.
2. Always run `npm run build` for code or configuration changes.
3. Run `./scripts/agent-check.sh` before completion when feasible.
4. Run `./scripts/agent-pr-summary.sh --task <task-id> [base-ref]` to draft
   final scope and summary without overwriting completed artifacts.
5. Perform an independent `$repo-review` for non-trivial code/config changes.
6. Address required findings, rerun affected checks, and review again.
7. Complete `review.md` and `pr-summary.md`.
8. Run `./scripts/agent-task.sh check <task-id> [base-ref]`.

The artifact schema and completion rules are in
[artifact-contract.md](references/artifact-contract.md).
