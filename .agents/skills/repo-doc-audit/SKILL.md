---
name: repo-doc-audit
description: Audit a Web Style Transfer diff, branch, or completed task for missing documentation updates, then patch only the narrow docs required. Use when the user asks whether docs are up to date, wants missing docs added after implementation, or needs a docs-only follow-up for an existing change.
---

# Repository Documentation Audit

Use this skill for documentation coverage checks on existing repository changes.

## Start From The Change, Not The Docs

1. Read `docs/change-policy.md` and `docs/code-map.md`.
2. Identify the actual change surface from the user request plus one of:
   - the current diff (`git diff --name-only <base>...HEAD` or working tree);
   - task artifacts under `.agent-artifacts/<task-id>/` when they already
     exist;
   - a provided PR description or touched-file list.
3. Treat task artifacts as accelerators, not prerequisites. If the user did not
   provide them, infer scope from the diff, changed files, source, tests, and
   existing docs.

## Audit Workflow

1. Map changed behavior to documentation categories from
   `docs/change-policy.md`:
   - user-visible behavior or controls;
   - architecture, ownership, or runtime flow;
   - routes, commands, setup, CI, or deployment;
   - fixture, model-pack, or other documented formats.
2. Read only the narrow code, tests, and docs needed to verify those surfaces.
3. Check whether the current docs already cover the shipped behavior. Look for
   stale descriptions, missing modules, omitted routes, missing asset READMEs,
   or docs that still describe an earlier iteration of the feature.
4. Update only the narrowest relevant docs. Prefer:
   - `README.md` for contributor-facing commands, route usage, and high-level
     user-visible behavior;
   - `docs/architecture*.md` for ownership and data-flow changes;
   - `docs/code-map.md` for new entry points, modules, or asset/doc locations;
   - format- or asset-specific READMEs for committed generated assets.
5. If no documentation changes are needed, state why in the task artifacts or
   final summary.

## Scope Rules

- Do not broaden into general cleanup.
- Do not restate implementation details that are not stable or user-relevant.
- Preserve the current architecture boundaries; docs should clarify them, not
  redefine them.
- Anchor every doc change to behavior that exists in source or tests on the
  current branch.

## Verification

- For docs-only changes, self-review the diff for accuracy and scope.
- If you are using task artifacts, keep `task.md`, `plan.md`, `context-map.md`,
  `touched-files.md`, `review.md`, and `pr-summary.md` current and run
  `./scripts/agent-task.sh check <task-id> HEAD` before completion.
- Build or browser checks are not required for documentation-only changes
  unless the task also changes executable code.
