# Delegation

## Stay Single-Agent

Keep work local when it is a bounded change in one ownership area, retrieval is
obvious, and an independent agent would mostly repeat the same reads.

## Use `repo-explorer`

Spawn a fresh read-only explorer when one or more apply:

- likely files are uncertain;
- the task crosses at least two of UI, protocol, worker routing, runtime, ops,
  pipeline, tests, Python reference, or deployment;
- several independent repository questions can be answered in parallel;
- raw search output, logs, or large documents would pollute the actor context.

Prompt it with the refined task and specific questions. Request:

- relevant files and symbols;
- existing utilities and patterns to reuse;
- related tests and documentation;
- constraints, risks, and files that should remain out of scope.

Do not fork the full parent context. Do not ask the explorer to implement.

## Use `repo-actor`

After broad retrieval, spawn a fresh actor when keeping the implementation
context separate is worth the coordination cost. Give it only:

- `.agent-artifacts/<task-id>/task.md`;
- `.agent-artifacts/<task-id>/plan.md`;
- `.agent-artifacts/<task-id>/context-map.md`;
- explicit file or module ownership.

The actor may read directly required files, edit within scope, update
`touched-files.md`, and run focused checks. The parent remains responsible for
integration, final verification, and reviewer coordination.

## Use `repo-reviewer`

For non-trivial code/config changes, spawn a fresh read-only reviewer after
checks. Give it:

- `.agent-artifacts/<task-id>/task.md`;
- `.agent-artifacts/<task-id>/plan.md`;
- `.agent-artifacts/<task-id>/context-map.md`;
- `.agent-artifacts/<task-id>/touched-files.md`;
- the final diff and check results.

Ask it to use `$repo-review`, report findings first, and avoid editing. The
actor records the result, fixes required findings, and requests re-review.

## Parallelism

Parallelize independent read-heavy work. Avoid parallel writers unless their
file ownership is disjoint and explicitly assigned.
