# Artifact Contract

Store task state in `.agent-artifacts/<task-id>/`.

Required files:

- `task.md`: original request, bounded outcome, assumptions, acceptance
  criteria, and exclusions.
- `plan.md`: implementation steps, likely files, checks, documentation impact,
  risks, and rollback.
- `context-map.md`: distilled files, symbols, reusable patterns, tests, docs,
  and constraints.
- `touched-files.md`: every changed file with a specific reason and category.
- `review.md`: pass/fail result, findings, risks, checks, and retry plan.
- `pr-summary.md`: final title, outcome, changes, checks, risks, rollback, and
  follow-up work.

Rules:

- Keep artifacts compact and current; they are state, not transcripts.
- Never include secrets, private user data, copied source files, model weights,
  or large fixture payloads.
- Replace all template comments and `TODO` markers before completion.
- A passing review must say `**Result:** pass`.
- Every changed tracked file must appear in `touched-files.md`.
- Check off every acceptance criterion before completion.
- Keep follow-up work separate from required work.
