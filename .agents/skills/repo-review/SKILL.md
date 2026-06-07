---
name: repo-review
description: Independently review a Web Style Transfer repository diff for correctness, regressions, missing tests or docs, architecture violations, unsafe scope, and incomplete task artifacts. Use for final review of non-trivial code/config changes, commit reviews, PR reviews, or when asked to assess a branch or diff.
---

# Repository Review

Review as an independent, read-only owner. Do not implement fixes.

## Inputs

Read:

1. the refined task and acceptance criteria;
2. the plan and context map;
3. the touched-file list;
4. `docs/review-rubric.md`;
5. the final diff;
6. recorded check results.

Load architecture or change policy only when the diff touches those concerns.

## Review

1. Verify acceptance criteria against actual behavior and tests.
2. Check that the diff is no larger than necessary and reuses existing
   abstractions.
3. Check types, validation, errors, protocol exhaustiveness, GPU ownership, and
   numerical parity where relevant.
4. Check documentation impact and recorded test coverage.
5. Check security, privacy, generated assets, dependencies, migrations,
   deployment, and rollback risk.
6. Verify every changed file is justified in `touched-files.md`.
7. Treat missing evidence or unexplained skipped checks as findings.

## Output

List required findings first, ordered by severity, with file and line
references:

```text
[P0-P3] Short title - path:line
Impact and evidence.
Required correction.
```

Then state:

- pass or fail;
- residual risks or test gaps;
- optional follow-up suggestions that do not expand the current task.

Any required finding means fail until corrected and re-reviewed.
