# Review Rubric

Review the final diff independently from implementation. Use the task, plan,
context map, touched-file list, check results, and `git diff` as inputs. Write
the result to `.agent-artifacts/<task-id>/review.md`.

For non-trivial code or configuration changes, use the read-only
`repo-reviewer` agent with fresh context and the `$repo-review` skill. A bounded
documentation-only change may use a justified self-review.

## Result Rules

- **Fail:** any required finding, unmet acceptance criterion, unexplained check
  failure/skip, or unresolved high-risk concern.
- **Pass:** no required findings remain, checks are adequate for the scope, and
  risks are explicitly documented.
- Optional suggestions must not expand the current task unless they prevent a
  concrete defect. Put broader ideas in follow-up work.

## Required Checklist

### Scope And Reuse

- [ ] The change is no larger than necessary for the refined task.
- [ ] Unrelated refactors, renames, formatting churn, and generated output are
      absent.
- [ ] Existing utilities, components, protocol helpers, test helpers, and GPU
      runtime abstractions were reused instead of duplicated.
- [ ] Every changed file appears in `touched-files.md` with a valid reason.

Reject changes that duplicate an existing abstraction or bypass it without a
documented constraint.

### Behavior, Types, And Errors

- [ ] Acceptance criteria and important edge cases are covered.
- [ ] Public behavior changes have focused tests and documentation.
- [ ] Types were not weakened with broad objects, optional fields, `any`, or
      unnecessary casts.
- [ ] Discriminated unions, type guards, and exhaustive handling remain intact.
- [ ] Error handling and validation were not removed, swallowed, or made less
      actionable.

Reject untested public behavior changes, weakened types, or weakened error
handling.

### Architecture

- [ ] React presentation, controller orchestration, typed protocol, worker
      routing, runtime infrastructure, operations, and pipelines remain within
      their documented boundaries.
- [ ] WebGPU buffer ownership and cleanup remain explicit.
- [ ] Numerical changes preserve fixture-based parity or explain intentional
      tolerance changes.
- [ ] New abstractions remove real duplication and fit established patterns.

Reject architecture bypasses and unexplained ownership changes.

### Verification And Documentation

- [ ] Focused tests were added or updated where required.
- [ ] Changed-file formatting validation, `npm run lint`, `npm run build`, and
      `npm test` were run through `scripts/agent-check.sh`, or each skip/failure
      is justified.
- [ ] Performance-sensitive changes include benchmark context and correctness
      evidence.
- [ ] Relevant user, architecture, protocol, setup, fixture, or workflow docs
      were updated.
- [ ] The final diff contains no accidental `.agent-artifacts/`, `dist/`,
      `test-results/`, optional generated fixtures, or unrequested model packs.

Reject skipped relevant tests without justification and accidental generated
artifacts.

### Security, Privacy, And Operations

- [ ] No secrets, credentials, private paths, uploaded image data, or sensitive
      model payloads were added to source, logs, artifacts, or docs.
- [ ] External URLs, manifests, shards, browser storage, and worker messages
      retain appropriate validation.
- [ ] Auth, migration, deletion, cache invalidation, deployment permission, and
      public API changes have explicit approval and rollback notes.
- [ ] New dependencies are approved and justified.

Reject unresolved security/privacy risk or unapproved high-risk changes.

### PR Summary

- [ ] The title and summary match the actual final diff.
- [ ] Changes and checks are specific.
- [ ] Risks/hazards are explicitly stated, including `None known` when that is
      the honest assessment.
- [ ] Rollback notes are practical.
- [ ] Follow-up work is separated from required work.

Reject a PR summary that omits risks/hazards or conceals failed/skipped checks.

## Finding Format

List required findings first, ordered by severity:

```text
[P0-P3] Short title - path:line
Impact and evidence.
Required correction.
```

Use:

- `P0`: immediate security, privacy, data-loss, or repository integrity risk;
- `P1`: likely broken behavior, contract break, or major regression;
- `P2`: bounded correctness, maintainability, test, or documentation issue;
- `P3`: minor issue worth fixing before merge.

When there are no findings, say so explicitly and record any residual test or
environment risk.

## Retry Guidance

For a failed review:

1. list only required corrections;
2. identify the smallest files/functions to revisit;
3. name focused checks to rerun;
4. state what must remain out of scope;
5. implement the corrections;
6. review the new diff again and replace the stale result.
