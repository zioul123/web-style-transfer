# Change Policy

This policy keeps agent-authored changes reviewable, reversible, and aligned
with the current architecture.

## Small Changes

A small change:

- has one clear outcome and a bounded set of acceptance criteria;
- changes only the files and layers needed for that outcome;
- preserves existing APIs, protocol variants, data formats, and architecture
  unless the task explicitly changes them;
- reuses current helpers, components, test infrastructure, and runtime
  ownership patterns;
- includes focused tests and documentation where behavior changes;
- can be reverted without a migration or data repair.

Examples include a focused UI correction, one typed worker variant plus its
router/test updates, a kernel bug fix with parity coverage, or a documentation
and workflow improvement.

Splitting a task into independently reviewable changes is preferred when its
implementation spans unrelated UI, protocol, runtime, and deployment concerns.

## Explicit Human Approval

Obtain explicit approval before:

- broad rewrites, framework swaps, large file moves, or architecture changes;
- adding, removing, or upgrading dependencies;
- migrations or persistent data-format changes, including IndexedDB records or
  cache invalidation that may discard user data;
- authentication, authorization, credentials, secrets, or permission changes;
- deletion of user data, model caches, generated assets, fixtures, or tracked
  files;
- security-sensitive parsing, external asset trust, checksum validation, CSP,
  cross-origin behavior, or deployment permissions;
- public API, worker protocol, model-pack manifest, fixture format, URL, or
  command changes that were not clearly requested;
- committing large generated fixtures, model packs, build outputs, or benchmark
  results;
- knowingly changing numerical tolerances, optimizer semantics, or performance
  defaults without evidence and an explicit task requirement.

Approval for the requested outcome does not imply approval for unrelated
cleanup or a broader implementation strategy.

## Documentation

Update documentation in the same change when any of these change:

- user-visible behavior or controls;
- architecture, ownership, or runtime data flow;
- worker requests/responses or model/fixture formats;
- setup, environment, npm/Python commands, tests, or CI;
- deployment, routes, base paths, or external model hosting;
- benchmark methodology, supported packs, or phase status.

Use the narrowest relevant document. Update `README.md` for contributor-facing
commands, `docs/architecture*.md` for boundaries, `docs/code-map.md` for moved
entry points, and format-specific READMEs for generated assets.

## Tests

Add or update tests when:

- behavior changes or a bug is fixed;
- a protocol variant, type guard, router branch, or error path changes;
- tensor shapes, numerical logic, kernels, optimizers, or tolerances change;
- model loading, caching, route behavior, or UI controls change;
- a regression could be observed through an existing Playwright helper.

Use the closest existing test and fixture. Add a new fixture only when a small
deterministic case cannot cover the behavior. Performance changes require
correctness coverage plus an appropriate benchmark result or explanation.

Tests may be omitted for documentation-only or non-executable template changes.
Record the reason in `.agent-artifacts/<task-id>/review.md` and the PR summary.

## Dependencies

Do not introduce a dependency when the platform, standard library, current
dependencies, or a small local implementation already covers the need.

When a new dependency is genuinely necessary:

1. obtain approval;
2. explain why existing options are insufficient;
3. assess maintenance, bundle size, browser support, license, and security;
4. use npm and update `package-lock.json`;
5. add tests and documentation;
6. mention the dependency and rollback path in the PR summary.

## High-Risk Changes

### Migrations And Persistence

The current app has no server database, but it does persist model-pack cache
records and settings in browser storage. Version persisted formats, preserve
backward compatibility where feasible, and provide a rollback or safe
invalidation plan. Never silently delete stored data merely to simplify code.

### Authentication And Authorization

The repository currently has no authentication system. Do not add one or
handle credentials without an explicitly approved design, threat model, and
test plan.

### Data Deletion

Require a deliberate user action, clear scope, and recoverability assessment.
Keep deletion paths separate from read/update paths and test failure behavior.

### Security-Sensitive Code

Treat model manifests, shard URLs, uploads, browser storage, checksums, worker
messages, and external hosting inputs as untrusted boundaries. Preserve
validation, avoid logging private content, and document new trust assumptions.

### Public Contracts

Worker protocol unions, model-pack manifests, fixture schemas, URL conventions,
and documented commands are public contracts for this repository. A contract
change must update producers, consumers, exhaustive routing, tests, and docs in
one change. Prefer additive compatibility unless the task explicitly permits a
breaking change.
