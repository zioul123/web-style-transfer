---
name: artifact-pr-summary
description: Draft a pull request description or branch summary by reading prior task folders under `.agent-artifacts/` and synthesizing the branch diff, checks, risks, and follow-ups. Use when the user asks for a PR summary, PR description, branch write-up, or changelog based on recent agent runs, task artifacts, or named task IDs.
---

# Artifact PR Summary

Draft one coherent PR description from prior repo task artifacts. Prefer
evidence from the branch diff and the task folders over guesswork.

## Inputs

Start with:

1. `docs/code-map.md` for repo navigation.
2. The current branch name and branch diff against the requested base, usually
   `origin/main`.
3. The relevant task folders under `.agent-artifacts/`.

If the user names the task IDs, use those IDs directly.

If the user does not name the task IDs:

1. Inspect `.agent-artifacts/` for the most recent task folders.
2. Suggest a short candidate list when the intended set is ambiguous.
3. Skip the suggestion and proceed when one grouping is obvious from the branch
   history, task names, or the user's wording.

## Read Narrowly

For each selected task folder, prefer this order:

1. `pr-summary.md`
2. `task.md`
3. `touched-files.md`
4. `review.md`
5. `plan.md` only if checks, risks, or scope are still unclear

Use the branch diff to reconcile per-task notes with the actual final scope:

- `git log --oneline --reverse <base>..HEAD`
- `git diff --stat <base>..HEAD`
- targeted `git diff` for files whose role or final behavior is unclear

Do not blindly concatenate task summaries. Remove artifact-only noise and merge
overlapping task notes into one branch-level story.

## Synthesis Rules

Write for the branch as it exists now, not for each intermediate run.

- Keep the top summary short and outcome-focused.
- Group changes by user-facing capability or technical theme.
- Mention validation that actually ran on the branch or in the selected tasks.
- Call out known risks, fallback behavior, or follow-up work when the artifacts
  mention them.
- Distinguish branch-owned files from files that were only referenced in a task
  artifact because they were already changed elsewhere on the branch.

When branch scope exceeds the named task folders, say so and decide whether to:

1. expand the read to include the missing artifact folders, or
2. tell the user the write-up covers only the requested subset.

## Output

Produce paste-ready Markdown for a GitHub PR description. Usually include:

- `## Summary`
- `## What changed`
- `## Validation`
- `## Notes` or `## Risks`
- `## Follow-ups`

Adapt section names to the repo's norms or the user's request.

## Guardrails

- Prefer concise prose over a file-by-file changelog.
- Preserve important caveats from the artifacts, especially skipped or flaky
  checks.
- Do not claim tests passed unless an artifact or command output says so.
- If artifact evidence conflicts with the current diff, trust the current diff
  and mention the discrepancy only when it materially affects the PR summary.
