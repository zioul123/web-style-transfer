#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  echo "Usage:" >&2
  echo "  $0 [base-ref] [--force]" >&2
  echo "  $0 --task <task-id> [base-ref] [--force]" >&2
}

TASK_ID=""
BASE_REF="origin/main"
FORCE=""

if [[ "${1:-}" == "--task" ]]; then
  TASK_ID="${2:-}"
  BASE_REF="${3:-origin/main}"
  FORCE="${4:-}"
  if [[ -z "$TASK_ID" || "$#" -gt 4 ]]; then
    usage
    exit 1
  fi
else
  BASE_REF="${1:-origin/main}"
  FORCE="${2:-}"
  if [[ "$#" -gt 2 ]]; then
    usage
    exit 1
  fi
fi

if [[ "$BASE_REF" == "--force" ]]; then
  FORCE="--force"
  BASE_REF="origin/main"
fi

if [[ "$FORCE" != "" && "$FORCE" != "--force" ]]; then
  usage
  exit 1
fi

if [[ -n "$TASK_ID" && ! "$TASK_ID" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo "Task ID must contain lowercase letters, digits, and hyphens." >&2
  exit 1
fi

if [[ -n "$TASK_ID" ]]; then
  ARTIFACT_DIR=".agent-artifacts/$TASK_ID"
else
  ARTIFACT_DIR=".agent-artifacts"
fi

TOUCHED_FILE="$ARTIFACT_DIR/touched-files.md"
SUMMARY_FILE="$ARTIFACT_DIR/pr-summary.md"

if [[ -n "$TASK_ID" && ! -d "$ARTIFACT_DIR" ]]; then
  echo "Task artifacts do not exist. Run: ./scripts/agent-task.sh init $TASK_ID" >&2
  exit 1
fi

is_template_file() {
  local path="$1"
  [[ ! -e "$path" ]] ||
    grep -q '<!--' "$path" ||
    grep -Fq 'Specific reason this task requires the file' "$path"
}

if [[ "$FORCE" != "--force" ]] &&
  { ! is_template_file "$TOUCHED_FILE" || ! is_template_file "$SUMMARY_FILE"; }; then
  echo "Refusing to overwrite completed agent artifacts." >&2
  echo "Review them first, or rerun with --force." >&2
  exit 1
fi

mkdir -p "$ARTIFACT_DIR"
CHANGED_FILES="$(mktemp)"
SORTED_FILES="$(mktemp)"
trap 'rm -f "$CHANGED_FILES" "$SORTED_FILES"' EXIT

if git rev-parse --verify "$BASE_REF" >/dev/null 2>&1; then
  if git merge-base "$BASE_REF" HEAD >/dev/null 2>&1; then
    git diff --name-only "$BASE_REF"...HEAD >>"$CHANGED_FILES"
  else
    git diff --name-only "$BASE_REF" HEAD >>"$CHANGED_FILES"
  fi
else
  echo "Warning: base ref '$BASE_REF' was not found; using working-tree changes only." >&2
fi

git diff --name-only >>"$CHANGED_FILES"
git diff --cached --name-only >>"$CHANGED_FILES"
git ls-files --others --exclude-standard >>"$CHANGED_FILES"
sed '/^\.agent-artifacts\//d; /^$/d' "$CHANGED_FILES" | sort -u >"$SORTED_FILES"

classify_file() {
  local path="$1"
  case "$path" in
    tests/* | benchmarks/* | *.spec.ts) echo "test" ;;
    docs/* | README.md | AGENTS.md | .agent-templates/* | .agents/skills/* | *.md) echo "docs" ;;
    .github/* | .codex/* | scripts/* | package.json | package-lock.json | tsconfig*.json | *.config.* | .gitignore)
      echo "config"
      ;;
    dist/* | test-results/* | .agent-artifacts/*) echo "generated" ;;
    *) echo "production" ;;
  esac
}

{
  printf '# Touched Files\n\n'
  printf '| File | Reason | Category |\n'
  printf '| --- | --- | --- |\n'
  if [[ -s "$SORTED_FILES" ]]; then
    while IFS= read -r path; do
      category="$(classify_file "$path")"
      printf '| `%s` | TODO: state why this task requires the file | %s |\n' \
        "$path" "$category"
    done <"$SORTED_FILES"
  else
    printf '| _None detected_ | Confirm the intended diff and base ref | config |\n'
  fi
} >"$TOUCHED_FILE"

{
  printf '# PR Summary\n\n'
  printf '## Title\n\n'
  printf 'TODO: concise imperative title\n\n'
  printf '## Summary\n\n'
  printf 'TODO: describe the outcome and why it is needed.\n\n'
  printf '## Changes Made\n\n'
  if [[ -s "$SORTED_FILES" ]]; then
    while IFS= read -r path; do
      printf -- '- `%s`: TODO: describe the final change.\n' "$path"
    done <"$SORTED_FILES"
  else
    printf -- '- TODO: no changed files were detected.\n'
  fi
  printf '\n## Tests And Checks Run\n\n'
  printf -- '- `./scripts/agent-check.sh` - TODO: pass, fail, or skipped with reason\n\n'
  printf '## Risks And Hazards\n\n'
  printf -- '- TODO: state concrete risks/hazards or `None known`.\n\n'
  printf '## Rollback Notes\n\n'
  printf 'TODO: describe the smallest revert and any data implications.\n\n'
  printf '## Follow-Up Work\n\n'
  printf -- '- TODO: list deferred work or `None`.\n'
} >"$SUMMARY_FILE"

echo "Created $TOUCHED_FILE and $SUMMARY_FILE"
echo "Base ref: $BASE_REF"
echo "Replace every TODO before task validation or PR handoff."
