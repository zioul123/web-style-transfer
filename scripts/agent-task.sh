#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  echo "Usage:" >&2
  echo "  $0 init <task-id>" >&2
  echo "  $0 check <task-id> [base-ref]" >&2
}

validate_task_id() {
  local task_id="$1"
  if [[ ! "$task_id" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
    echo "Task ID must contain lowercase letters, digits, and hyphens." >&2
    exit 1
  fi
}

changed_files() {
  local base_ref="$1"
  local output_file="$2"

  if git rev-parse --verify "$base_ref" >/dev/null 2>&1; then
    if git merge-base "$base_ref" HEAD >/dev/null 2>&1; then
      git diff --name-only "$base_ref"...HEAD >>"$output_file"
    else
      git diff --name-only "$base_ref" HEAD >>"$output_file"
    fi
  else
    echo "Warning: base ref '$base_ref' was not found; checking working-tree changes only." >&2
  fi

  git diff --name-only >>"$output_file"
  git diff --cached --name-only >>"$output_file"
  git ls-files --others --exclude-standard >>"$output_file"
}

init_task() {
  local task_id="$1"
  local task_dir=".agent-artifacts/$task_id"

  if [[ -e "$task_dir" ]]; then
    echo "Refusing to overwrite existing task artifacts: $task_dir" >&2
    exit 1
  fi

  mkdir -p "$task_dir"
  cp .agent-templates/task-template.md "$task_dir/task.md"
  cp .agent-templates/plan-template.md "$task_dir/plan.md"
  cp .agent-templates/context-map-template.md "$task_dir/context-map.md"
  cp .agent-templates/touched-files-template.md "$task_dir/touched-files.md"
  cp .agent-templates/review-template.md "$task_dir/review.md"
  cp .agent-templates/pr-summary-template.md "$task_dir/pr-summary.md"

  echo "Created task artifacts in $task_dir"
}

check_task() {
  local task_id="$1"
  local base_ref="${2:-origin/main}"
  local task_dir=".agent-artifacts/$task_id"
  local required_files=(
    task.md
    plan.md
    context-map.md
    touched-files.md
    review.md
    pr-summary.md
  )
  local file
  local changed
  local changed_file_list
  local sorted_file_list

  if [[ ! -d "$task_dir" ]]; then
    echo "Task artifact directory does not exist: $task_dir" >&2
    exit 1
  fi

  for file in "${required_files[@]}"; do
    if [[ ! -s "$task_dir/$file" ]]; then
      echo "Missing or empty task artifact: $task_dir/$file" >&2
      exit 1
    fi
  done

  if grep -R -n -E 'TODO|<!--[[:space:]]*[^>]' "$task_dir" >/dev/null; then
    echo "Task artifacts still contain template placeholders:" >&2
    grep -R -n -E 'TODO|<!--[[:space:]]*[^>]' "$task_dir" >&2
    exit 1
  fi

  if grep -n -- '- \[ \]' "$task_dir/task.md" >/dev/null; then
    echo "Task acceptance criteria are not all checked." >&2
    exit 1
  fi

  if ! grep -Fq '**Result:** pass' "$task_dir/review.md"; then
    echo "Review artifact does not record a passing result." >&2
    exit 1
  fi

  changed_file_list="$(mktemp)"
  sorted_file_list="$(mktemp)"
  trap "rm -f '$changed_file_list' '$sorted_file_list'" EXIT
  changed_files "$base_ref" "$changed_file_list"
  sed '/^\.agent-artifacts\//d; /^$/d' "$changed_file_list" | sort -u >"$sorted_file_list"

  while IFS= read -r changed; do
    if ! grep -Fq "\`$changed\`" "$task_dir/touched-files.md"; then
      echo "Changed file missing from touched-files.md: $changed" >&2
      exit 1
    fi
  done <"$sorted_file_list"

  if git ls-files '.agent-artifacts/*' |
    grep -v -F '.agent-artifacts/.gitkeep' |
    grep -q .; then
    echo "Generated task artifacts must not be tracked." >&2
    exit 1
  fi

  echo "Task artifacts passed validation: $task_dir"
}

command="${1:-}"
task_id="${2:-}"

if [[ -z "$command" || -z "$task_id" ]]; then
  usage
  exit 1
fi

validate_task_id "$task_id"

case "$command" in
  init)
    if [[ "$#" -ne 2 ]]; then
      usage
      exit 1
    fi
    init_task "$task_id"
    ;;
  check)
    if [[ "$#" -gt 3 ]]; then
      usage
      exit 1
    fi
    check_task "$task_id" "${3:-origin/main}"
    ;;
  *)
    usage
    exit 1
    ;;
esac
