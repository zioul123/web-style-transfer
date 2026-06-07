#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

required_skills=(repo-change python-reference repo-review)
readonly_agents=(repo-explorer repo-reviewer)
workflow_diagram="$(
  sed -n \
    '/<!-- agent-workflow-diagram:start -->/,/<!-- agent-workflow-diagram:end -->/p' \
    README.md
)"

if [[ -z "$workflow_diagram" ]] ||
  ! grep -Fq '<!-- agent-workflow-diagram:start -->' <<<"$workflow_diagram" ||
  ! grep -Fq '<!-- agent-workflow-diagram:end -->' <<<"$workflow_diagram" ||
  ! grep -Fq '```mermaid' <<<"$workflow_diagram"; then
  echo "README.md is missing the marked Mermaid agent workflow diagram." >&2
  exit 1
fi

actual_skills="$(
  find .agents/skills -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort
)"
diagram_skills="$(
  grep -oE '\$[a-z0-9-]+' <<<"$workflow_diagram" | sed 's/^\$//' | sort -u
)"
actual_agents="$(
  find .codex/agents -mindepth 1 -maxdepth 1 -type f -name '*.toml' \
    -exec basename {} .toml \; | sort
)"
diagram_agents="$(
  grep -oE 'Agent: [a-z0-9-]+' <<<"$workflow_diagram" |
    sed 's/^Agent: //' |
    sort -u
)"

if [[ "$actual_skills" != "$diagram_skills" ]]; then
  echo "README agent workflow skill inventory is stale." >&2
  printf 'Expected:\n%s\nDiagram:\n%s\n' "$actual_skills" "$diagram_skills" >&2
  exit 1
fi

if [[ "$actual_agents" != "$diagram_agents" ]]; then
  echo "README agent workflow agent inventory is stale." >&2
  printf 'Expected:\n%s\nDiagram:\n%s\n' "$actual_agents" "$diagram_agents" >&2
  exit 1
fi

for script in scripts/agent-check.sh scripts/agent-pr-summary.sh scripts/agent-task.sh scripts/agent-workflow-check.sh; do
  bash -n "$script"
done

for skill in "${required_skills[@]}"; do
  skill_dir=".agents/skills/$skill"
  skill_file="$skill_dir/SKILL.md"

  if [[ ! -s "$skill_file" || ! -s "$skill_dir/agents/openai.yaml" ]]; then
    echo "Skill is incomplete: $skill_dir" >&2
    exit 1
  fi

  if [[ "$(sed -n '1p' "$skill_file")" != "---" ]] ||
    ! grep -Fq "name: $skill" "$skill_file" ||
    ! grep -Fq "description:" "$skill_file"; then
    echo "Skill frontmatter is invalid: $skill_file" >&2
    exit 1
  fi
done

for agent in "${readonly_agents[@]}"; do
  agent_file=".codex/agents/$agent.toml"
  if [[ ! -s "$agent_file" ]] ||
    ! grep -Fq "name = \"$agent\"" "$agent_file" ||
    ! grep -Fq 'sandbox_mode = "read-only"' "$agent_file" ||
    ! grep -Fq 'developer_instructions = """' "$agent_file"; then
    echo "Custom agent is incomplete or not read-only: $agent_file" >&2
    exit 1
  fi
done

actor_file=".codex/agents/repo-actor.toml"
if [[ ! -s "$actor_file" ]] ||
  ! grep -Fq 'name = "repo-actor"' "$actor_file" ||
  ! grep -Fq 'sandbox_mode = "workspace-write"' "$actor_file" ||
  ! grep -Fq 'developer_instructions = """' "$actor_file"; then
  echo "Custom actor is incomplete or not workspace-scoped: $actor_file" >&2
  exit 1
fi

if find .agents/skills -name README.md -print -quit | grep -q .; then
  echo "Skills should not contain auxiliary README files." >&2
  exit 1
fi

if grep -R -n -E '^\[TODO|Structuring This Skill|TODO: Complete|TODO: Replace' .agents/skills >/dev/null; then
  echo "Tracked skills contain unresolved scaffold markers:" >&2
  grep -R -n -E '^\[TODO|Structuring This Skill|TODO: Complete|TODO: Replace' .agents/skills >&2
  exit 1
fi

if [[ "$(wc -l <AGENTS.md)" -gt 140 ]]; then
  echo "AGENTS.md exceeds the 140-line context budget." >&2
  exit 1
fi

if [[ -n "${AGENT_TASK_ID:-}" ]]; then
  ./scripts/agent-task.sh check "$AGENT_TASK_ID" "${AGENT_BASE_REF:-origin/main}"
fi

echo "Agent workflow checks passed."
