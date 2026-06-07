#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

./scripts/agent-workflow-check.sh

if [[ -s "${NVM_DIR:-$HOME/.nvm}/nvm.sh" ]]; then
  # shellcheck disable=SC1090
  source "${NVM_DIR:-$HOME/.nvm}/nvm.sh"
  nvm use 22 >/dev/null
fi

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js is required. Install/use Node.js 22 before running checks." >&2
  exit 1
fi

NODE_MAJOR="$(node --eval 'process.stdout.write(process.versions.node.split(".")[0])')"
if [[ "$NODE_MAJOR" != "22" ]]; then
  echo "Node.js 22 is required; found $(node --version)." >&2
  echo "Run: nvm use 22" >&2
  exit 1
fi

if [[ -f package-lock.json ]]; then
  PACKAGE_MANAGER="npm"
elif [[ -f pnpm-lock.yaml ]]; then
  PACKAGE_MANAGER="pnpm"
elif [[ -f yarn.lock ]]; then
  PACKAGE_MANAGER="yarn"
else
  echo "No supported package-manager lockfile found." >&2
  exit 1
fi

if [[ ! -d node_modules ]]; then
  echo "Dependencies are not installed. Run the locked install command first." >&2
  case "$PACKAGE_MANAGER" in
    npm) echo "  npm ci" >&2 ;;
    pnpm) echo "  pnpm install --frozen-lockfile" >&2 ;;
    yarn) echo "  yarn install --immutable" >&2 ;;
  esac
  exit 1
fi

has_script() {
  node --eval '
    const fs = require("node:fs");
    const packageJson = JSON.parse(fs.readFileSync("package.json", "utf8"));
    process.exit(packageJson.scripts?.[process.argv[1]] === undefined ? 1 : 0);
  ' "$1"
}

run_script() {
  local script_name="$1"
  echo
  echo "==> Running ${script_name}"
  case "$PACKAGE_MANAGER" in
    npm) npm run "$script_name" ;;
    pnpm) pnpm run "$script_name" ;;
    yarn) yarn "$script_name" ;;
  esac
}

run_format_check() {
  if [[ "${AGENT_FULL_FORMAT_CHECK:-0}" == "1" ]]; then
    run_script "format:check"
    return
  fi

  local changed_files
  local prettier_files
  local path
  local -a prettier_paths=()
  changed_files="$(mktemp)"
  prettier_files="$(mktemp)"

  if [[ -n "${AGENT_BASE_REF:-}" ]] &&
    git rev-parse --verify "$AGENT_BASE_REF" >/dev/null 2>&1; then
    if git merge-base "$AGENT_BASE_REF" HEAD >/dev/null 2>&1; then
      git diff --name-only --diff-filter=ACMRTUXB \
        "$AGENT_BASE_REF"...HEAD >>"$changed_files"
    else
      git diff --name-only --diff-filter=ACMRTUXB \
        "$AGENT_BASE_REF" HEAD >>"$changed_files"
    fi
  fi

  git diff --name-only --diff-filter=ACMRTUXB >>"$changed_files"
  git diff --cached --name-only --diff-filter=ACMRTUXB >>"$changed_files"
  git ls-files --others --exclude-standard >>"$changed_files"

  sort -u "$changed_files" |
    while IFS= read -r path; do
      case "$path" in
        .agent-artifacts/* | public/*.json | dist/* | test-results/*) continue ;;
        *.js | *.jsx | *.ts | *.tsx | *.json | *.css | *.md | *.html | *.yml | *.yaml)
          printf '%s\n' "$path"
          ;;
      esac
    done >"$prettier_files"

  rm -f "$changed_files"

  if [[ ! -s "$prettier_files" ]]; then
    rm -f "$prettier_files"
    echo "==> Skipping format:check: no changed supported files"
    return
  fi

  while IFS= read -r path; do
    prettier_paths+=("$path")
  done <"$prettier_files"
  rm -f "$prettier_files"

  echo
  echo "==> Running format:check on ${#prettier_paths[@]} changed files"
  if [[ ! -x node_modules/.bin/prettier ]]; then
    echo "Local Prettier executable is missing; reinstall locked dependencies." >&2
    return 1
  fi
  node_modules/.bin/prettier --check "${prettier_paths[@]}"
}

if has_script "format:check"; then
  run_format_check
else
  echo "==> Skipping format:check: no package.json script"
fi

for script_name in lint build test; do
  if has_script "$script_name"; then
    if ! run_script "$script_name"; then
      if [[ "$script_name" == "test" ]]; then
        echo >&2
        echo "If Playwright reported a missing browser or system library, run:" >&2
        echo "  npx playwright install chromium" >&2
        echo "  npx playwright install-deps chromium" >&2
      fi
      exit 1
    fi
  else
    echo "==> Skipping ${script_name}: no package.json script"
  fi
done

echo
echo "Agent checks passed."
