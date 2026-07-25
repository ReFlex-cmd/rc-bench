#!/usr/bin/env bash
set -euo pipefail

mode="${1:-quick}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v poetry >/dev/null 2>&1; then
  echo "Poetry is required. Install it and run: poetry install" >&2
  exit 127
fi

export POSTGRES_USER="${POSTGRES_USER:-rc_user}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-rc_password}"
export POSTGRES_DB="${POSTGRES_DB:-rc_bench}"
export DB_HOST="${DB_HOST:-localhost}"
export DB_PORT="${DB_PORT:-5432}"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
export SECRET_KEY="${SECRET_KEY:-local-verification-key-not-for-production}"

run_quick() {
  poetry run pytest -q -m "not integration"
  poetry run rcbench list-datasets >/dev/null
  poetry run rcbench list-reservoirs >/dev/null
  git diff --check
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

run_smoke() {
  run_quick
  require_file "configs/jmlc/smoke.yaml"
  require_file "reports/jmlc_2026/dataset_manifest.json"
  poetry run rcbench validate-spec configs/jmlc/smoke.yaml >/dev/null

  if [[ ! -x "scripts/run_jmlc_matrix.py" && ! -f "scripts/run_jmlc_matrix.py" ]]; then
    echo "Required smoke runner is missing: scripts/run_jmlc_matrix.py" >&2
    exit 1
  fi

  poetry run python scripts/run_jmlc_matrix.py \
    --config configs/jmlc/smoke.yaml \
    --output reports/jmlc_2026/smoke
}

run_release() {
  run_quick
  require_file "reports/jmlc_2026/README.md"
  require_file "reports/jmlc_2026/dataset_manifest.json"
  require_file "reports/jmlc_2026/hardware_profile.json"
  # Имя агрегатной таблицы больше не константа внутри валидатора — оно
  # выводится из режима самих строк. Без этой проверки бандл, в котором
  # headline-таблицы нет вовсе, прошёл бы гейт зелёным, хотя именно её
  # читают README бандла и построитель Pareto-диаграмм.
  require_file "reports/jmlc_2026/aggregates/matrix_table.json"
  require_file "AI_USAGE.md"

  poetry run python scripts/validate_evidence.py reports/jmlc_2026

  # grep, not rg: a missing tool makes `if <tool> ...` evaluate false, which
  # turns both checks below into silent passes. These two are what keep a
  # hostname or the 133 MB raw dataset out of a published bundle, so they use
  # the tool every POSIX system ships.
  if grep -rEn \
    --include='*.json' \
    --include='*.md' \
    '("hostname"[[:space:]]*:|[A-Za-z]:\\Users\\|/home/[^ /]+/|/Users/[^ /]+/)' \
    reports/jmlc_2026; then
    echo "Potential local identifier found in evidence bundle" >&2
    exit 1
  fi

  if git ls-files | grep -Eq \
    '(^|/)(household_power_consumption|individual_household).*'; then
    echo "Raw UCI dataset appears to be tracked by Git" >&2
    exit 1
  fi
}

case "$mode" in
  quick)
    run_quick
    ;;
  smoke)
    run_smoke
    ;;
  release)
    run_release
    ;;
  *)
    echo "Usage: bash scripts/verify.sh {quick|smoke|release}" >&2
    exit 2
    ;;
esac
