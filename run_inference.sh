#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -x "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/python" ]]; then
  exec "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/python" -m ufo.inference "$@"
fi
if command -v uv >/dev/null 2>&1; then
  exec uv run python -m ufo.inference "$@"
fi
printf '%s\n' 'Missing Python environment. Install uv and run uv sync --locked from the repository root.' >&2
exit 1
