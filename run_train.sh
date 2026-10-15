#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${UFO_CACHE_DIR:-}" ]]; then
  work_dir=""
  args=("$@")
  for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
      --work-dir)
        if ((i + 1 < ${#args[@]})); then
          work_dir="${args[$((i + 1))]}"
        fi
        ;;
      --work-dir=*)
        work_dir="${args[$i]#--work-dir=}"
        ;;
    esac
  done

  if [[ -n "$work_dir" ]]; then
    work_parent="$(dirname "$work_dir")"
    if [[ "$(basename "$work_parent")" == "runs" ]]; then
      export UFO_CACHE_DIR="$(dirname "$work_parent")/cache"
    else
      export UFO_CACHE_DIR="$work_parent/cache"
    fi
  else
    export UFO_CACHE_DIR="$script_dir/cache"
  fi
fi

mkdir -p "$UFO_CACHE_DIR/uv" "$UFO_CACHE_DIR/pycache" "$UFO_CACHE_DIR/tmp"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$UFO_CACHE_DIR/uv}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$UFO_CACHE_DIR/pycache}"
export TMPDIR="${TMPDIR:-$UFO_CACHE_DIR/tmp}"
export TEMP="${TEMP:-$TMPDIR}"
export TMP="${TMP:-$TMPDIR}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

if [[ -x "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/python" ]]; then
  exec "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/python" -m ufo.train "$@"
fi
if command -v uv >/dev/null 2>&1; then
  exec uv run python -m ufo.train "$@"
fi
printf '%s\n' 'Missing Python environment. Install uv and run uv sync --locked from the repository root.' >&2
exit 1
