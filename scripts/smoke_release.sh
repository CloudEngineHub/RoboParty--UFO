#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

echo "[smoke_release] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

uv run pytest -q
uv run ruff check ufo tests
uv run python -m compileall -q ufo tests
git diff --check

rm -rf /tmp/ufo_smoke_release_train
./run_train.sh \
  --agent fb \
  --data-path ufo/data/lafan_29dof_10s-clipped.pkl \
  --gpu-ids single \
  --smoke \
  --work-dir /tmp/ufo_smoke_release_train

rm -f /tmp/g1_robot_state_release_smoke.yaml
uv run python -m ufo.tools.motion_data build \
  --robot configs/robots/g1_29dof.yaml \
  --source ufo/data/examples/g1_robot_state_sample.csv \
  --format robot_state_csv \
  --name g1_robot_state_sample \
  --fps 50 \
  --clip-seconds 10 \
  --out /tmp/g1_robot_state_release_smoke.yaml \
  --rebuild-cache \
  --force

git diff --check
git status --short
