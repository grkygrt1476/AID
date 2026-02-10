#!/usr/bin/env bash
set -Eeuo pipefail

STAGE="00_prep"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# run_ts: arg1 or auto
RUN_TS="${1:-}"
if [[ -z "${RUN_TS}" ]]; then
  RUN_TS="$(date +"%Y%m%d_%H%M%S")"
fi

LOG_DIR="${PROJECT_ROOT}/outputs/logs/${STAGE}"
OUT_DIR="${PROJECT_ROOT}/outputs/${STAGE}/${RUN_TS}"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

SCRIPT_STEM="$(basename "${BASH_SOURCE[0]}" .sh)"
CMD_PATH="${LOG_DIR}/${SCRIPT_STEM}_${RUN_TS}.cmd.txt"
LOG_PATH="${LOG_DIR}/${SCRIPT_STEM}_${RUN_TS}.log"

# Save exact command (best-effort quoting)
{
  printf "cd %q && " "${PROJECT_ROOT}"
  printf "%q " "${BASH_SOURCE[0]}" "$@"
  printf "\n"
} > "${CMD_PATH}"

# Run body and tee logs
{
  echo "[RUN] stage=${STAGE} run_ts=${RUN_TS}"
  echo "[PATH] project_root=${PROJECT_ROOT}"
  echo "[LOG] cmd saved: ${CMD_PATH}"
  echo "[LOG] log saved: ${LOG_PATH}"
  echo "[OUT] out_dir: ${OUT_DIR}"
  echo

  cd "${PROJECT_ROOT}"

  if [[ ! -d ".venv" ]]; then
    echo "[STEP] create venv: .venv"
    python3 -m venv .venv
  else
    echo "[STEP] .venv already exists (skip create)"
  fi

  # shellcheck disable=SC1091
  source .venv/bin/activate

  echo "[STEP] upgrade pip/setuptools/wheel"
  python -m pip install -U pip setuptools wheel

  echo "[STEP] write requirements.txt (snapshot)"
  pip freeze > requirements.txt

  echo
  echo "[CHECK] which python: $(which python)"
  echo "[CHECK] python -V: $(python -V 2>&1)"
  echo "[CHECK] pip -V: $(pip -V 2>&1)"

  echo
  echo "[DONE] venv ready"
} 2>&1 | tee "${LOG_PATH}"

# Preserve original exit code from the block when using pipe
exit "${PIPESTATUS[0]}"
