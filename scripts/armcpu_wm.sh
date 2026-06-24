#!/usr/bin/env bash
set -euo pipefail

# Armv8 CPU launcher for K-Search.
#
# Required:
# - TASK_PATH: path to task.json/task.yaml or a directory containing task.json
# - LLM_API_KEY or API_KEY
#
# Common options:
# - KSEARCH_ROOT: repo root (default: parent of this script)
# - MODEL_NAME: LLM model name (default: gpt-5.2)
# - BASE_URL: OpenAI-compatible API base URL
# - TARGET_CPU: Arm CPU architecture hint (default: armv8-a)
# - CPU_FEATURES: comma-separated features (default: neon)
# - CXX: compiler command on the Arm machine (default: c++)
# - CXXFLAGS: optional override compiler flags
# - MAX_OPT_ROUNDS: optimization rounds (default: 50)
# - ARTIFACTS_DIR: output dir (default: .ksearch-output-armcpu)

KSEARCH_ROOT="${KSEARCH_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

TASK_PATH="${TASK_PATH:-}"
MODEL_NAME="${MODEL_NAME:-gpt-5.2}"
API_KEY="${API_KEY:-${LLM_API_KEY:-}}"
BASE_URL="${BASE_URL:-https://api.openai.com/v1}"

TARGET_CPU="${TARGET_CPU:-armv8-a}"
CPU_FEATURES="${CPU_FEATURES:-neon}"
CXX="${CXX:-c++}"
CXXFLAGS="${CXXFLAGS:-}"

LANGUAGE="${LANGUAGE:-cpp}"
MAX_OPT_ROUNDS="${MAX_OPT_ROUNDS:-50}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-.ksearch-output-armcpu}"
CONTINUE_FROM_SOLUTION="${CONTINUE_FROM_SOLUTION:-}"
WORLD_MODEL_JSON="${WORLD_MODEL_JSON:-}"

WARMUP_RUNS="${WARMUP_RUNS:-10}"
ITERATIONS="${ITERATIONS:-100}"
NUM_TRIALS="${NUM_TRIALS:-3}"
TIMEOUT="${TIMEOUT:-300}"

WM="${WM:-1}"
WM_STAGNATION_WINDOW="${WM_STAGNATION_WINDOW:-5}"
WM_MAX_DIFFICULTY="${WM_MAX_DIFFICULTY:-}"

WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-ksearch-armcpu}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}-${TARGET_CPU}-armcpu-wm-opt${MAX_OPT_ROUNDS}}"

if [[ -z "${TASK_PATH}" ]]; then
  echo "ERROR: TASK_PATH is required" >&2
  exit 2
fi
if [[ -z "${API_KEY}" ]]; then
  echo "ERROR: API key is required (set LLM_API_KEY or API_KEY)" >&2
  exit 2
fi

CONT_ARGS=()
if [[ -n "${CONTINUE_FROM_SOLUTION}" ]]; then
  CONT_ARGS+=(--continue-from-solution "${CONTINUE_FROM_SOLUTION}")
fi
if [[ -n "${WORLD_MODEL_JSON}" ]]; then
  CONT_ARGS+=(--continue-from-world-model "${WORLD_MODEL_JSON}")
fi

WM_ARGS=()
if [[ "${WM}" == "1" ]]; then
  WM_ARGS+=(--world-model --wm-stagnation-window "${WM_STAGNATION_WINDOW}")
fi
if [[ -n "${WM_MAX_DIFFICULTY}" ]]; then
  WM_ARGS+=(--wm-max-difficulty "${WM_MAX_DIFFICULTY}")
fi

WANDB_ARGS=()
if [[ "${WANDB}" == "1" ]]; then
  export WANDB_API_KEY="${WANDB_API_KEY:-}"
  WANDB_ARGS+=(--wandb --run-name "${RUN_NAME}" --wandb-project "${WANDB_PROJECT}")
fi

CXXFLAGS_ARGS=()
if [[ -n "${CXXFLAGS}" ]]; then
  CXXFLAGS_ARGS+=(--cxxflags "${CXXFLAGS}")
fi

python3 -u "${KSEARCH_ROOT}/generate_kernels_and_eval.py" \
  --task-source armcpu \
  --task-path "${TASK_PATH}" \
  --model-name "${MODEL_NAME}" \
  --api-key "${API_KEY}" \
  --base-url "${BASE_URL}" \
  --language "${LANGUAGE}" \
  --target-cpu "${TARGET_CPU}" \
  --cpu-features "${CPU_FEATURES}" \
  --cxx "${CXX}" \
  "${CXXFLAGS_ARGS[@]}" \
  --warmup-runs "${WARMUP_RUNS}" \
  --iterations "${ITERATIONS}" \
  --num-trials "${NUM_TRIALS}" \
  --timeout "${TIMEOUT}" \
  --max-opt-rounds "${MAX_OPT_ROUNDS}" \
  --save-solutions \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  "${CONT_ARGS[@]}" \
  "${WM_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
