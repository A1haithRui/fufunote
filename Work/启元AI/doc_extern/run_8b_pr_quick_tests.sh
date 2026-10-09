#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/root/autodl-tmp/models/Llama-3.1-8B-Instruct}"
PORT="${PORT:-8102}"
RESULT_DIR="${RESULT_DIR:-/root/autodl-tmp/bench-results-pr-screenshot}"

source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/vllm-bench312-conda/

mkdir -p "${RESULT_DIR}"

run_case() {
  local con="$1"
  local input_len="$2"
  local output_len="$3"
  local num_prompts="$4"
  local seed
  seed="$(date +%s)"
  local log_file="${RESULT_DIR}/quick_8b_con=${con}_in=${input_len}_out=${output_len}_seed=${seed}.log"

  echo "Running con=${con}, input=${input_len}, output=${output_len}, num_prompts=${num_prompts}"
  vllm bench serve \
    --model "${MODEL}" \
    --port "${PORT}" \
    --backend openai-chat \
    --tokenizer "${MODEL}" \
    --endpoint /v1/chat/completions \
    --request-rate inf \
    --num-prompts "${num_prompts}" \
    --max-concurrency "${con}" \
    --random-input-len "${input_len}" \
    --random-output-len "${output_len}" \
    --ignore-eos \
    --save-result \
    --result-dir "${RESULT_DIR}" \
    --seed "${seed}" | tee "${log_file}"
}

run_case 1 32 256 20
run_case 4 4096 1024 40
run_case 16 256 256 80
run_case 64 32 256 64

echo
echo "Quick PR screenshot tests finished. Result dir: ${RESULT_DIR}"
