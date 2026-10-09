#!/usr/bin/env bash

# Remaining 8B cases from the official matrix. Run inside tmux infini_bench.
set -u

model_path="/root/autodl-tmp/models/Llama-3.1-8B-Instruct"
result_dir="/root/autodl-tmp/bench-results-official-8b"
port=8102

source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/vllm-bench312-conda/
mkdir -p "$result_dir"

run_case() {
    local concurrency="$1"
    local input_len="$2"
    local output_len="$3"
    local seed
    local log_file

    seed=$(date +%s)
    log_file="$result_dir/bench_8b_con=${concurrency}_in=${input_len}_out=${output_len}_seed=${seed}.log"

    echo "Running con=${concurrency}, in=${input_len}, out=${output_len}, seed=${seed}" | tee -a "$log_file"
    vllm bench serve \
        --model "$model_path" \
        --port "$port" \
        --backend openai-chat \
        --tokenizer "$model_path" \
        --endpoint /v1/chat/completions \
        --request-rate inf \
        --num-prompts "$concurrency" \
        --max-concurrency "$concurrency" \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --ignore-eos \
        --save-result \
        --result-dir "$result_dir" \
        --seed "$seed" >> "$log_file" 2>&1
    local status=$?
    echo "Finished con=${concurrency}, in=${input_len}, out=${output_len}, status=${status}" | tee -a "$log_file"
}

run_case 16 32 256
run_case 16 32 1024
run_case 16 32 4096
run_case 64 32 1024
run_case 64 32 4096
run_case 64 256 4096
