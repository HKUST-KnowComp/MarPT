#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
log_dir=${VLLM_LOG_DIR:-"${script_dir}/../vllm_logs"}
mkdir -p "${log_dir}"

if ! command -v vllm >/dev/null 2>&1; then
    echo "vllm is not installed or is not available on PATH." >&2
    exit 1
fi

start_server() {
    local gpu=$1
    local model=$2
    local port=$3
    local log_name=$4

    nohup env CUDA_VISIBLE_DEVICES="${gpu}" \
        vllm serve "${model}" --port "${port}" \
        >"${log_dir}/${log_name}" 2>&1 &
    echo "Started ${model} on GPU ${gpu}, port ${port} (PID $!, log: ${log_dir}/${log_name})"
}

start_server 0 meta-llama/Llama-3.1-8B-Instruct 40001 llama.log
start_server 1 Qwen/Qwen2.5-7B-Instruct 40002 qwen.log
start_server 2 deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 40003 ds_qwen.log
start_server 3 deepseek-ai/DeepSeek-R1-Distill-Llama-8B 40004 ds_llama.log
