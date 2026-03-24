#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 40001 > vllm_logs/llama.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
vllm serve Qwen/Qwen2.5-7B-Instruct --port 40002 > vllm_logs/qwen.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 40003 > vllm_logs/ds_qwen.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B --port 40004 > vllm_logs/ds_llama.log 2>&1 &