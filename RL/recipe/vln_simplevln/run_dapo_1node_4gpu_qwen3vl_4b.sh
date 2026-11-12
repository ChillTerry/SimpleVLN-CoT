#!/usr/bin/env bash
# SimpleVLN DAPO preset: Qwen3-VL 4B on one node with four 48 GB GPUs.
# Start Ray separately with the same CUDA_VISIBLE_DEVICES and --num-gpus=4;
# set RAY_ADDRESS to that cluster. The launcher does not start Ray or Habitat.
# Matching Habitat service options: --pool-size 8 --gpu-ids 1,5,6,7.
# Set MODEL_PATH and data paths through the shared launcher's environment variables.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,5,6,7}"

export NNODES="${NNODES:-1}"
export NGPUS="${NGPUS:-4}"
export FSDP_SIZE="${FSDP_SIZE:-4}"
export ROLLOUT_TP="${ROLLOUT_TP:-2}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-48}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-2}"
export LOG_PROB_MICRO="${LOG_PROB_MICRO:-4}"

export NUM_WORKERS="${NUM_WORKERS:-8}"
export ROLLOUT_WINDOW="${ROLLOUT_WINDOW:-${VLN_ROLLOUT_WINDOW:-8}}"

# vLLM sleep mode uses CUDA memory pools, incompatible with expandable segments.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export EXPERIMENT="${EXPERIMENT:-qwen3vl_4b_1node_4gpu_bs32_n8}"

exec bash "${SCRIPT_DIR}/run_dapo.sh" qwen3vl_4b "$@"
