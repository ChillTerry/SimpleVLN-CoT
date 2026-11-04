#!/usr/bin/env bash
set -euo pipefail
SFT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(dirname -- "$SFT_DIR")"
SFT_VENV="${SFT_VENV:-$REPO_ROOT/.venv-sft}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/Qwen3-VL-4B-Instruct}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/SimpleVLN_SFT_Data/processed/ms_swift}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/sft/qwen3-vl-4b-$(date -u +%Y%m%dT%H%M%SZ)}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available PPU index}"
if [[ ! -x "$SFT_VENV/bin/python" ]]; then
    echo "Run bash $SFT_DIR/setup_env.sh first" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MAX_JOBS="${MAX_JOBS:-2}" TOKENIZERS_PARALLELISM=false
export USE_HF=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MODELSCOPE_OFFLINE=1
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-256}"
export TORCH_EXTENSIONS_DIR="$SFT_VENV/.cache/torch_extensions"
# This host's inherited eth0 setting has no IP; one local rank uses loopback.
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1

exec "$SFT_VENV/bin/python" -m torch.distributed.run \
    --standalone --nproc_per_node=1 --module swift.cli.sft \
    --model "$MODEL_PATH" --model_type qwen3_vl \
    --dataset "$DATA_DIR/R2R.jsonl" "$DATA_DIR/RxR.jsonl" "$DATA_DIR/ScaleVLN.jsonl" \
    --train_type full --freeze_llm false --freeze_vit false --freeze_aligner false \
    --torch_dtype bfloat16 --attn_impl sdpa \
    --deepspeed "$SFT_DIR/zero3_cpu.json" \
    --gradient_checkpointing true --vit_gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant":false}' \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 --lr_scheduler_type cosine --warmup_ratio 0.05 \
    --num_train_epochs 1 --max_length 4096 --max_grad_norm 1 \
    --split_dataset_ratio 0 --dataset_num_proc 1 --dataloader_num_workers 0 \
    --strict true --seed 0 --data_seed 0 \
    --logging_steps 10 --report_to none \
    --save_steps 1000 --save_total_limit 2 --save_only_model false \
    --output_dir "$OUTPUT_DIR" --add_version false \
    "$@"
