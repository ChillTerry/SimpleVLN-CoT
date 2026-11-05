#!/usr/bin/env bash
set -euo pipefail
SFT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(dirname -- "$SFT_DIR")"
SFT_VENV="${SFT_VENV:-$REPO_ROOT/.venv-sft}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/Qwen3-VL-4B-Instruct}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/SimpleVLN_SFT_Data/processed/ms_swift}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/sft/qwen3-vl-4b-initial}"

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    echo "Usage: CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 bash sft/run_sft.sh [train] [--dry-run] [swift options]"
    exit 0
fi
if [[ "${1:-}" == train ]]; then
    shift
elif [[ -n "${1:-}" && "$1" != -* ]]; then
    echo "Usage: bash sft/run_sft.sh [train] [--dry-run] [swift options]" >&2
    exit 2
fi

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to eight allocated physical devices}"
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+){7}$ ]]; then
    echo "Select exactly eight distinct physical devices" >&2
    exit 2
fi
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
declare -A selected_devices=()
for device in "${devices[@]}"; do
    if [[ -n "${selected_devices[$device]:-}" ]]; then
        echo "Select exactly eight distinct physical devices" >&2
        exit 2
    fi
    selected_devices[$device]=1
done
if [[ ! -x "$SFT_VENV/bin/python" ]]; then
    echo "Run bash $SFT_DIR/setup_env.sh first" >&2
    exit 1
fi

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export USE_HF=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MODELSCOPE_OFFLINE=1
export IMAGE_MAX_TOKEN_NUM=300 MAX_PIXELS=307200
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN
export TORCH_EXTENSIONS_DIR="$SFT_VENV/.cache/torch_extensions"

dry_run=false
extra_arguments=()
for argument in "$@"; do
    if [[ "$argument" == --dry-run ]]; then
        dry_run=true
    else
        extra_arguments+=("$argument")
    fi
done

command=(
    "$SFT_VENV/bin/python" -m torch.distributed.run
    --standalone --nproc_per_node=8 "$SFT_DIR/train.py"
    --model "$MODEL_PATH" --model_type qwen3_vl
    --dataset "$DATA_DIR/R2R.jsonl" "$DATA_DIR/RxR.jsonl"
    --train_type full --freeze_llm false --freeze_vit true --freeze_aligner false
    --torch_dtype float32 --bf16 true --fp16 false
    --attn_impl flash_attn
    --gradient_checkpointing false --vit_gradient_checkpointing false
    --gradient_checkpointing_kwargs '{"use_reentrant":false}'
    --per_device_train_batch_size 1 --gradient_accumulation_steps 8
    --learning_rate 2e-5 --optim adamw_torch
    --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8
    --weight_decay 0.01 --max_grad_norm 1 --num_train_epochs 3
    --lr_scheduler_type cosine --warmup_ratio 0.05 --max_length 4096
    --split_dataset_ratio 0 --dataset_num_proc 1 --lazy_tokenize true
    --dataloader_num_workers 4 --dataloader_pin_memory true
    --dataloader_persistent_workers true --dataloader_drop_last false
    --remove_unused_columns false --dataset_shuffle false --train_dataloader_shuffle true
    --strict true --seed 42 --data_seed 42
    --loss_scale default --acc_strategy token --average_tokens_across_devices true
    --ddp_find_unused_parameters false --padding_free false --use_logits_to_keep true
    --logging_steps 10 --logging_nan_inf_filter false --report_to none
    --save_strategy steps --save_steps 1000 --save_total_limit 2 --save_only_model false
    --output_dir "$OUTPUT_DIR" --add_version false
    --deepspeed "$SFT_DIR/zero1.json" --dataloader_prefetch_factor 2
    "${extra_arguments[@]}"
)
if [[ "$dry_run" == true ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
