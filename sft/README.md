# SFT

Train Qwen3-VL-4B-Instruct on the prepared R2R/RxR data with ms-swift.
The language model, main visual merger, and DeepStack mergers are trainable;
the ViT backbone is frozen.

## Setup

```bash
bash sft/setup_env.sh
```

The `.venv-sft` environment inherits the host's vendor PPU stack and installs
the locked overlay dependencies. Prepare trajectories with
`DataProcess/tool/prepare_sft.py`; local data paths are in `data/paths.env`.
The launcher defaults to `models/Qwen3-VL-4B-Instruct` and
`data/SimpleVLN_SFT_Data/processed/ms_swift/{R2R,RxR}.jsonl`.

## Training

Run from the repository root with eight allocated devices:

```bash
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
OUTPUT_DIR="$PWD/outputs/sft/qwen3-vl-4b-initial" \
bash sft/run_sft.sh train
```

The launcher uses ZeRO-1 without CPU offload, FlashAttention 2, BF16 compute
and FP32 AdamW state. Each device processes one sample with eight accumulation
steps, giving an effective batch size of 64. Gradient checkpointing is disabled;
each rank uses four data-loader workers, ordinary padding, and supervised-position
logits. Inputs retain up to eight history frames plus the current observation,
640x480 resolution, a 4096-token limit, and assistant-only supervision.

Training runs for three epochs on all R2R/RxR samples with no validation split.
It uses a peak learning rate of 2e-5, cosine decay, 5% warm-up, AdamW
(betas 0.9/0.999, epsilon 1e-8, weight decay 0.01), gradient clipping at 1.0,
and seed 42. Resumable checkpoints are saved every 1,000 updates; the latest
two are retained. Metrics are written to `OUTPUT_DIR/logging.jsonl`.

Append `--dry-run` to print the command without starting training. Additional
ms-swift options are passed through, including checkpoint restoration:

```bash
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
bash sft/run_sft.sh train \
    --resume_from_checkpoint "$PWD/outputs/sft/qwen3-vl-4b-initial/checkpoint-4000"
```

`MODEL_PATH`, `DATA_DIR`, `OUTPUT_DIR`, and `SFT_VENV` override the default paths.
`train.py` retains the source-checked PPU DeepSpeed accumulation compatibility
fix in `ppu_compat.py`; it does not modify installed vendor files.
