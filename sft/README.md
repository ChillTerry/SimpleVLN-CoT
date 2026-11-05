# SFT

We use [ms-swift](https://github.com/modelscope/ms-swift) to conduct SFT.

## Environment Setup

Please refer to [ms-swift](https://github.com/modelscope/ms-swift) for more details on installing.
We recommend using the official Docker image.

## SFT Training

We use [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) as our foundation model.
Qwen2.5-VL / Qwen3-VL / Qwen3.5 are also supported.

Please follow the official ms-swift documentation for training.
Training data is available at [honglyhly/SimpleVLN_SFT_Data](https://huggingface.co/datasets/honglyhly/SimpleVLN_SFT_Data).

Run the cache and training commands below from the `sft/` directory.

### 1. Cache Training Data

Edit these two assignments in [cache/cache_dataset.sh](cache/cache_dataset.sh):

```bash
MODEL="/path/to/models/Qwen3-VL-4B-Instruct"
CACHE_DIR="/path/to/cache_data"
```

Converted JSONs are read from `${CACHE_DIR}/annotations/` by default; set
`JSON_DIR` to override. Use absolute paths. Use a separate cache root when
changing tokenizer, template, or image processor settings. These paths are
assigned inside the script; edit the assignments directly.

Cache all five datasets, or select individual datasets:

```bash
bash cache/cache_all.sh

# Alternatively, prepare stage-1 inputs only:
# bash cache/cache_dataset.sh SFT/R2R SFT/RxR
```

`cache_all.sh` calls `cache_dataset.sh` with all five dataset names. The latter
only exports the cache: DAgger JSONs must already be converted with
`--mode atomic`, and every conversion uses `--max_history 8 --seed 0`.


The outputs are:

| Dataset | Converted JSON, relative to `CACHE_DIR` | Cache, relative to `CACHE_DIR` |
|---|---|---|
| SFT/R2R | `annotations/SFT/R2R.json` | `SFT/R2R/train/` |
| SFT/RxR | `annotations/SFT/RxR.json` | `SFT/RxR/train/` |
| SFT/ScaleVLN | `annotations/SFT/ScaleVLN.json` | `SFT/ScaleVLN/train/` |
| DAgger/R2R | `annotations/DAgger/R2R.json` | `DAgger/R2R/train/` |
| DAgger/RxR | `annotations/DAgger/RxR.json` | `DAgger/RxR/train/` |

The cache stores conversations, image paths, and sequence lengths. Training
still loads the images. Generated JSON files contain local image paths, so the
same absolute image paths must be accessible on every training node. Rebuild
the inputs after relocating the images. Use a fresh cache output directory
when regenerating: ms-swift rejects existing export directories by default.

The fast exporter assumes **640×480 images and patch size 16**. Qwen2.5 uses
patch size 14 and trains from the converted JSONs directly.

### 2. Stage 1 Pretrain
Choose a model and edit its stage-1 script:

| Field | What to set |
|---|---|
| `--model` | Local path to the matching base model |
| `OUTPUT_DIR` | Output directory for this model and stage |
| `CACHE_DIR` | Prepared cache root for Qwen3 / Qwen3.5 |
| `ANNOTATION_DIR` | Converted JSON root for Qwen2.5 |

The scripts contain `/path/to/...` placeholders. Replace them inside the
scripts before launching. Model, image, input, and output paths must be
accessible consistently on all four nodes. Each node must expose eight GPUs.

Run the same script **once on each node**, using node ranks `0`, `1`, `2`, and
`3`. Set the same reachable node-0 address and port on every node. For example,
on node 0:

```bash
export NODE_RANK=0
export MASTER_ADDR=192.0.2.10
export MASTER_PORT=29500

bash stage1/train_qwen_3_4b_stage1.sh
```

Replace the example address with node 0's actual address. Repeat on nodes 1–3
with their respective `NODE_RANK`. The scripts fix `NNODES=4` and
`NPROC_PER_NODE=8`; `megatron` starts torchrun internally. `NODE_RANK` is the
machine index, and `MASTER_PORT` defaults to `29500` when omitted.

For another model, change the script name. For example:

```bash
bash stage1/train_qwen_25_3b_stage1.sh
```

Stage 1 targets approximately 100–200 optimizer iterations at global batch
256. The recorded R2R/RxR packing count gives about 147 iterations for the
cached configuration; Qwen2.5 is estimated at about 189 after accounting for
its image token count. Actual iteration counts depend on the converted data,
tokenizer, and resulting packing. The scripts retain one epoch and let
ms-swift calculate the iteration count from the prepared dataset.

### 3. Stage 2 Train

Prepare all five datasets. Convert any missing JSON first (section 3), then
cache the datasets that were not exported for stage 1:

```bash
bash cache/cache_dataset.sh SFT/ScaleVLN DAgger/R2R DAgger/RxR
```

Skip this command if all five caches already exist. Qwen2.5 needs only the
converted JSONs. Then edit the matching stage-2 script:

- Replace `--model /path/to/sft_checkpoint` with the **actual stage-1 checkpoint
  directory for the same model**, containing `config.json` and model weights.
- Set `OUTPUT_DIR` for the new stage.
- Set `CACHE_DIR` or `ANNOTATION_DIR` to the prepared inputs, as in stage 1.

Locate the checkpoint under the stage-1 script's `OUTPUT_DIR`. ms-swift may
create a versioned run directory beneath it; use the saved checkpoint path
reported by the completed run.

With `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` configured on all four nodes,
run the corresponding script once per node:

```bash
bash stage2/train_qwen_3_4b_stage2.sh

# For Qwen2.5-3B, select this script instead:
# bash stage2/train_qwen_25_3b_stage2.sh
```

Stage 2 initializes from stage-1 weights with `--finetune true` and trains for
two epochs with global batch 1024. Its iteration count is calculated from all
five datasets; the stage-1 estimate of 100–200 iterations applies to stage 1.

## Single-node PPU Training

Train Qwen3-VL-4B-Instruct on the prepared R2R/RxR data with ms-swift.
The language model, main visual merger, and DeepStack mergers are trainable;
the ViT backbone is frozen.

Run the setup and training commands from the repository root.

### Setup

```bash
bash sft/setup_env.sh
```

The `.venv-sft` environment inherits the host's vendor PPU stack and installs
the locked overlay dependencies. Prepare trajectories with
`DataProcess/tool/prepare_sft.py`; local data paths are in `data/paths.env`.
The launcher defaults to `models/Qwen3-VL-4B-Instruct` and
`data/SimpleVLN_SFT_Data/processed/ms_swift/{R2R,RxR}.jsonl`.

### Training

Launch training with eight allocated devices.

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
