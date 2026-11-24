# SimpleVLN Reinforcement Learning

SimpleVLN uses full navigation episodes for online reinforcement learning. Each
rollout produces a sequence of navigation decisions and a terminal success reward.
DAPO filters episode groups with identical rewards and samples additional groups
until the training batch has enough useful groups.

Run the commands below from the `RL/` directory.

## Installation

Training and Habitat use separate environments. Install mutually compatible
PyTorch, Transformers, and vLLM versions that support the selected backbone;
follow the [vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
and [verl installation documentation](https://verl.readthedocs.io/en/latest/start/install.html).
Qwen3.5 additionally needs compatible `flash-linear-attention` and `causal-conv1d`
kernels. Its presets disable packed attention by default.

Install this checkout and the recipe dependencies in the training environment:

```bash
uv pip install -e .
uv pip install -r recipe/vln_simplevln/requirements.txt
```

For Habitat on Linux x86-64, use the official Habitat-Sim 0.3.3 headless build
in a separate Python 3.9 environment. The service runs from the `RL/` checkout.
[Habitat-Sim packages](https://anaconda.org/aihabitat/habitat-sim/files?version=0.3.3)

```bash
conda create -y -n simplevln-habitat --override-channels -c conda-forge -c aihabitat \
  python=3.9 numpy=1.26.4 'habitat-sim=0.3.3=py3.9_headless_linux_*'
conda activate simplevln-habitat
uv pip install --python "$CONDA_PREFIX/bin/python" \
  -r recipe/vln_simplevln/env_server/requirements.txt 'opencv-python<4.12'
```

The environment service includes the oracle navigation measurements it uses.
The released annotations share Habitat's `R2RVLN-v1` episode schema.

## Training Data

Use `manifests/vln_train.parquet` from the RL data release.

Set the data roots on every environment-service node:

```bash
export VLN_RL_DATA_ROOT=/path/to/release_rl_data
export VLN_SCENE_ROOT=/path/to/scene_datasets
```

The [dataset configuration](config/vln_datasets.yaml) reads
`manifests/vln_train.parquet`. Set `VLN_SCENE_ROOT` to the `scene_datasets`
directory that contains `mp3d/` and `hm3d/train/`. MP3D episode paths are
`mp3d/<scene>/<scene>.glb`. HM3D episode paths are
`hm3d/train/<scene>/<scene>.basis.glb`.

Each manifest row routes to an environment episode using
`(scene_id, episode_id)`. When that pair is shared, the instruction text selects the episode.

For custom annotations, point the YAML at the annotation files and build one manifest:

```bash
python -m recipe.vln_simplevln.make_manifest \
  --config recipe/vln_simplevln/config/vln_datasets.yaml \
  --output data/manifests/vln_train.parquet
```

The released manifest can be used directly without regeneration.

## Environment Service

Start a service in the Habitat environment on each rollout node. Choose the
worker count and GPU IDs for the available simulation resources:

```bash
python -m recipe.vln_simplevln.env_server.launch \
  --config recipe/vln_simplevln/config/vln_datasets.yaml \
  --port 8002 --pool-size 8 --gpu-ids 0,1,2,3,4,5,6,7
```

Workers load the same dataset configuration and serve one active episode each.
Idle sessions expire after 1,800 seconds by default; configure this with
`--session-ttl-sec`. Check readiness at `http://localhost:8002/health` and pool
availability at `http://localhost:8002/stats`.

The default rollout endpoint is `http://127.0.0.1:8002`, local to each Ray worker.
For a shared remote service, set `VLN_ENV_SERVER_URL=http://ENV_HOST:8002` when
launching training. The launcher forwards this setting to the Ray actors.

### CPU rendering on a PPU node

Habitat renders on the host CPU with Mesa llvmpipe. On Ubuntu 24.04, install
`libegl1`, `libegl-mesa0`, `libgl1-mesa-dri`, `libgl1`, and `libglib2.0-0t64`
for Mesa and OpenCV. Start the service on loopback after setting
`VLN_RL_DATA_ROOT` and `VLN_SCENE_ROOT` in the Habitat environment.

```bash
LP_NUM_THREADS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m recipe.vln_simplevln.env_server.launch \
  --config recipe/vln_simplevln/config/vln_datasets.yaml \
  --cpu --host 127.0.0.1 --port 8002 --pool-size 8
```

`--cpu` selects llvmpipe and sets `VLN_SIM_GPU_DEVICE_ID=-1` for the recipe's
RGB-only Habitat config. With a custom Habitat config, set
`habitat.simulator.habitat_sim_v0.gpu_device_id` to `-1` and `gpu_gpu` to `false`.
Confirm that the startup log shows `Renderer: llvmpipe` and that
`curl -fsS http://127.0.0.1:8002/stats` reports the workers ready.

Run the trainer from the separate PPU environment. Use vendor PyTorch and a PPU
vLLM build compatible with Qwen3-VL and veRL. The Alibaba Cloud 25.11 PPU image
with vLLM 0.10.2 requires the documented vendor upgrade for Qwen3-VL.
[Alibaba Cloud PPU release notes](https://help.aliyun.com/zh/document_detail/2998453.html)

For a four-device Qwen3-VL-4B run, substitute the device IDs allocated to the job.
Set `TRAIN_FILE` to the training Parquet loaded by the service.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
ray start --head --num-gpus=4

MODEL_PATH=/path/to/qwen3-vl-4b-sft-checkpoint \
TRAIN_FILE=/path/to/release_rl_data/manifests/vln_train.parquet \
VLN_ENV_SERVER_URL=http://127.0.0.1:8002 \
NUM_WORKERS=8 ROLLOUT_WINDOW=8 USE_REMOVE_PADDING=False \
bash recipe/vln_simplevln/run_dapo_1node_4gpu_qwen3vl_4b.sh
```

One active rollout occupies one Habitat worker. Set `ROLLOUT_WINDOW` no larger
than the service's `--pool-size` to avoid retries for a free worker.

## Training

Start a Ray cluster in the training environment, with all nodes using the same
code, model files, and compatible dependencies. See the
[Ray cluster guide](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html).
For the head node, [start_ray.sh](start_ray.sh) configures Ray storage through
three environment variables and forwards other arguments to `ray start`:

```bash
RAY_LOG_DIR=/data/simplevln-ray \
RAY_SPILL_DIR=/data/simplevln-ray/spill \
RAY_LOCAL_FS_CAPACITY_THRESHOLD=0.99 \
bash recipe/vln_simplevln/start_ray.sh --port=6380 --num-gpus=8
```

These are the defaults; spill defaults to `${RAY_LOG_DIR}/spill`. Ray logs are in
`${RAY_LOG_DIR}/session_latest/logs`. `RAY_TMPDIR` remains a fallback when
`RAY_LOG_DIR` is unset. Use the same directory setting for training, and set
`RAY_ADDRESS=127.0.0.1:6380` for this example. These options apply when Ray starts;
passing them to a training process that connects to an existing cluster does not
reconfigure it. The threshold raises Ray's disk capacity limit to 99%; actual
disk write failures still propagate.

Run the training launcher once on the head node. It connects through
`RAY_ADDRESS=auto`; set `RAY_ADDRESS` explicitly for another cluster address.

```bash
MODEL_PATH=/path/to/hf-checkpoint \
TRAIN_FILE=/path/to/release_rl_data/manifests/vln_train.parquet \
bash recipe/vln_simplevln/run_dapo_4node_qwen3vl_4b.sh
```

The four presets share [run_dapo.sh](run_dapo.sh):

| Preset script | Actor micro-batch per GPU | Log-prob micro-batch per GPU | Remove padding |
|---|---:|---:|---|
| `run_dapo_4node_qwen3vl_4b.sh` | 16 | 32 | True |
| `run_dapo_4node_qwen3vl_8b.sh` | 16 | 32 | True |
| `run_dapo_4node_qwen35_4b.sh` | 4 | 8 | False |
| `run_dapo_4node_qwen35_9b.sh` | 2 | 4 | False |

The presets default to four nodes with eight GPUs per node. Override `NNODES`,
`NGPUS`, `ROLLOUT_TP`, `FSDP_SIZE`, and micro-batch sizes to suit the cluster.
These are configuration defaults, not hardware requirements.

| Setting | Default | Meaning |
|---|---:|---|
| `TRAIN_BATCH_SIZE` | 128 | Episode groups retained per update |
| `GEN_BATCH_SIZE` | 192 | Episode groups sampled per generation batch |
| `ROLLOUT_N` | 8 | Trajectories per episode group |
| `NUM_WORKERS` / `ROLLOUT_WINDOW` | 48 / 48 | Agent workers / concurrent trajectories |
| `ROLLOUT_SCHEDULER` | `sliding` | Dispatch new trajectories as workers become free |
| `TEMPERATURE` | 1.0 | Rollout sampling temperature |
| `ACTOR_LR` | 5e-7 | Actor learning rate |
| `CLIP_RATIO_LOW` / `CLIP_RATIO_HIGH` | 0.2 / 0.28 | PPO clipping bounds |
| `GPU_MEM_UTIL` | 0.7 | vLLM memory utilization |
| `MAX_PROMPT_LENGTH` / `MAX_RESPONSE_LENGTH` | 4096 / 256 | Decision token limits |
| `MAX_MODEL_LEN` | 4352 | Prompt plus response budget |
| `TOTAL_STEPS` / `TOTAL_EPOCHS` | 99999 / 20 | Training limits; whichever is reached first |
| `SAVE_FREQ` | 5 | Checkpoint interval |

Reward is binary success by default (`VLN_PROGRESS_COEF=0`). Optional shaping
adds `VLN_PROGRESS_COEF * oracle_success`. KL loss and KL-in-reward are disabled.
Rollouts allow up to 64 model decisions and 200 atomic environment steps, with
90° RGB observations, 0.25 m forward steps, and 15° turns.

`FILTER_GROUPS_ENABLE=False` disables dynamic group filtering. A positive
`FILTER_MAX_GEN` caps generation batches per update; if the cap is reached, the
trainer uses the useful groups collected so far. Zero leaves the cap disabled.
When the final data epoch ends, any remaining useful groups form a partial update,
and the last model update is saved when checkpointing is enabled.
Validation is disabled by default. To enable it, provide a separate `VAL_FILE`,
serve its dataset keys, and configure validation through the trainer overrides.
Validation returns one result per episode; failed episodes stop validation instead
of being dropped from the reported scores.
Use `trainer.val_only=True` to run validation without any training updates.

Check the resolved command without starting Ray or loading a model:

```bash
MODEL_PATH=/path/to/hf-checkpoint DRY_RUN=1 \
bash recipe/vln_simplevln/run_dapo_4node_qwen3vl_4b.sh
```

Additional Hydra overrides can be appended to the command. Path settings are
`MODEL_PATH`, `TRAIN_FILE`, `VAL_FILE`, `AGENT_CFG`, `CKPT_DIR`, and `LOG_FILE`.
Logging defaults to `console`; opt into installed trackers with, for example,
`LOGGER=console,wandb`.

## Checkpoints

Checkpoints are saved under `checkpoints/<project>/<experiment>/`; logs under
`logs/<experiment>.log`. Set `PROJECT` and `EXPERIMENT` to distinguish runs.
The trainer resumes from a compatible existing checkpoint when available.
Keep `data.pt` and `vln_progress.json` with the actor checkpoint: they preserve the
data cursor and epoch separately from the optimizer step count. Use the same
training data and batch configuration when resuming. New checkpoints include a
checksum to detect mismatched data and progress files. Older checkpoints without
`vln_progress.json` can initialize a new run after exporting their actor to Hugging
Face format; set `MODEL_PATH` to that export and append `trainer.resume_mode=disable`.

Export an FSDP actor checkpoint to Hugging Face format:

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir checkpoints/simplevln-dapo/qwen3vl_4b/global_step_5/actor \
  --target_dir outputs/hf-checkpoint
```

## Verification

Run the recipe's CPU tests in the training environment:

```bash
uv pip install -r requirements-test.txt
PYTHONDONTWRITEBYTECODE=1 python -m pytest recipe/vln_simplevln/tests -q
```

They cover launcher options, dataset identity, action limits, HTTP failures,
session ownership, worker cleanup, validation alignment, and training batch layout.
They also exercise epoch limits, checkpoint retention, and resume behavior using
the actual verl checkpoint functions and a stateful dataloader. Runtime tests
check entrypoint imports and patch installation in a fresh process. These tests
require the training dependencies and are skipped if those dependencies are absent.
Tests use simulated workers and do not start Habitat, a Ray cluster, or a model.
Use the [standalone rollout tool](rollout_smoke.py) for
an actual environment/model check; its command-line options are shown with
`python -m recipe.vln_simplevln.rollout_smoke --help`.
