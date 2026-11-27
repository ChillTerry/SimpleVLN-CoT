#!/usr/bin/env python3
"""Run local vLLM and navigation evaluation with a per-GPU memory budget."""

import argparse
from collections import Counter
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener


EVAL_DIR = Path(__file__).resolve().parents[1]
MIN_VLLM_MIB = math.ceil(13.3 * 1024)  # Qwen3-VL-4B with the eager, small-batch profile.
POLL_SECONDS = 30
HTTP = build_opener(ProxyHandler({}))
MEMORY_ERROR = re.compile(
    rb"out of (?:device |gpu |cuda )?memory|EGL_BAD_ALLOC|GL_OUT_OF_MEMORY|"
    rb"cudaErrorMemoryAllocation|CUDA_ERROR_OUT_OF_MEMORY|std::bad_alloc|MemoryError|"
    rb"failed workers[^\n]*?, -9\)",
    re.IGNORECASE,
)
WORKER_COUNT = re.compile(rb"^Habitat workers: (\d+)\s*$")


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    total_mib: int
    free_mib: int


@dataclass(frozen=True)
class MemoryPolicy:
    vllm_fraction: float
    worker_mib: int
    reserve_fraction: float

    def reserve(self, gpu):
        return math.ceil(gpu.total_mib * self.reserve_fraction)

    def vllm_target(self, gpu):
        return max(MIN_VLLM_MIB, int(gpu.total_mib * self.vllm_fraction))

    def vllm_budget(self, gpu):
        if gpu.free_mib < MIN_VLLM_MIB:
            return 0
        available = gpu.free_mib - self.worker_mib - self.reserve(gpu)
        return max(MIN_VLLM_MIB, min(self.vllm_target(gpu), available))

    def worker_capacity(self, gpu, vllm_budget):
        if gpu.free_mib < self.worker_mib:
            return 0
        if vllm_budget < self.vllm_target(gpu):
            return 1
        return max(1, (gpu.free_mib - self.reserve(gpu)) // self.worker_mib)


def read_gpus():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free",
         "--format=csv,noheader,nounits"], text=True, timeout=10,
    )
    gpus = {}
    for line in output.splitlines():
        index, uuid, total, free = (value.strip() for value in line.split(","))
        gpu = GPU(int(index), uuid, int(total), int(free))
        gpus[gpu.index] = gpu
    return gpus


def select_gpus(gpus, visible_devices):
    if visible_devices is None:
        return list(gpus)
    selected = []
    for token in visible_devices.split(","):
        token = token.strip()
        matches = [gpu.index for gpu in gpus.values()
                   if token == str(gpu.index) or (token.startswith("GPU-") and gpu.uuid.startswith(token))]
        if len(matches) != 1 or matches[0] in selected:
            raise ValueError(f"Invalid or duplicate CUDA_VISIBLE_DEVICES entry {token!r}")
        selected.append(matches[0])
    return selected


def worker_layout(gpus, vllm_budgets, policy, limit):
    capacity = {gpu.index: policy.worker_capacity(gpu, vllm_budgets[gpu.index])
                for gpu in gpus.values() if gpu.index in vllm_budgets}
    excess = sum(capacity.values()) - limit
    for _ in range(excess):
        gpu = min((index for index, count in capacity.items() if count),
                  key=lambda index: (gpus[index].free_mib - capacity[index] * policy.worker_mib, index))
        capacity[gpu] -= 1

    ranked = sorted(capacity, key=lambda index: (-gpus[index].free_mib, index))
    return [index for slot in range(max(capacity.values(), default=0))
            for index in ranked if capacity[index] > slot]


def stop_groups(processes):
    remaining = list(processes)
    for process in remaining:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    while remaining and time.monotonic() < deadline:
        for process in remaining[:]:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                remaining.remove(process)
        if remaining:
            time.sleep(0.2)
    for process in remaining:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def attempt_status(log_path, offset, slots):
    """Read only the failed attempt, including the worker count after resume filtering."""
    memory_error = False
    with log_path.open("rb") as source:
        source.seek(offset)
        for line in source:
            count = WORKER_COUNT.match(line)
            if count:
                slots = int(count[1])
            if MEMORY_ERROR.search(line):
                memory_error = True
    return slots, memory_error


@dataclass
class Server:
    port: int
    budget: int
    launcher: subprocess.Popen


class EvaluationRun:
    def __init__(self, args, gpus, vllm_executable, base_port, startup_timeout):
        self.args = args
        self.gpu_indices = list(gpus)
        self.cuda_indices = {index: i for i, index in enumerate(gpus)}
        self.ports = {index: base_port + i for i, index in enumerate(gpus)}
        self.startup_timeout = startup_timeout
        self.model_name = args.model_path.name
        self.policy = MemoryPolicy(args.vllm_memory_utilization, math.ceil(args.habitat_worker_gib * 1024),
                                   args.gpu_reserve_fraction)
        self.log_dir = EVAL_DIR / "logs" / args.exp_name
        self.eval_log = self.log_dir / f"{args.benchmark}.log"
        self.results = EVAL_DIR / "results" / args.exp_name / f"{args.benchmark}_pass{args.pass_k}"
        # All requests in this launcher are local. Some HTTP clients load proxy
        # transports at initialization even when localhost is in NO_PROXY.
        self.env = {name: value for name, value in os.environ.items()
                    if not name.lower().endswith("_proxy")} | {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
            "PYTHONUNBUFFERED": "1", "VLLM_NO_USAGE_STATS": "1",
        }
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.env.setdefault(name, "1")
        self.vllm_path = str(Path(vllm_executable).absolute().parent)
        self.servers = {}
        self.failed_startups = set()

    def log(self, message):
        line = f"[{time.strftime('%FT%TZ', time.gmtime())}] {message}"
        print(line, flush=True)
        with (self.log_dir / "run.log").open("a", encoding="utf-8") as log_file:
            print(line, file=log_file)

    def snapshot(self):
        gpus = read_gpus()
        missing = set(self.gpu_indices) - gpus.keys()
        if missing:
            raise RuntimeError(f"nvidia-smi did not report selected GPUs {sorted(missing)}")
        return {index: gpus[index] for index in self.gpu_indices}

    def spawn_server(self, gpu, budget, eager):
        port = self.ports[gpu.index]
        log_dir = self.log_dir / "vllm" / f"gpu{gpu.index}"
        log_dir.mkdir(parents=True, exist_ok=True)
        utilization = budget / gpu.total_mib
        small_batch = eager or budget == MIN_VLLM_MIB or utilization < 0.32
        env = self.env | {
            "CUDA_VISIBLE_DEVICES": str(gpu.index), "BASE_PORT": str(port),
            "MODEL_PATH": str(self.args.model_path), "MODEL_NAME": self.model_name,
            "GPU_MEMORY_UTILIZATION": str(utilization), "LOG_DIR": str(log_dir),
            "PYTHON": sys.executable, "STARTUP_TIMEOUT": str(self.startup_timeout),
            "PATH": os.pathsep.join((self.vllm_path, self.env.get("PATH", os.defpath))),
        }
        command = [
            "bash", str(EVAL_DIR / "scripts/serve_vllm.sh"), "1", "1",
            "--host", "127.0.0.1", "--dtype", "bfloat16", "--max-model-len", "16384",
            "--max-num-seqs", "4" if small_batch else "16",
            "--max-num-batched-tokens", "4096" if small_batch else "16384",
            "--limit-mm-per-prompt", '{"image":9,"video":0}',
            "--mm-processor-kwargs", '{"max_pixels":307200}',
        ]
        if small_batch:
            command.append("--enforce-eager")
        self.log(f"Starting vLLM on GPU {gpu.index}, port {port}, budget {budget} MiB, "
                 f"free {gpu.free_mib} MiB, eager={small_batch}")
        with (log_dir / "launcher.log").open("ab") as output:
            process = subprocess.Popen(command, env=env, cwd=EVAL_DIR,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            self.servers[gpu.index] = Server(port, budget, process)

    def server_ready(self, server):
        try:
            with HTTP.open(f"http://127.0.0.1:{server.port}/v1/models", timeout=2) as response:
                models = json.load(response)["data"]
            return any(model["id"] == self.model_name and model.get("root") == str(self.args.model_path)
                       for model in models)
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def start_available(self, gpus):
        pending = [index for index in gpus if index not in self.servers and index not in self.failed_startups]
        for eager in (False, True):
            launched = []
            for index in pending:
                gpu = gpus[index]
                budget = self.policy.vllm_budget(gpu)
                if budget:
                    self.spawn_server(gpu, budget, eager)
                    launched.append(index)
            pending = []
            for index in launched:
                server = self.servers[index]
                try:
                    ready = server.launcher.wait(timeout=self.startup_timeout + 15) == 0 and self.server_ready(server)
                except subprocess.TimeoutExpired:
                    ready = False
                if ready:
                    self.log(f"vLLM is ready on GPU {index}, port {server.port}")
                else:
                    pending.append(index)
            if not pending:
                break
            stop_groups(self.servers[index].launcher for index in pending)
            for index in pending:
                del self.servers[index]
            self.log(f"vLLM launch failed on GPUs {pending}. See {self.log_dir / 'vllm'}")
            gpus = self.snapshot()
        self.failed_startups.update(pending)
        if not self.servers and len(self.failed_startups) == len(self.gpu_indices):
            raise RuntimeError(f"No vLLM server could start. See {self.log_dir / 'vllm'}")

    def unavailable_servers(self):
        unavailable = []
        for index, server in self.servers.items():
            try:
                with HTTP.open(f"http://127.0.0.1:{server.port}/health", timeout=2):
                    pass
            except OSError:
                unavailable.append(index)
        return unavailable

    def evaluate(self, slots):
        urls = ",".join(f"http://127.0.0.1:{server.port}/v1" for server in self.servers.values())
        device_ids = ",".join(str(self.cuda_indices[index]) for index in slots)
        command = [
            sys.executable, "vln/eval.py", "--exp-config", f"config/vln_{self.args.benchmark}.yaml",
            "--vllm_base_url", urls, "--vllm_model_name", self.model_name,
            "--policy-format", self.args.policy_format, "--pass-k", str(self.args.pass_k),
            "--split-num", str(len(slots)), "--sim-gpu-ids", device_ids, "--result-path", str(self.results),
        ]
        with self.eval_log.open("ab", buffering=0) as output:
            output.write(f"\n{command!r}\n".encode())
            offset = output.tell()
            process = subprocess.Popen(command, cwd=EVAL_DIR, env=self.env, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while True:
                    try:
                        code = process.wait(timeout=POLL_SECONDS)
                        missing = self.unavailable_servers() if code else []
                        break
                    except subprocess.TimeoutExpired:
                        missing = self.unavailable_servers()
                        if missing:
                            time.sleep(3)
                            missing = self.unavailable_servers()
                        if missing:
                            code = 1
                            break
            finally:
                stop_groups([process])
        workers, memory_error = attempt_status(self.eval_log, offset, len(slots)) if code else (len(slots), False)
        return code, missing, workers, memory_error

    def run(self):
        limit = self.args.max_habitat_workers
        retry_baseline = None
        restarts = 0
        missing = []
        try:
            while True:
                if missing:
                    restarts += 1
                    if restarts > 3:
                        raise RuntimeError(f"vLLM repeatedly failed on GPUs {missing}")
                    self.log(f"Restarting vLLM on GPUs {missing}")
                    stop_groups(self.servers[index].launcher for index in missing)
                    for index in missing:
                        del self.servers[index]
                self.start_available(self.snapshot())
                gpus = self.snapshot()
                budgets = {index: server.budget for index, server in self.servers.items()}
                eligible = gpus
                if retry_baseline is not None:
                    eligible = {index: gpu for index, gpu in gpus.items()
                                if gpu.free_mib - retry_baseline[index] >= self.policy.worker_mib}
                slots = worker_layout(eligible, budgets, self.policy, limit)
                if not slots:
                    waiting = (f"a GPU with at least {self.policy.worker_mib} MiB more than its last pre-attempt reading"
                               if retry_baseline is not None else "vLLM and Habitat capacity")
                    self.log(f"Waiting for {waiting}")
                    time.sleep(POLL_SECONDS)
                    missing = self.unavailable_servers()
                    continue

                free_mib = {index: gpu.free_mib for index, gpu in gpus.items()}
                self.log(f"GPU free MiB {free_mib}. Habitat workers {dict(Counter(slots))}")
                code, missing, workers, memory_error = self.evaluate(slots)
                if code == 0:
                    self.log(f"Evaluation complete. Results in {self.results}")
                    return
                if missing:
                    continue
                if not memory_error:
                    raise RuntimeError(f"Evaluator exited with status {code}. See {self.eval_log}")
                limit = max(1, workers // 2)
                retry_baseline = free_mib if workers <= 1 else None
                self.log(f"Habitat memory error with {workers} worker(s). Retrying with up to {limit}")
        finally:
            stop_groups(server.launcher for server in self.servers.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="local Hugging Face checkpoint directory")
    parser.add_argument("--exp-name", type=Path, required=True, help="relative directory for logs and results")
    parser.add_argument("--benchmark", choices=("r2r", "rxr"), required=True)
    parser.add_argument("--policy-format", choices=("legacy", "cot"), default="legacy")
    parser.add_argument("--pass-k", type=int, default=1)
    parser.add_argument("--vllm-memory-utilization", type=float, default=0.50,
                        help="target fraction of total memory for vLLM (default: 0.50)")
    parser.add_argument("--habitat-worker-gib", type=float, default=1.5,
                        help="estimated memory per Habitat worker in GiB (default: 1.5)")
    parser.add_argument("--gpu-reserve-fraction", type=float, default=0.05,
                        help="reserve fraction of total GPU memory (default: 0.05)")
    parser.add_argument("--max-habitat-workers", type=int, default=32,
                        help="maximum Habitat workers across all GPUs (default: 32)")
    args = parser.parse_args()
    if not 0 < args.vllm_memory_utilization <= 1:
        parser.error("--vllm-memory-utilization must be in (0, 1]")
    if not 0 <= args.gpu_reserve_fraction < 1:
        parser.error("--gpu-reserve-fraction must be in [0, 1)")
    if not math.isfinite(args.habitat_worker_gib * 1024) or args.habitat_worker_gib <= 0:
        parser.error("--habitat-worker-gib must be finite and positive")
    if args.max_habitat_workers < 1 or args.pass_k < 1:
        parser.error("--max-habitat-workers and --pass-k must be positive")
    if args.exp_name.is_absolute() or not args.exp_name.parts or ".." in args.exp_name.parts:
        parser.error("--exp-name must be a relative path below the log and result directories")
    args.model_path = args.model_path.expanduser().resolve()
    if not (args.model_path / "config.json").is_file():
        parser.error("--model-path must be an unpacked Hugging Face checkpoint with config.json")
    vllm = shutil.which("vllm")
    if not vllm:
        parser.error("vllm must be on PATH. Use the Habitat Python interpreter to run this script")

    try:
        gpus = select_gpus(read_gpus(), os.environ.get("CUDA_VISIBLE_DEVICES"))
        base_port = int(os.environ.get("BASE_PORT", "22001"))
        startup_timeout = int(os.environ.get("STARTUP_TIMEOUT", "1800"))
    except ValueError as error:
        parser.error(str(error))
    if not gpus:
        parser.error("No NVIDIA GPUs found")
    if not 0 < base_port <= 65536 - len(gpus) or startup_timeout < 1:
        parser.error("Invalid BASE_PORT or STARTUP_TIMEOUT")

    run = EvaluationRun(args, gpus, vllm, base_port, startup_timeout)
    run.log_dir.mkdir(parents=True, exist_ok=True)
    with (run.log_dir / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"An evaluation is already running in {run.log_dir}")
        run.run()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda signum, _: sys.exit(128 + signum))
    main()
