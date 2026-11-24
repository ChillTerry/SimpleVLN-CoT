#!/usr/bin/env bash
# One command starts a detached supervisor, Ray, Habitat, and the four-GPU preset.
# Usage: bash start_dapo_1node_4gpu_qwen3vl_4b.sh [--check|--dry-run|--foreground] [Hydra overrides...]
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SVLN_ROOT="${SVLN_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
export PYTHON="${PYTHON:-${SVLN_ROOT}/data/envs/rl-train/bin/python}"

exec "${PYTHON}" - "${SVLN_ROOT}" "${SCRIPT_DIR}/start_dapo_1node_4gpu_qwen3vl_4b.sh" "$@" <<'PY'
import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import psutil

root = Path(sys.argv[1]).resolve()
script = Path(sys.argv[2]).resolve()
arguments = sys.argv[3:]
mode = arguments.pop(0) if arguments and arguments[0] in (
    '--check', '--dry-run', '--foreground', '--help', '-h'
) else ''
if mode in ('--help', '-h'):
    print('Usage: bash ' + str(script) + ' [--check|--dry-run|--foreground] [Hydra overrides...]')
    print('Default: start in the background. --check validates prerequisites; --dry-run prints commands.')
    print('Overrides: MODEL_PATH, VLN_RL_DATA_ROOT, VLN_SCENE_ROOT, SVLN_RUN, PYTHON, HABITAT_PYTHON,')
    print('CUDA_VISIBLE_DEVICES, RAY_PORT, HABITAT_PORT, RAY_DASHBOARD_PORT, and preset environment variables.')
    sys.exit(0)

env = os.environ.copy()
env.setdefault('EXPERIMENT', 'qwen3vl_4b_1node_4gpu_bs32_n8')
run_dir = Path(env.get('SVLN_RUN', str(root / 'outputs' / env['EXPERIMENT']))).resolve()
env['SVLN_ROOT'] = str(root)
env['SVLN_RUN'] = str(run_dir)
defaults = {
    'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': '1,5,6,7',
    'NNODES': '1', 'NGPUS': '4', 'NUM_WORKERS': '8', 'GPU_MEM_UTIL': '0.7',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:False', 'PYTHONUNBUFFERED': '1',
    'MODEL_PATH': str(root / 'outputs/qwen3vl_4b_rl_20261109/model_input'),
    'VLN_RL_DATA_ROOT': str(root / 'data/rl_repro/release_rl_data'),
    'VLN_SCENE_ROOT': str(root / 'data/rl_repro/scene_datasets'),
    'HABITAT_PYTHON': str(root / 'data/envs/rl-habitat/bin/python'),
    'RAY_BIN': str(Path(env['PYTHON']).parent / 'ray'),
    'RAY_TMPDIR': '/data/svln-ray-4gpu',
    'LOG_FILE': str(run_dir / 'train.log'),
    'NO_PROXY': '127.0.0.1,localhost,::1',
}
for key, value in defaults.items():
    env.setdefault(key, value)
env['RAY_TMPDIR'] = env.get('RAY_LOG_DIR', env['RAY_TMPDIR'])
env['PYTHONPATH'] = str(root / 'RL') + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
env['TRAIN_FILE'] = env.get('TRAIN_FILE', str(Path(env['VLN_RL_DATA_ROOT']) / 'manifests/vln_train.parquet'))
for key in ('PYTHON', 'HABITAT_PYTHON', 'RAY_BIN', 'RAY_TMPDIR', 'MODEL_PATH', 'VLN_RL_DATA_ROOT',
            'VLN_SCENE_ROOT', 'LOG_FILE', 'TRAIN_FILE', 'VAL_FILE', 'AGENT_CFG'):
    if key in env:
        # Keep virtual-environment executable symlinks intact.
        env[key] = str(Path(env[key]).absolute())
env['PATH'] = str(Path(env['PYTHON']).parent) + os.pathsep + env.get('PATH', '')
env['RAY_LOG_DIR'] = env['RAY_TMPDIR']
env['DRY_RUN'] = '0'
ray_port = int(env.get('RAY_PORT', '6380'))
habitat_port = int(env.get('HABITAT_PORT', '8002'))
dashboard_port = int(env.get('RAY_DASHBOARD_PORT', '8266'))
workers = int(env['NUM_WORKERS'])
object_store = int(env.get('RAY_OBJECT_STORE_MEMORY_BYTES', str(64 * 2**30)))
disk_threshold = float(env.get('RAY_LOCAL_FS_CAPACITY_THRESHOLD', '0.99'))
env['RAY_LOCAL_FS_CAPACITY_THRESHOLD'] = str(disk_threshold)
spill_dir = Path(env.get('RAY_SPILL_DIR', str(run_dir / 'ray_spill'))).resolve()
env['RAY_SPILL_DIR'] = str(spill_dir)
env['RAY_ADDRESS'] = f'127.0.0.1:{ray_port}'
env['VLN_ENV_SERVER_URL'] = f'http://127.0.0.1:{habitat_port}'
launcher = Path(env.get('TRAIN_LAUNCHER', str(script.with_name('run_dapo_1node_4gpu_qwen3vl_4b.sh')))).resolve()
ray_launcher = script.with_name('start_ray.sh')
habitat_config = Path(env.get('HABITAT_CONFIG', str(script.parent / 'config/vln_datasets.yaml'))).resolve()
env['TRAIN_LAUNCHER'] = str(launcher)
env['HABITAT_CONFIG'] = str(habitat_config)
commands = {
    'ray': ['bash', str(ray_launcher), '--block', f'--port={ray_port}',
            f'--num-gpus={env["NGPUS"]}', f'--num-cpus={env.get("RAY_NUM_CPUS", "48")}',
            f'--object-store-memory={object_store}', '--dashboard-host=127.0.0.1',
            f'--dashboard-port={dashboard_port}', '--min-worker-port=24000', '--max-worker-port=24999',
            '--node-manager-port=25001', '--object-manager-port=25002',
            '--dashboard-agent-listen-port=25003', '--dashboard-agent-grpc-port=25004',
            '--runtime-env-agent-port=25005', '--metrics-export-port=25006',
            '--ray-client-server-port=25111', '--disable-usage-stats'],
    'habitat': [env['HABITAT_PYTHON'], '-m', 'recipe.vln_simplevln.env_server.launch',
                '--config', str(habitat_config), '--host', '127.0.0.1', '--port', str(habitat_port),
                '--pool-size', str(workers), '--gpu-ids', env['CUDA_VISIBLE_DEVICES']],
    'train': ['bash', str(launcher), *arguments],
}
logs = {'ray': run_dir / 'ray.log', 'habitat': run_dir / 'habitat.log', 'train': run_dir / 'nohup.log'}
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
owned = {}


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(value)
    temporary.replace(path)


def status(stage, **details):
    value = dict(stage=stage, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **details)
    atomic(run_dir / 'status.json', json.dumps(value, indent=2) + '\n')
    print(json.dumps(value), flush=True)


def same_process(identity):
    try:
        process = psutil.Process(identity['pid'])
        return process.create_time() == identity['create_time'] and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def check_storage():
    for path in (run_dir, Path(env['RAY_TMPDIR']), spill_dir):
        existing = path
        while not existing.exists():
            existing = existing.parent
        if existing.stat().st_dev == Path('/').stat().st_dev:
            raise RuntimeError(f'Use the /data filesystem for run and Ray storage: {path}')
        usage = shutil.disk_usage(existing)
        if 1 - usage.free / usage.total >= disk_threshold or usage.free < 256 * 2**30:
            raise RuntimeError(f'Insufficient disk space for {path}: need usage below {disk_threshold:.0%} and 256 GiB free')
    if shutil.disk_usage('/dev/shm').free < object_store + 2**30:
        raise RuntimeError('Insufficient /dev/shm space for the Ray object store plus 1 GiB reserve')


def preflight():
    gpu_ids = env['CUDA_VISIBLE_DEVICES'].split(',')
    if (not all(item.isdigit() for item in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids)
            or len(gpu_ids) != int(env['NGPUS']) or int(env['NNODES']) != 1):
        raise RuntimeError('This launcher requires one node and NGPUS distinct numeric CUDA_VISIBLE_DEVICES entries')
    if workers <= 0 or object_store <= 0:
        raise RuntimeError('NUM_WORKERS and RAY_OBJECT_STORE_MEMORY_BYTES must be positive')
    if not 0 < disk_threshold <= 1:
        raise RuntimeError('RAY_LOCAL_FS_CAPACITY_THRESHOLD must be greater than zero and at most one')
    for name in ('PYTHON', 'HABITAT_PYTHON', 'RAY_BIN'):
        if not os.access(env[name], os.X_OK):
            raise RuntimeError(f'{name} is not executable: {env[name]}')
    for path in (launcher, ray_launcher, habitat_config, Path(env['MODEL_PATH']) / 'config.json',
                 Path(env['TRAIN_FILE']), Path(env.get('VAL_FILE', env['TRAIN_FILE']))):
        if not path.is_file():
            raise RuntimeError(f'Missing input file: {path}')
    if not Path(env['VLN_SCENE_ROOT']).is_dir():
        raise RuntimeError(f'Missing scene directory: {env["VLN_SCENE_ROOT"]}')
    previous = run_dir / 'owned_processes.json'
    if previous.exists():
        for name, identity in json.loads(previous.read_text()).items():
            if same_process(identity):
                raise RuntimeError(f'Previous {name} process is still running (PID {identity["pid"]}); see {previous}')
    ports = [ray_port, habitat_port, dashboard_port, 25111, *range(25001, 25007)]
    if len(set(ports)) != len(ports):
        raise RuntimeError('Service ports must be distinct')
    for port in ports:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('0.0.0.0', port))
            except OSError as exc:
                raise RuntimeError(f'Port {port} is unavailable; existing services are left running') from exc
    check_storage()
    subprocess.run(commands['train'], cwd=root, env={**env, 'DRY_RUN': '1'},
                   check=True, stdout=subprocess.PIPE, text=True)


def launch(name):
    with logs[name].open('a') as stream:
        stream.write('\n' + datetime.datetime.now(datetime.timezone.utc).isoformat()
                     + ' ' + shlex.join(commands[name]) + '\n')
        stream.flush()
        process = subprocess.Popen(commands[name], cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    identity = dict(pid=process.pid, create_time=psutil.Process(process.pid).create_time())
    owned[name] = (process, identity)
    atomic(run_dir / f'{name}.pid', str(process.pid) + '\n')
    atomic(run_dir / 'owned_processes.json', json.dumps({key: value[1] for key, value in owned.items()}, indent=2) + '\n')
    return process


def wait_ready(name, url, timeout, valid):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for dependency in owned:
            if owned[dependency][0].poll() is not None:
                raise RuntimeError(f'{dependency} exited during startup; see {logs[dependency]}')
        try:
            with opener.open(url, timeout=2) as response:
                if valid(json.load(response)):
                    return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError(f'{name} readiness timed out after {timeout}s; see {logs[name]}')


def cleanup():
    for process, identity in reversed(list(owned.values())):
        try:
            parent = psutil.Process(identity['pid'])
            if parent.create_time() != identity['create_time']:
                continue
            targets = parent.children(recursive=True) + [parent]
        except psutil.NoSuchProcess:
            targets = []
        # Each service has its own session; never call the global `ray stop`.
        try:
            os.killpg(identity['pid'], signal.SIGTERM)
        except ProcessLookupError:
            pass
        for target in targets:
            try:
                target.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(targets, timeout=10)
        for target in alive:
            try:
                target.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=5)
        process.wait(timeout=5)


def supervise():
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        status('starting_ray')
        launch('ray')
        wait_ready('ray', f'http://127.0.0.1:{dashboard_port}/api/version',
                   float(env.get('RAY_STARTUP_TIMEOUT', '180')), lambda value: isinstance(value, dict))
        subprocess.run([env['RAY_BIN'], 'status', f'--address={env["RAY_ADDRESS"]}'],
                       env=env, check=True, timeout=30)
        status('starting_habitat')
        launch('habitat')
        wait_ready('habitat', env['VLN_ENV_SERVER_URL'] + '/stats',
                   float(env.get('HABITAT_STARTUP_TIMEOUT', '900')),
                   lambda value: value.get('pool_size') == workers and value.get('free_workers') == workers
                   and value.get('unhealthy_workers') == 0)
        train = launch('train')
        status('training', pid=train.pid)
        while train.poll() is None:
            for name in ('ray', 'habitat'):
                if owned[name][0].poll() is not None:
                    raise RuntimeError(f'{name} exited during training; see {logs[name]}')
            time.sleep(1)
        if train.returncode:
            raise RuntimeError(f'Training exited with code {train.returncode}; see {logs["train"]}')
        status('complete')
        return 0
    except BaseException as exc:
        status('failed', error=str(exc) or type(exc).__name__)
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        return exc.code if isinstance(exc, SystemExit) else 1
    finally:
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, signal.SIG_IGN)
        cleanup()


def main():
    if mode == '--dry-run' or os.environ.get('DRY_RUN') == '1':
        print(f'Run directory: {run_dir}\nGPUs: {env["CUDA_VISIBLE_DEVICES"]}\nGPU_MEM_UTIL: {env["GPU_MEM_UTIL"]}')
        for key in ('RAY_LOG_DIR', 'RAY_SPILL_DIR', 'RAY_LOCAL_FS_CAPACITY_THRESHOLD'):
            print(f'{key}: {env[key]}')
        for name, command in commands.items():
            print(f'{name}: {shlex.join(command)}')
        sys.stdout.flush()
        return subprocess.run(commands['train'], cwd=root, env={**env, 'DRY_RUN': '1'}).returncode
    inherited_lock = os.environ.get('SVLN_LAUNCH_LOCK_FD') if mode == '--foreground' else None
    if inherited_lock is None:
        preflight()
        if mode == '--check':
            print('Prerequisites passed; no services started.')
            return 0
        run_dir.mkdir(parents=True, exist_ok=True)
        lock = (run_dir / '.launcher.lock').open('a+')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'A launcher already owns {run_dir}') from exc
    else:
        lock = os.fdopen(int(inherited_lock), 'a+')
    with lock:
        if mode != '--foreground':
            status('queued')
            with (run_dir / 'launcher.log').open('a') as stream:
                process = subprocess.Popen(['nohup', 'bash', str(script), '--foreground', *arguments],
                                           cwd=root, env={**env, 'SVLN_LAUNCH_LOCK_FD': str(lock.fileno())},
                                           stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True, pass_fds=(lock.fileno(),))
            atomic(run_dir / 'controller.pid', str(process.pid) + '\n')
            print(f'Background launcher PID: {process.pid}\nProgress: {run_dir / "launcher.log"}'
                  f'\nTraining: {logs["train"]}\nStatus: {run_dir / "status.json"}')
            return 0
        atomic(run_dir / 'controller.pid', str(os.getpid()) + '\n')
        spill_dir.mkdir(parents=True, exist_ok=True)
        Path(env['RAY_TMPDIR']).mkdir(parents=True, exist_ok=True)
        return supervise()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
PY
