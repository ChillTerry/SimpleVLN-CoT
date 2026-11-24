"""Exercise background startup and cleanup with CPU-only stand-in services."""

import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import types

import psutil
import pytest


RECIPE = Path(__file__).resolve().parents[1]
ROOT = RECIPE.parents[2]
SCRIPT = RECIPE / 'start_dapo_1node_4gpu_qwen3vl_4b.sh'

FAKE_SERVICE = '''
import json
import os
from pathlib import Path
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

role = 'ray' if Path(sys.argv[0]).name == 'fake-ray' else 'habitat'
if role == 'ray' and sys.argv[1] == 'status':
    sys.exit(0)
exit_code = int(os.environ.get('FAKE_' + role.upper() + '_EXIT', '0'))
if exit_code:
    sys.exit(exit_code)
if role == 'ray':
    port = int(next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--dashboard-port=')))
else:
    port = int(sys.argv[sys.argv.index('--port') + 1])

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = {'version': 'fake'} if role == 'ray' else {
            'pool_size': int(os.environ['NUM_WORKERS']),
            'free_workers': int(os.environ.get('FAKE_FREE_WORKERS', os.environ['NUM_WORKERS'])),
            'unhealthy_workers': 0,
        }
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())
    def log_message(self, *args):
        pass

server = HTTPServer(('127.0.0.1', port), Handler)
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
with open(os.environ['FAKE_EVENTS'], 'a') as stream:
    stream.write(json.dumps({'role': role, 'pid': os.getpid(), 'child_pid': child.pid,
                             'args': sys.argv[1:]}) + '\\n')
server.serve_forever()
'''

FAKE_TRAIN = '''#!/usr/bin/env bash
set -euo pipefail
if [[ ${DRY_RUN:-0} == 1 ]]; then exit 0; fi
exec "$PYTHON" - "$@" <<'PY'
import json
import os
import sys
import time
with open(os.environ['FAKE_EVENTS'], 'a') as stream:
    stream.write(json.dumps({'role': 'train', 'args': sys.argv[1:],
                            'gpus': os.environ['CUDA_VISIBLE_DEVICES'],
                            'gpu_mem_util': os.environ['GPU_MEM_UTIL']}) + '\\n')
time.sleep(float(os.environ.get('FAKE_TRAIN_SECONDS', '0.1')))
sys.exit(int(os.environ.get('FAKE_TRAIN_EXIT', '0')))
PY
'''


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    for key in ('SVLN_LAUNCH_LOCK_FD', 'DRY_RUN', 'VAL_FILE', 'AGENT_CFG', 'RAY_SPILL_DIR',
                'RAY_LOG_DIR', 'RAY_LOCAL_FS_CAPACITY_THRESHOLD'):
        monkeypatch.delenv(key, raising=False)
    model = tmp_path / 'model with spaces'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    data = tmp_path / 'data/manifests'
    data.mkdir(parents=True)
    (data / 'vln_train.parquet').touch()
    scenes = tmp_path / 'scenes'
    scenes.mkdir()
    executables = {}
    for name in ('fake-ray', 'fake-habitat'):
        path = tmp_path / name
        path.write_text(f'#!{sys.executable}\n' + FAKE_SERVICE)
        path.chmod(0o755)
        executables[name] = str(path)
    train = tmp_path / 'fake-train.sh'
    train.write_text(FAKE_TRAIN)
    train.chmod(0o755)
    sockets = [socket.socket() for _ in range(3)]
    try:
        for sock in sockets:
            sock.bind(('127.0.0.1', 0))
        ports = [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()
    settings = dict(SVLN_ROOT=str(ROOT), SVLN_RUN=str(tmp_path / 'run'), PYTHON=sys.executable,
                    RAY_BIN=executables['fake-ray'], HABITAT_PYTHON=executables['fake-habitat'],
                    TRAIN_LAUNCHER=str(train), MODEL_PATH=str(model),
                    VLN_RL_DATA_ROOT=str(data.parent), TRAIN_FILE=str(data / 'vln_train.parquet'),
                    VLN_SCENE_ROOT=str(scenes), RAY_TMPDIR=str(tmp_path / 'ray'),
                    CUDA_VISIBLE_DEVICES='1,5,6,7', NNODES='1', NGPUS='4', NUM_WORKERS='8',
                    GPU_MEM_UTIL='0.7', RAY_PORT=str(ports[0]), HABITAT_PORT=str(ports[1]),
                    RAY_DASHBOARD_PORT=str(ports[2]), FAKE_EVENTS=str(tmp_path / 'events.jsonl'),
                    RAY_STARTUP_TIMEOUT='5', HABITAT_STARTUP_TIMEOUT='5')
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, 'argv', ['-', str(ROOT), str(SCRIPT)])
    module = types.ModuleType('background_launcher_under_test')
    source = SCRIPT.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
    exec(compile(source, str(SCRIPT), 'exec'), module.__dict__)
    # Disk capacity is a separate prerequisite, independent of lifecycle behavior.
    monkeypatch.setattr(module, 'check_storage', lambda: None)
    yield module
    controller = module.run_dir / 'controller.pid'
    if controller.exists():
        try:
            process = psutil.Process(int(controller.read_text()))
            process.terminate()
            process.wait(timeout=20)
        except psutil.NoSuchProcess:
            pass


def finish(module, expected_stage):
    controller = psutil.Process(int((module.run_dir / 'controller.pid').read_text()))
    controller.wait(timeout=25)
    status = json.loads((module.run_dir / 'status.json').read_text())
    log = (module.run_dir / 'launcher.log').read_text()
    assert status['stage'] == expected_stage, log
    identities = json.loads((module.run_dir / 'owned_processes.json').read_text())
    assert not any(module.same_process(identity) for identity in identities.values()), log
    events = Path(module.env['FAKE_EVENTS'])
    records = [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
    for record in records:
        if 'child_pid' in record and psutil.pid_exists(record['child_pid']):
            assert psutil.Process(record['child_pid']).status() == psutil.STATUS_ZOMBIE, log
    return records


def test_one_command_runs_services_then_training_and_cleans_up(launcher):
    launcher.arguments.append('trainer.total_training_steps=1')
    launcher.commands['train'].append('trainer.total_training_steps=1')
    assert launcher.main() == 0
    records = finish(launcher, 'complete')
    assert [record['role'] for record in records] == ['ray', 'habitat', 'train']
    assert records[-1]['args'] == ['trainer.total_training_steps=1']
    assert records[-1]['gpus'] == '1,5,6,7'
    assert records[-1]['gpu_mem_util'] == '0.7'
    ray_args = records[0]['args']
    assert ray_args[:2] == ['start', '--head']
    assert ray_args.count('--head') == 1
    assert f'--temp-dir={launcher.env["RAY_LOG_DIR"]}' in ray_args
    assert f'--object-spilling-directory={launcher.env["RAY_SPILL_DIR"]}' in ray_args
    config = next(arg.split('=', 1)[1] for arg in ray_args if arg.startswith('--system-config='))
    assert json.loads(config)['local_fs_capacity_threshold'] == 0.99


@pytest.mark.parametrize('override,value', [('FAKE_HABITAT_EXIT', '7'), ('FAKE_FREE_WORKERS', '0')])
def test_startup_failure_prevents_training_and_cleans_up(launcher, override, value):
    launcher.env[override] = value
    launcher.env['HABITAT_STARTUP_TIMEOUT'] = '1'
    assert launcher.main() == 0
    records = finish(launcher, 'failed')
    assert all(record['role'] != 'train' for record in records)


def test_training_failure_is_reported_and_services_are_cleaned_up(launcher):
    launcher.env['FAKE_TRAIN_EXIT'] = '17'
    assert launcher.main() == 0
    records = finish(launcher, 'failed')
    assert records[-1]['role'] == 'train'
    assert 'code 17' in json.loads((launcher.run_dir / 'status.json').read_text())['error']


def test_duplicate_launch_is_rejected_and_sigterm_cleans_owned_processes(launcher):
    launcher.env['FAKE_TRAIN_SECONDS'] = '60'
    assert launcher.main() == 0
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if json.loads((launcher.run_dir / 'status.json').read_text())['stage'] == 'training':
            break
        time.sleep(0.05)
    else:
        pytest.fail((launcher.run_dir / 'launcher.log').read_text())
    with (launcher.run_dir / '.launcher.lock').open('a+') as lock:
        with pytest.raises(BlockingIOError):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with pytest.raises(RuntimeError, match='still running|already owns'):
        launcher.main()
    os.kill(int((launcher.run_dir / 'controller.pid').read_text()), signal.SIGTERM)
    finish(launcher, 'failed')


def test_occupied_port_is_rejected_without_touching_existing_service(launcher):
    with socket.socket() as existing:
        existing.bind(('0.0.0.0', launcher.ray_port))
        existing.listen()
        with pytest.raises(RuntimeError, match='Port .* unavailable'):
            launcher.preflight()
        assert existing.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
    assert not launcher.run_dir.exists()


def test_check_and_dry_run_do_not_start_services_or_create_run_directory(launcher):
    for mode in ('--check', '--dry-run'):
        launcher.mode = mode
        assert launcher.main() == 0
        assert not launcher.run_dir.exists()
        assert not Path(launcher.env['FAKE_EVENTS']).exists()


def test_dry_run_uses_real_preset_and_preserves_venv_python(launcher):
    env = {**launcher.env, 'TRAIN_LAUNCHER': str(RECIPE / 'run_dapo_1node_4gpu_qwen3vl_4b.sh')}
    result = subprocess.run(['bash', str(SCRIPT), '--dry-run'], cwd='/tmp', env=env,
                            text=True, capture_output=True, check=True)
    assert 'data.train_batch_size=32' in result.stdout
    assert '+data.gen_batch_size=48' in result.stdout
    assert 'actor_rollout_ref.rollout.n=8' in result.stdout
    assert 'actor_rollout_ref.rollout.gpu_memory_utilization=0.7' in result.stdout
    assert launcher.env['PYTHON'] in result.stdout
    assert not launcher.run_dir.exists()


def test_ray_storage_overrides_reach_startup_command(launcher, tmp_path):
    env = {**launcher.env, 'RAY_LOG_DIR': str(tmp_path / 'ray-logs'),
           'RAY_SPILL_DIR': str(tmp_path / 'spill'), 'RAY_LOCAL_FS_CAPACITY_THRESHOLD': '0.99'}
    result = subprocess.run(['bash', str(SCRIPT), '--dry-run'], cwd='/tmp', env=env,
                            text=True, capture_output=True, check=True)
    assert f'RAY_LOG_DIR: {env["RAY_LOG_DIR"]}' in result.stdout
    assert f'RAY_SPILL_DIR: {env["RAY_SPILL_DIR"]}' in result.stdout
    assert 'RAY_LOCAL_FS_CAPACITY_THRESHOLD: 0.99' in result.stdout
    command = shlex.split(next(line[5:] for line in result.stdout.splitlines() if line.startswith('ray: ')))
    assert command[:2] == ['bash', str(RECIPE / 'start_ray.sh')]
    assert not any(arg.startswith(('--temp-dir=', '--object-spilling-directory=', '--system-config='))
                   for arg in command)
    assert not launcher.run_dir.exists()


@pytest.mark.parametrize('overrides,log_dir,spill_dir,threshold', [
    ({}, '/data/simplevln-ray', '/data/simplevln-ray/spill', 0.99),
    ({'RAY_LOG_DIR': '/custom/ray logs', 'RAY_SPILL_DIR': '/custom/spill dir',
      'RAY_LOCAL_FS_CAPACITY_THRESHOLD': '0.97'}, '/custom/ray logs', '/custom/spill dir', 0.97),
])
def test_start_ray_resolves_storage_and_forwards_arguments(tmp_path, overrides, log_dir, spill_dir, threshold):
    capture = tmp_path / 'capture-ray'
    capture.write_text(f'#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
    capture.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if key not in ('RAY_LOG_DIR', 'RAY_TMPDIR', 'RAY_SPILL_DIR', 'RAY_LOCAL_FS_CAPACITY_THRESHOLD')}
    env.update(RAY_BIN=str(capture), **overrides)
    result = subprocess.run(['bash', str(RECIPE / 'start_ray.sh'), '--block', '--num-gpus=4'],
                            cwd='/tmp', env=env, text=True, capture_output=True, check=True)
    args = json.loads(result.stdout)
    assert args[:2] == ['start', '--head']
    assert f'--temp-dir={log_dir}' in args
    assert f'--object-spilling-directory={spill_dir}' in args
    config = next(arg.split('=', 1)[1] for arg in args if arg.startswith('--system-config='))
    assert json.loads(config)['local_fs_capacity_threshold'] == threshold
    assert args[-2:] == ['--block', '--num-gpus=4']
