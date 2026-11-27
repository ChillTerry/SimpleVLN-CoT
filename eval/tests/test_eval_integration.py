"""Exercise real multiprocessing orchestration with a tiny simulator substitute."""
import ast
from contextlib import ExitStack, nullcontext, redirect_stdout
from copy import copy, deepcopy
from hashlib import sha256
import io
from itertools import chain
import json
import math
import multiprocessing as mp
from pathlib import Path
from queue import Empty, Queue
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import call, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vln'))
from result_store import append_result, load_result_rows, result_directory_lock


class Dataset:
    def __init__(self, episodes):
        self.episodes = episodes


class Environment:
    def __init__(self, config, dataset):
        self.episodes = dataset.episodes
        self.current_episode = None
    def reset(self):
        if self.current_episode.fail:
            raise RuntimeError('This completed episode should have been skipped')
        self.episode_over = False
        return {'instruction': {'text': self.current_episode.instruction.instruction_text}}
    def get_metrics(self):
        return dict(success=1, spl=.5, oracle_success=1, distance_to_goal=1, path_length=2)
    def step(self, action):
        self.episode_over = True
        return {}
    def close(self):
        pass


class Agent:
    def __init__(self, *args, **kwargs):
        pass
    def reset(self):
        pass
    def act(self, *args):
        return {'action': 0}


class Progress:
    def __init__(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def update(self, *args): pass
    def set_postfix(self, **kwargs): pass


def run_mock_evaluation(dataset, result_path, *, environment=Environment, dataset_config=None, **overrides):
    simulator = SimpleNamespace(forward_step_size=.25, turn_angle=15,
                                habitat_sim_v0=SimpleNamespace(gpu_device_id=-1))
    if dataset_config is None:
        dataset_config = SimpleNamespace(type='mock')
    cfg = SimpleNamespace(habitat=SimpleNamespace(dataset=dataset_config,
                          simulator=simulator, task=SimpleNamespace(measurements={'path_length': {}})))
    habitat = SimpleNamespace(
        datasets=SimpleNamespace(make_dataset=lambda **kwargs: dataset),
        config=SimpleNamespace(read_write=nullcontext),
    )
    ns = dict(Env=environment, SimpleVLN_Agent=Agent, time=time, ExitStack=ExitStack,
              chain=chain, copy=copy, deepcopy=deepcopy, json=json, mp=mp.get_context('fork'), Empty=Empty,
              sha256=sha256, Path=Path,
              np=SimpleNamespace(isfinite=math.isfinite), tqdm=Progress,
              habitat=habitat, get_config=lambda _: cfg,
              load_result_rows=load_result_rows, append_result=append_result)
    tree = ast.parse((ROOT / 'vln/eval.py').read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                 ('evaluate_agent', 'exclude_unreachable_rxr_episodes', 'run_evaluation',
                  'evaluate_pending', 'stop_workers', 'summarize_results')]
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<eval>', 'exec'), ns)
    args = SimpleNamespace(exp_config='mock', save_video=False, pass_k=2, split_num=64,
                           vllm_model_name='mock', result_path=result_path,
                           forward_distance=25, turn_angle=15, max_action_history=200,
                           resolution_ratio=1)
    vars(args).update(overrides)
    with result_directory_lock(result_path), redirect_stdout(io.StringIO()):
        ns['run_evaluation'](args, 'http://localhost:8001/v1')


class IntegrationTests(unittest.TestCase):
    def test_cot_failure_uses_current_metrics_and_continues_trials(self):
        episode = SimpleNamespace(scene_id='/scenes/scene/a.glb', episode_id='1', fail=False,
                                  instruction=SimpleNamespace(instruction_text='Go forward'))
        dataset = Dataset([episode])
        attempted_actions = iter([None, {'action': 1}, None, {'action': 0}])
        stepped_actions = []
        recorded_actions = []

        class CotEnvironment(Environment):
            def reset(self):
                self.distance = 2
                stepped_actions.append([])
                return super().reset()

            def step(self, action):
                stepped_actions[-1].append(action['action'])
                self.episode_over = action['action'] == 0
                if action['action'] == 1:
                    self.distance -= .25
                return {}

            def get_metrics(self):
                return dict(success=1, spl=.5, oracle_success=1, distance_to_goal=self.distance,
                            path_length=2 - self.distance, ndtw=.6, sdtw=.6)

        class CotAgent(Agent):
            def act(self, *args):
                return next(attempted_actions)

            def record_action(self, action_id):
                recorded_actions.append(action_id)

        tree = ast.parse((ROOT / 'vln/eval.py').read_text())
        worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'evaluate_agent')
        artifacts = Mock()
        ns = dict(Env=CotEnvironment, SimpleVLN_Agent=CotAgent, time=time, ExitStack=ExitStack,
                  chain=chain, copy=copy, EpisodeArtifacts=lambda *args: artifacts)
        exec(compile(ast.Module(body=[worker], type_ignores=[]), '<eval>', 'exec'), ns)
        trials = []
        tasks = Queue()
        tasks.put((0, (0, 1, 2)))
        tasks.put(None)
        with tempfile.TemporaryDirectory() as path:
            ns['evaluate_agent'](SimpleNamespace(put=trials.append), 'http://localhost/v1',
                                 'mock', SimpleNamespace(habitat=None), dataset, path,
                                 25, 15, 10, 1, True, 3, tasks, policy_format='cot')
        self.assertEqual(stepped_actions, [[], [1], [0]])
        self.assertEqual(recorded_actions, [1, 0])
        self.assertEqual([trial['success'] for trial in trials], [0, 0, 1])
        rows = [trial['result_row'] for trial in trials]
        self.assertEqual([(row['steps'], row['success'], row['spl'], row['sdtw']) for row in rows],
                         [(0, 0, 0, 0), (1, 0, 0, 0), (1, 1, .5, .6)])
        self.assertEqual([(row['ne'], row['os'], row['ndtw'], row['path_length']) for row in rows[:2]],
                         [(2, 1, .6, 0), (1.75, 1, .6, .25)])
        self.assertEqual(artifacts.start.call_args_list, [call('1', 'Go forward', i) for i in range(3)])
        self.assertEqual(artifacts.finish.call_args_list, [call(0), call(0), call(1)])
        self.assertEqual([action for action, _, _ in artifacts.mock_calls], [
            'start', 'write_frame', 'finish',
            'start', 'write_frame', 'write_frame', 'finish',
            'start', 'write_frame', 'write_frame', 'finish', 'close',
        ])

    @unittest.skipUnless('fork' in mp.get_all_start_methods(), 'Test uses Linux fork workers')
    def test_multiprocess_results_and_resume(self):
        episodes = [SimpleNamespace(scene_id='/scenes/scene/a.glb', episode_id=str(i),
                                   instruction=SimpleNamespace(instruction_text='向前走' * 5000),
                                   fail=False) for i in range(2)]
        dataset = Dataset(episodes)
        with tempfile.TemporaryDirectory() as path:
            run_mock_evaluation(dataset, path)
            records = [json.loads(line) for line in (Path(path) / 'result.json').read_text().splitlines()]
            self.assertEqual(len(records), 5)  # 4 trials plus summary, all valid JSON
            self.assertEqual(records[-1]['pass@2'], 1)
            self.assertFalse((Path(path) / 'video').exists())
            # Simulate duplicates and a different pass-k in a reused output directory.
            append_result(path, records[0])
            append_result(path, {**records[0], 'trial_total': 1, 'trial_id': 0, 'success': 0})
            for ep in episodes:
                ep.fail = True
            run_mock_evaluation(dataset, path)
            records = [json.loads(line) for line in (Path(path) / 'result.json').read_text().splitlines()]
            self.assertEqual(len(records), 8)  # only a new summary on resume
            self.assertEqual(records[-1]['sucs_all'], 1)
            self.assertEqual(len(load_result_rows(path, pass_k=2)), 4)

    @unittest.skipUnless('fork' in mp.get_all_start_methods(), 'Test uses Linux fork workers')
    def test_rxr_filters_unreachable_episodes_before_limit_and_resume(self):
        config = SimpleNamespace(
            type='R2RVLN-v1', split='val_unseen',
            data_path=str(ROOT / 'data/vln_data/eval_data/rxr/{split}/{split}_guide_en.json.gz'),
        )
        rxr_scene = 'oLBMNvg9in8'
        episodes = [
            SimpleNamespace(scene_id=f'/scenes/mp3d/{scene}/{scene}.glb', episode_id=str(index),
                            trajectory_id=trajectory_id, fail=fail,
                            instruction=SimpleNamespace(instruction_text='Go forward'))
            for index, (scene, trajectory_id, fail) in enumerate([
                (rxr_scene, '1011', True),
                (rxr_scene, '1012', False),
                ('other_scene', '1011', False),
                (rxr_scene, '1013', True),
            ])
        ]
        with tempfile.TemporaryDirectory() as path:
            append_result(path, dict(scene_id=rxr_scene, episode_id=0, episode_instruction='Go forward',
                                     trial_id=0, trial_total=2, success=0, spl=math.nan, os=0,
                                     ne=math.inf, steps=1, path_length=0))
            run_mock_evaluation(Dataset(episodes), path, dataset_config=config, max_episodes=2)
            records = [json.loads(line) for line in (Path(path) / 'result.json').read_text().splitlines()]

        self.assertTrue(math.isnan(records[0]['spl']))
        self.assertEqual(sorted((r['episode_id'], r['trial_id']) for r in records[1:-1]),
                         [(1, 0), (1, 1), (2, 0), (2, 1)])
        self.assertEqual(records[-1]['pass@2'], 1)
        self.assertEqual(records[-1]['spls_all'], .5)
        self.assertEqual(records[-1]['ones_all'], 1)

    @unittest.skipUnless('fork' in mp.get_all_start_methods(), 'Test uses Linux fork workers')
    def test_fast_worker_takes_episodes_from_the_shared_queue(self):
        ctx = mp.get_context('fork')
        unblock_slow_episode = ctx.Event()
        gpu_by_episode = ctx.Array('i', [-1] * 6)
        episodes = [SimpleNamespace(scene_id='/scenes/scene/a.glb', episode_id=str(i),
                                    instruction=SimpleNamespace(instruction_text='Go forward'),
                                    fail=False) for i in range(6)]

        class UnevenEnvironment(Environment):
            def __init__(self, config, dataset):
                super().__init__(config, dataset)
                self.gpu = config.simulator.habitat_sim_v0.gpu_device_id

            def reset(self):
                index = int(self.current_episode.episode_id)
                gpu_by_episode[index] = self.gpu
                if index == 0 and not unblock_slow_episode.wait(timeout=10):
                    raise RuntimeError('The other worker did not pick up the remaining episodes')
                if index == 4:
                    unblock_slow_episode.set()
                return super().reset()

        with tempfile.TemporaryDirectory() as path:
            run_mock_evaluation(Dataset(episodes), path, environment=UnevenEnvironment,
                                split_num=2, sim_gpu_ids=[2, 0], pass_k=1)
            records = [json.loads(line) for line in (Path(path) / 'result.json').read_text().splitlines()]
        self.assertEqual(sorted(row['episode_id'] for row in records[:-1]), list(range(6)))
        self.assertEqual(set(gpu_by_episode), {0, 2})
        self.assertEqual(len(set(gpu_by_episode[1:5])), 1)
        self.assertNotEqual(gpu_by_episode[0], gpu_by_episode[1])

    @unittest.skipUnless('fork' in mp.get_all_start_methods(), 'Test uses Linux fork workers')
    def test_failed_worker_leaves_completed_trials_available_for_resume(self):
        ctx = mp.get_context('fork')
        resets = ctx.Value('i', 0)
        episode = SimpleNamespace(scene_id='/scenes/scene/a.glb', episode_id='0', fail=False,
                                  instruction=SimpleNamespace(instruction_text='Go forward'))

        class FailingEnvironment(Environment):
            def reset(self):
                resets.value += 1
                if resets.value == 2:
                    raise RuntimeError('EGL_BAD_ALLOC')
                return super().reset()

        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(RuntimeError, '1/2 results'):
                run_mock_evaluation(Dataset([episode]), path, environment=FailingEnvironment)
            rows = load_result_rows(path, pass_k=2)
            self.assertEqual([key[3] for key in rows], [0])
            self.assertEqual(len((Path(path) / 'result.json').read_text().splitlines()), 1)
            run_mock_evaluation(Dataset([episode]), path, environment=FailingEnvironment)
            self.assertEqual(len(load_result_rows(path, pass_k=2)), 2)
        self.assertEqual(resets.value, 3)


if __name__ == '__main__':
    unittest.main()
