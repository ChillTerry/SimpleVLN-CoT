"""Exercise real multiprocessing orchestration with a tiny simulator substitute."""
import ast
from contextlib import ExitStack, nullcontext, redirect_stdout
import io
import json
import math
import multiprocessing as mp
from pathlib import Path
from queue import Empty
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
    def get_splits(self, count, **kwargs):
        return [Dataset(self.episodes[i::count]) for i in range(count)]


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
                  EpisodeArtifacts=lambda *args: artifacts)
        exec(compile(ast.Module(body=[worker], type_ignores=[]), '<eval>', 'exec'), ns)
        trials = []
        with tempfile.TemporaryDirectory() as path:
            ns['evaluate_agent'](SimpleNamespace(put=trials.append), 'http://localhost/v1',
                                 'mock', SimpleNamespace(habitat=None), dataset, path,
                                 25, 15, 10, 1, True, 3, {}, policy_format='cot')
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
        cfg = SimpleNamespace(habitat=SimpleNamespace(dataset=SimpleNamespace(type='mock'),
                              simulator=SimpleNamespace(forward_step_size=.25, turn_angle=15),
                              task=SimpleNamespace(measurements={'path_length': {}})))
        habitat = SimpleNamespace(config=SimpleNamespace(read_write=lambda _: nullcontext()),
                                  datasets=SimpleNamespace(make_dataset=lambda **_: dataset))
        ns = dict(Env=Environment, SimpleVLN_Agent=Agent, time=time, ExitStack=ExitStack,
                  json=json, mp=mp.get_context('fork'), Empty=Empty,
                  np=SimpleNamespace(isfinite=math.isfinite), tqdm=Progress,
                  habitat=habitat, get_config=lambda _: cfg,
                  load_result_rows=load_result_rows, append_result=append_result)
        tree = ast.parse((ROOT / 'vln/eval.py').read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                     ('evaluate_agent', 'run_evaluation', 'summarize_results')]
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<eval>', 'exec'), ns)
        with tempfile.TemporaryDirectory() as path:
            args = SimpleNamespace(exp_config='mock', save_video=False, pass_k=2, split_num=64,
                                   vllm_model_name='mock', result_path=path,
                                   forward_distance=25, turn_angle=15, max_action_history=200,
                                   resolution_ratio=1)
            with result_directory_lock(path), redirect_stdout(io.StringIO()):
                ns['run_evaluation'](args, 'http://localhost:8001/v1')
            records = [json.loads(line) for line in (Path(path) / 'result.json').read_text().splitlines()]
            self.assertEqual(len(records), 5)  # 4 trials plus summary, all valid JSON
            self.assertEqual(records[-1]['pass@2'], 1)
            self.assertFalse((Path(path) / 'video').exists())
            # Simulate duplicates and a different pass-k in a reused output directory.
            append_result(path, records[0])
            append_result(path, {**records[0], 'trial_total': 1, 'trial_id': 0, 'success': 0})
            for ep in episodes:
                ep.fail = True
            with result_directory_lock(path), redirect_stdout(io.StringIO()):
                ns['run_evaluation'](args, 'http://localhost:8001/v1')
            records = [json.loads(line) for line in (Path(path) / 'result.json').read_text().splitlines()]
            self.assertEqual(len(records), 8)  # only a new summary on resume
            self.assertEqual(records[-1]['sucs_all'], 1)
            self.assertEqual(len(load_result_rows(path, pass_k=2)), 4)


if __name__ == '__main__':
    unittest.main()
