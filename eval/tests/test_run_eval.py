"""Exercise memory allocation and retry decisions without allocating GPU memory."""

from collections import Counter
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import call, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_eval
from run_eval import EvaluationRun, GPU, MemoryPolicy, MIN_VLLM_MIB, worker_layout


def gpus(free_by_index):
    return {index: GPU(index, f"GPU-test-{index}", 49140, free)
            for index, free in free_by_index.items()}


class AllocationTests(unittest.TestCase):
    policy = MemoryPolicy(vllm_fraction=0.50, worker_mib=1536, reserve_fraction=0.05)

    def test_vllm_first_then_worker_then_reserve_then_vllm_growth(self):
        snapshots = gpus({0: MIN_VLLM_MIB - 1, 1: MIN_VLLM_MIB + 100,
                          2: MIN_VLLM_MIB + 1536, 3: MIN_VLLM_MIB + 1536 + 2457 + 800})
        self.assertEqual([self.policy.vllm_budget(gpu) for gpu in snapshots.values()],
                         [0, MIN_VLLM_MIB, MIN_VLLM_MIB, MIN_VLLM_MIB + 800])
        ready = gpus({0: 12000, 1: 100, 2: 1536, 3: 7000})
        budgets = {1: MIN_VLLM_MIB, 2: MIN_VLLM_MIB, 3: MIN_VLLM_MIB + 800}
        self.assertEqual(Counter(worker_layout(ready, budgets, self.policy, 32)), {2: 1, 3: 1})

    def test_replay_previous_rxr_startup_snapshot(self):
        free = dict(enumerate([15318, 15158, 28840, 30431, 29663, 30268, 15382, 22950]))
        budgets = {index: self.policy.vllm_budget(gpu) for index, gpu in gpus(free).items()}
        ready = gpus({index: mib - budgets[index] for index, mib in free.items()})
        self.assertEqual(Counter(worker_layout(ready, budgets, self.policy, 32)),
                         {0: 1, 1: 1, 2: 1, 3: 2, 4: 1, 5: 2, 6: 1, 7: 1})

    def test_global_limit_balances_projected_headroom(self):
        snapshots = gpus({index: 20380 + 350 * index for index in range(8)})
        budgets = {index: 24570 for index in snapshots}
        assigned = Counter(worker_layout(snapshots, budgets, self.policy, 32))
        self.assertEqual(assigned, {0: 3, 1: 3, 2: 4, 3: 4, 4: 4, 5: 4, 6: 5, 7: 5})

    def test_nonconsecutive_devices_keep_visible_order(self):
        snapshots = gpus({1: 0, 2: 0, 5: 0})
        self.assertEqual(run_eval.select_gpus(snapshots, "5,GPU-test-2"), [5, 2])
        for invalid in ("", "-1", "2,2", "7", "GPU-test-"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                run_eval.select_gpus(snapshots, invalid)


class RetryTests(unittest.TestCase):
    def make_run(self, worker_gib):
        args = SimpleNamespace(model_path=Path("checkpoint-16674"), exp_name=Path("test"),
                               vllm_memory_utilization=0.50, habitat_worker_gib=worker_gib,
                               gpu_reserve_fraction=0.05, max_habitat_workers=32,
                               benchmark="rxr", pass_k=1)
        run = EvaluationRun(args, [5, 2], "/fake/bin/vllm", 22001, 120)
        self.assertEqual(run.cuda_indices, {5: 0, 2: 1})
        run.servers = {index: SimpleNamespace(budget=24570, launcher=Mock()) for index in (5, 2)}
        run.log = Mock()
        run.start_available = Mock()
        run.unavailable_servers = Mock(return_value=[])
        return run

    def test_single_worker_waits_for_configured_per_card_increase(self):
        for worker_gib in (1.5, 0.75):
            with self.subTest(worker_gib=worker_gib):
                run = self.make_run(worker_gib)
                delta = int(worker_gib * 1024)
                baseline = gpus({5: 7500, 2: 4300})
                almost = gpus({5: 7500 + delta - 1, 2: 4300 + delta - 1})
                ready = almost | gpus({2: 4300 + delta})
                run.snapshot = Mock(side_effect=[baseline, baseline, almost, almost, ready, ready])
                # The evaluator clipped the requested slots to one pending episode.
                run.evaluate = Mock(side_effect=[(1, [], 1, True), (0, [], 1, False)])
                with patch.object(run_eval.time, "sleep") as sleep, patch.object(run_eval, "stop_groups"):
                    run.run()
                sleep.assert_called_once_with(30)
                self.assertGreater(len(run.evaluate.call_args_list[0].args[0]), 1)
                self.assertEqual(run.evaluate.call_args_list[1], call([2]))

    def test_memory_failure_halves_actual_worker_count(self):
        run = self.make_run(1.5)
        run.snapshot = Mock(return_value=gpus({5: 9500, 2: 7500}))
        run.evaluate = Mock(side_effect=[(1, [], 6, True), (0, [], 3, False)])
        with patch.object(run_eval, "stop_groups"):
            run.run()
        self.assertEqual([len(c.args[0]) for c in run.evaluate.call_args_list], [7, 3])

    def test_memory_failure_is_scoped_to_current_log_attempt(self):
        with TemporaryDirectory() as directory:
            log = Path(directory) / "rxr.log"
            previous = b"Habitat workers: 13\nEGL_BAD_ALLOC\n"
            log.write_bytes(previous + b"Habitat workers: 1\nAnother error\n")
            self.assertEqual(run_eval.attempt_status(log, len(previous), 32), (1, False))
            with log.open("ab") as stream:
                stream.write(b"RuntimeError: CUDA out of memory\n")
            self.assertEqual(run_eval.attempt_status(log, len(previous), 32), (1, True))

    def test_loopback_launcher_drops_inherited_proxy_transports(self):
        with patch.dict(os.environ, {'ALL_PROXY': 'socks5://127.0.0.1:1', 'http_proxy': 'http://127.0.0.1:1'}):
            run = self.make_run(1.5)
        self.assertNotIn('ALL_PROXY', run.env)
        self.assertNotIn('http_proxy', run.env)


if __name__ == "__main__":
    unittest.main()
