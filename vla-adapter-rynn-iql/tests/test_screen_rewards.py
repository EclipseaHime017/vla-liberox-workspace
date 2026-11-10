import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "screen_rewards.py"
spec = importlib.util.spec_from_file_location("screen_rewards", SCRIPT)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


class RewardScreenTests(unittest.TestCase):
    def test_discount_and_explicit_tail(self):
        np.testing.assert_allclose(screen.discounted_returns([-1, -1], .9), [-1.9, -1])
        np.testing.assert_allclose(screen.discounted_returns([-1, -1], .9, tail_value=-10), [-10, -10])
        np.testing.assert_allclose(screen.discounted_returns([1, 2], .5, lengths=[2, 1]), [1.5, 2])

    def test_expectile(self):
        self.assertAlmostEqual(screen.expectile([0, 1], .8), .8)
        self.assertAlmostEqual(screen.expectile([2] * 11, .8), 2.)

    def fixture(self, root):
        chunks = [{"start": i * 10, "end": (i + 1) * 10, "length": 10,
                   "action_source": "policy", "copied_prefix": False} for i in range(4)]
        parent = {"run_id": "p", "root_run_id": "p", "kind": "original", "split": "train",
                  "recorded_action_count": 40, "terminal_step": None, "chunks": chunks}
        branch_chunks = [dict(c, action_source="policy" if i == 0 else "human", copied_prefix=i == 0) for i, c in enumerate(chunks)]
        branch = dict(parent, run_id="b", parent_run_id="p", kind="branch", resume_step=10,
                      terminal_step=29, chunks=branch_chunks)
        entries = []
        for rid, values in (("p", [-1, -1, -1, -1]), ("b", [-1, -.5, 0, 0])):
            path = root / f"{rid}.npz"
            np.savez(path, final_reward=values, final_reward_scale=1.)
            entries.append({"run_id": rid, "reward_path": path.name, "reward_sha256": screen.digest(path)})
        dataset, reward = root / "dataset.json", root / "rewards.json"
        dataset.write_text(json.dumps({"dataset_sha256": "fixture", "episodes": [parent, branch]}))
        reward.write_text(json.dumps({"dataset_sha256": "fixture", "complete": True, "episodes": entries}))
        return dataset, reward

    def test_real_schema_alignment_grouping_and_success_cut(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, rewards = self.fixture(Path(directory))
            full = screen.analyze(dataset, rewards, gamma=.9)
            self.assertEqual(full["handoff_summary"]["human_higher_fraction"], 1.)
            self.assertAlmostEqual(full["comparisons"][0]["policy_return"], -2.71)
            self.assertEqual(full["reward_groups"]["human_post_success"]["count"], 1)
            self.assertEqual(full["reward_groups"]["failed_parent"]["count"], 4)
            trimmed = screen.analyze(dataset, rewards, gamma=.9, include_post_success=False)
            self.assertEqual(trimmed["reward_groups"]["human_post_success"]["count"], 0)
            with self.assertRaisesRegex(ValueError, "Explicit terminal V"):
                screen.analyze(dataset, rewards, gamma=.9, bootstrap_failures=True)
            with self.assertRaisesRegex(ValueError, "identity-bound"):
                screen.analyze(dataset, rewards, gamma=.9, bootstrap_failures=True, tail_values={"p": -10})

    def test_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, rewards = self.fixture(Path(directory))
            (Path(directory) / "p.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                screen.analyze(dataset, rewards, gamma=.9)

    def test_group_reference_does_not_select_a_policy_return(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, rewards = self.fixture(root)
            data, reward = json.loads(dataset.read_text()), json.loads(rewards.read_text())
            branch = dict(data["episodes"][1], run_id="b2", resume_step=20,
                          chunks=[dict(c, copied_prefix=i < 2, action_source="policy" if i < 2 else "human")
                                  for i, c in enumerate(data["episodes"][1]["chunks"])])
            data["episodes"].append(branch)
            reward["episodes"].append(dict(reward["episodes"][1], run_id="b2"))
            dataset.write_text(json.dumps(data)); rewards.write_text(json.dumps(reward))
            report = screen.analyze(dataset, rewards, gamma=.9)
            self.assertEqual(len(report["comparisons"]), 2)
            self.assertEqual(report["expectile_groups"], [])
            group = report["parent_return_groups"][0]
            self.assertEqual(group["unique_policy_return_count"], 2)
            self.assertEqual(group["parent_chunk_starts"], [10, 20])
            self.assertEqual(group["paired_covering_policy_returns"]["count"], 2)

    def test_explicit_path_map_preserves_manifests_and_checks_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, rewards = self.fixture(root)
            original = rewards.read_bytes()
            mapping = {f"{rid}.npz": root / f"{rid}.npz" for rid in ("p", "b")}
            result = screen.analyze(dataset, rewards, gamma=.9, path_map=mapping)
            self.assertEqual(result["handoff_summary"]["count"], 1)
            self.assertEqual(original, rewards.read_bytes())
            with self.assertRaisesRegex(ValueError, "path missing"):
                screen.analyze(dataset, rewards, gamma=.9, path_map={})
            np.savez(root / "p.npz", final_reward=[-1] * 4, boundary_steps=[0, 9, 20, 30, 40])
            reward = json.loads(original)
            reward["episodes"][0]["reward_sha256"] = screen.digest(root / "p.npz")
            rewards.write_text(json.dumps(reward))
            with self.assertRaisesRegex(ValueError, "boundaries differ"):
                screen.analyze(dataset, rewards, gamma=.9, path_map=mapping)

    def test_mixed_takeover_states_report_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, rewards = self.fixture(Path(directory))
            data = json.loads(dataset.read_text())
            child = dict(data["episodes"][1], run_id="b2", resume_step=20,
                         chunks=[dict(c, copied_prefix=i < 2) for i, c in enumerate(data["episodes"][1]["chunks"])])
            data["episodes"].append(child)
            dataset.write_text(json.dumps(data))
            reward = json.loads(rewards.read_text())
            reward["episodes"].append(dict(reward["episodes"][1], run_id="b2"))
            rewards.write_text(json.dumps(reward))
            result = screen.analyze(dataset, rewards, gamma=.9)
            self.assertEqual(len(result["expectile_ambiguities"]), 1)
            self.assertEqual(result["expectile_groups"], [])


if __name__ == "__main__":
    unittest.main()
