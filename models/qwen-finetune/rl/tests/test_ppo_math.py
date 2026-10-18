"""PPO arithmetic without a model: GAE and the right-padded batches (needs torch: run with .venv-train).

  .venv-train/bin/python -m unittest rl/tests/test_ppo_math.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parents[3] / "term-generalizer")]

try:
    import torch
except ImportError:   # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed (use .venv-train)")
class TestGae(unittest.TestCase):
    def test_lambda_one_is_reward_to_go(self):
        from ppo import gae
        r = torch.tensor([0.0, 0.5, 0.0, -1.0, 0.25])
        v = torch.tensor([0.3, -0.2, 0.1, 0.0, 0.4])
        adv, ret = gae(r, v, 1.0, 1.0)
        rtg = torch.flip(torch.cumsum(torch.flip(r, [0]), 0), [0])
        self.assertTrue(torch.allclose(ret, rtg, atol=1e-6))
        self.assertTrue(torch.allclose(adv, rtg - v, atol=1e-6))

    def test_lambda_zero_is_td(self):
        from ppo import gae
        r = torch.tensor([1.0, 0.0, 2.0])
        v = torch.tensor([0.5, 0.25, 1.0])
        adv, _ = gae(r, v, 1.0, 0.0)
        self.assertTrue(torch.allclose(adv, torch.tensor([1.0 + 0.25 - 0.5, 0.0 + 1.0 - 0.25, 2.0 - 1.0])))

    def test_group_tensors(self):
        from ppo import group_tensors
        seqs = [{"prompt": [1, 2, 3], "ids": [4, 5]}, {"prompt": [1, 2, 3], "ids": [6, 7, 8, 9]}]
        ids, mask, pls, rls = group_tensors(seqs, "cpu")
        self.assertEqual(ids.tolist(), [[1, 2, 3, 4, 5, 0, 0], [1, 2, 3, 6, 7, 8, 9]])
        self.assertEqual(mask.sum(1).tolist(), [5, 7])
        self.assertEqual((pls, rls), ([3, 3], [2, 4]))


if __name__ == "__main__":
    unittest.main()
