import unittest
import torch
import sys
import os

# Path hack
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from envs.tunnel_env import TunnelEnv
from models.bio_policy import SparsePolicy

class TestBioDrone(unittest.TestCase):
    def test_env_step(self):
        env = TunnelEnv()
        env.reset()
        obs, reward, terminated, truncated, _ = env.step(1)
        self.assertEqual(obs.shape, (2,))
        self.assertIsInstance(reward, float)

    def test_policy_forward_and_mask(self):
        policy = SparsePolicy(2, 3)
        x = torch.randn(1, 2)
        out = policy(x)
        self.assertEqual(out.shape, (1, 3))
        # Check if mask buffer exists
        self.assertTrue(hasattr(policy.bio_layer, 'mask'))
        self.assertEqual(policy.bio_layer.mask.shape, policy.bio_layer.weight.shape)

if __name__ == '__main__':
    unittest.main()