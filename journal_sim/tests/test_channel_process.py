import unittest
import numpy as np
from journal_sim.core.channels import generate_channel
from journal_sim.dynamics.channel_process import channel_trajectory, evolve_channel, trajectory_id
from .helpers import tiny_config


class ChannelProcessTests(unittest.TestCase):
    def test_dynamic_channel_reproducible(self):
        cfg = tiny_config(time_steps=4)
        a, b = channel_trajectory(cfg, 60001), channel_trajectory(cfg, 60001)
        self.assertEqual(trajectory_id(a), trajectory_id(b))
        self.assertNotEqual(trajectory_id(a), trajectory_id(channel_trajectory(cfg, 60002)))
        np.testing.assert_array_equal(a[0].h_br, a[-1].h_br)
        self.assertFalse(np.array_equal(a[0].h_ru, a[-1].h_ru))
        for level in ("slow", "medium", "fast"):
            self.assertGreaterEqual(cfg.with_overrides(mobility_level=level).correlation, 0)

    def test_innovation_power_normalization(self):
        cfg = tiny_config(channel_correlation=0, N=4000)
        initial = generate_channel(cfg, 20260706)
        nxt = evolve_channel(initial, cfg, np.random.default_rng(60001))
        residual = (nxt.h_ru - nxt.mean_ru) / nxt.std_ru
        self.assertAlmostEqual(float(np.mean(abs(residual) ** 2)), 1., delta=.05)
        np.testing.assert_array_equal(nxt.mean_ru, initial.mean_ru)

    def test_correlation_one_constant(self):
        cfg = tiny_config(time_steps=3, channel_correlation=1)
        trajectory = channel_trajectory(cfg, 60001)
        np.testing.assert_array_equal(trajectory[0].h_bu, trajectory[-1].h_bu)
        np.testing.assert_array_equal(trajectory[0].h_ru, trajectory[-1].h_ru)
