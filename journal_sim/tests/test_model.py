import unittest
import numpy as np
from journal_sim.config import JournalConfig
from journal_sim.core.channels import generate_channel, effective_channels
from journal_sim.core.models import Configuration, phase_set
from journal_sim.core.sinr import compute_sinr, rate
from journal_sim.core.power import ris_power, system_power
from .helpers import tiny_config


class ModelTests(unittest.TestCase):
    def test_sinr_equation(self):
        cfg = JournalConfig(M=1, K=2, N=2, channel_scale=1, noise_power_dbm=30).validate()
        H = np.arange(1, 9).reshape(2, 2, 2, 1).astype(complex)
        w = np.array([[[1], [2]], [[3], [4]]], complex)
        actual = compute_sinr(w, H, cfg)
        for l in range(2):
            for k in range(2):
                desired = abs(np.vdot(H[l, l, k], w[l, k])) ** 2
                interference = sum(abs(np.vdot(H[i, l, k], w[i, j])) ** 2
                                   for i in range(2) for j in range(2) if (i, j) != (l, k))
                self.assertAlmostEqual(actual[l, k], desired / (interference + 1))

    def test_direct_intercell_and_paired_ablation(self):
        cfg = JournalConfig().validate()
        on = generate_channel(cfg, 20260706)
        off_cfg = cfg.with_overrides(include_direct_intercell=False)
        off = generate_channel(off_cfg, 20260706)
        np.testing.assert_array_equal(on.h_bu, off.h_bu)
        np.testing.assert_array_equal(on.h_ru, off.h_ru)
        H_on = effective_channels(on, np.ones(cfg.N), cfg)
        H_off = effective_channels(off, np.ones(cfg.N), off_cfg)
        np.testing.assert_allclose(H_on[0, 1] - H_off[0, 1], on.h_bu[0, 1])
        np.testing.assert_array_equal(H_on[0, 0], H_off[0, 0])
        self.assertGreater(np.linalg.norm(on.h_bu[0, 1]), 0)

    def test_effective_channel_conjugation(self):
        cfg = tiny_config()
        c = generate_channel(cfg, 11)
        theta = phase_set(1)
        expected = c.h_bu[0, 0, 0] + sum(c.h_br[0, n].conj() * c.h_ru[0, 0, n] * theta[n].conj() for n in range(2))
        np.testing.assert_allclose(effective_channels(c, theta, cfg)[0, 0, 0], expected)

    def test_state_dependent_power(self):
        cfg = tiny_config(N=4, bits=2)
        theta = phase_set(2)
        self.assertAlmostEqual(ris_power(theta, cfg), cfg.p_ris_controller + 4 * cfg.p_cell_idle + 4 * cfg.p_diode_on)
        p = system_power(np.zeros((1, 1, 1)), theta, cfg)
        self.assertAlmostEqual(p.total, p.transmission_circuit + p.ris_static_state)

    def test_configuration_immutable_and_validated(self):
        cfg = tiny_config()
        c = Configuration(np.ones((1, 1, 1)) * .1, np.ones(2)).validate(cfg)
        with self.assertRaises(ValueError):
            c.w[0, 0, 0] = .2
        self.assertNotEqual(c.configuration_id, Configuration(c.w * 1.00000000001, c.theta).configuration_id)
        with self.assertRaises(ValueError):
            Configuration(c.w, np.array([1, np.exp(.2j)])).validate(cfg)

    def test_rate_units(self):
        cfg = tiny_config()
        self.assertAlmostEqual(rate(np.ones((1, 1, 1)), np.ones((1, 1, 1, 1)), cfg),
                               cfg.bandwidth_hz * np.log2(1 + 1 / cfg.noise_power))
