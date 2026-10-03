import unittest
from types import SimpleNamespace
import numpy as np
from journal_sim.core.models import phase_set, Configuration
from journal_sim.core.power import transition_energy
from journal_sim.design.selection import select_lifetime_aware, certified_lifetime
from journal_sim.evaluation.metrics import long_term_metrics
from .helpers import tiny_config


class TransitionTests(unittest.TestCase):
    def test_gray_code_switching_energy(self):
        cfg = tiny_config(N=4, bits=2, energy_per_bit_switch=.1, E_controller_fixed=.2)
        old = phase_set(2)[[0, 1, 2, 3]]
        new = phase_set(2)[[1, 2, 3, 0]]
        result = transition_energy(old, new, cfg)
        self.assertEqual(result.n_changed_elements, 4)
        self.assertEqual(result.n_changed_bits, 4)
        self.assertAlmostEqual(result.total, .6)
        same = transition_energy(old, old, cfg)
        self.assertEqual(same.switching_energy, 0)
        self.assertEqual(same.total, .2)

    def test_lifetime_selection_retains_dominated_candidate(self):
        cfg = tiny_config(energy_per_bit_switch=100, E_controller_fixed=0, drift_rate_nu=1)
        def cand(index, R, eps, theta):
            return SimpleNamespace(index=index, rate=R, system_power=1, wee=R, epsilon_cert=eps,
                                   epsilon_est=0, valid_certificate=True,
                                   configuration=Configuration(np.ones((1, 1, 1)), theta),
                                   certificate=SimpleNamespace(status="VALIDATED"))
        pool = [cand(0, 10, 1, -np.ones(2)), cand(1, 9, .9, np.ones(2))]
        out = select_lifetime_aware(pool, np.ones(2), cfg)
        self.assertEqual(out.candidate.index, 1)
        self.assertEqual(len(out.scores), 2)
        self.assertEqual(certified_lifetime(.01, .02, .002, .9), 0)
        self.assertIsNone(select_lifetime_aware(pool, np.ones(2), cfg, epsilon_est=2).candidate)

    def test_long_term_energy_accounting(self):
        cfg = tiny_config(slot_duration=.5)
        records = [dict(rate=10, transmission_circuit_power=2, ris_static_state_power=1,
                        switching_energy=.2, controller_energy=.1, configuration_id="A",
                        qos_hold=True, runtime=100., installed=True, reconfigured=False,
                        time_index=0, n_changed_elements=1, n_changed_bits=2),
                   dict(rate=20, transmission_circuit_power=4, ris_static_state_power=1,
                        switching_energy=0., controller_energy=0., configuration_id="A",
                        qos_hold=False, runtime=200., installed=False, reconfigured=False,
                        time_index=1, n_changed_elements=0, n_changed_bits=0)]
        out = long_term_metrics(records, cfg)
        self.assertAlmostEqual(out["total_bits"], 15)
        self.assertAlmostEqual(out["total_energy"], 4.3)
        self.assertAlmostEqual(out["long_term_ee"], 15 / 4.3)
        self.assertEqual(out["runtime"], 300)
        self.assertEqual(out["qos_outage_rate"], .5)
        self.assertEqual(out["number_of_reconfigurations"], 0)

    def test_candidate_pool_seed_policy_and_failures(self):
        from journal_sim.core.channels import generate_channel
        from journal_sim.design.candidate_pool import build_candidate_pool, candidate_seed
        cfg = tiny_config(M=2, pool_size=2, pool_attempts=4, noise_power_dbm=-100, channel_scale=1e5)
        channel = generate_channel(cfg, 60001)
        channel = channel.with_links(h_bu=np.full_like(channel.h_bu, 10.),
                                     h_ru=np.zeros_like(channel.h_ru))
        pool, stats = build_candidate_pool(channel, cfg, 60001, 0)
        self.assertGreater(len(pool), 0)
        self.assertEqual(pool[0].seed, candidate_seed(cfg, 60001, 0, 0))
        self.assertTrue(all(c.certificate.matches(c.configuration.w, c.H_hat, c.configuration.theta, cfg) for c in pool))
        impossible = cfg.with_overrides(p_max_dbm=-200)
        pool, stats = build_candidate_pool(channel, impossible, 60001, 0)
        self.assertEqual(len(pool), 0)
        self.assertEqual(len(stats["attempts"]), 4)
        self.assertTrue(all(x["status"] == "GENERATION_FAILED" for x in stats["attempts"]))
