"""Engineering-layer guards: --mobility semantics, pool cache, mobility checkpoints.

The candidate pool cache must be invisible to the science: identical seeds,
identical certificates, identical selections - only the duplicated SDP work
disappears, and only completed mobilities survive an interruption.
"""
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
from journal_sim.core.channels import generate_channel
from journal_sim.design.pool_cache import CandidatePoolProvider
from journal_sim.design.selection import select_lifetime_aware
from journal_sim.dynamics.reconfiguration import PolicyRunner
from journal_sim.evaluation.dynamic import run_paired_policies, evaluate_configuration
from journal_sim.evaluation.metrics import long_term_metrics
from journal_sim.experiments.common import cli_config, execute
from .helpers import tiny_config


def strong_channel_config(**kwargs):
    """Tiny config whose pools actually build: strong direct link, no RIS path."""
    return tiny_config(pool_size=2, pool_attempts=2, time_steps=3,
                       policies=("always_reconfigure", "certificate_triggered", "static"),
                       calibration_samples=10, noise_power_dbm=30, p_max_dbm=40,
                       estimation_snr_db=80, design_gamma_mult=(2.,), power_slack_grid=(1.,),
                       epsilon_est=.02, **kwargs)


def strong_trajectory(cfg):
    c = generate_channel(cfg, 60001)
    c = c.with_links(h_bu=np.full_like(c.h_bu, 10), h_ru=np.zeros_like(c.h_ru))
    return (c,) * cfg.time_steps, (c,) * cfg.time_steps


class MobilityCliTests(unittest.TestCase):
    def test_default_runs_all_regimes_and_explicit_flag_restricts(self):
        for argv, expected in ((["prog"], ("slow", "medium", "fast")),
                               (["prog", "--mobility", "slow"], ("slow",)),
                               (["prog", "--mobility", "medium"], ("medium",)),
                               (["prog", "--mobility", "fast"], ("fast",))):
            with patch.object(sys, "argv", argv):
                cfg, _ = cli_config("test")
            self.assertEqual(cfg.mobility_regimes, expected, argv)
            if len(argv) > 2:
                self.assertEqual(cfg.mobility_level, argv[-1], argv)


class PoolCacheScienceTests(unittest.TestCase):
    def test_cached_and_uncached_runs_are_science_identical(self):
        cfg = strong_channel_config()
        trajectory, estimates = strong_trajectory(cfg)
        cached = run_paired_policies(cfg, 60001, trajectory, estimates)
        # Reference: the pre-cache path, one independent PolicyRunner per policy.
        reference_records, reference_summaries = [], []
        for policy in cfg.policies:
            runner = PolicyRunner(policy, cfg, 60001)
            records = []
            for t in range(cfg.time_steps):
                X, event = runner.step(estimates[t], t)
                event.update(evaluate_configuration(X, trajectory[t], cfg))
                records.append(event)
            reference_records.extend(records)
            reference_summaries.append(long_term_metrics(records, cfg))
        self.assertEqual(len(cached["records"]), len(reference_records))
        for cached_event, reference in zip(cached["records"], reference_records):
            self.assertEqual(cached_event["policy"], reference["policy"])
            for field in ("configuration_id", "design_status", "reason", "triggered",
                          "installed", "qos_hold", "certified_budget_hold", "trigger_budget_hold"):
                self.assertEqual(cached_event[field], reference[field], field)
            self.assertEqual(cached_event["sinr_min"], reference["sinr_min"])
            self.assertEqual(cached_event["epsilon_cert"], reference["epsilon_cert"])
        for cached_summary, reference in zip(cached["summaries"], reference_summaries):
            for field in ("long_term_ee", "qos_outage_rate", "number_of_reconfigurations",
                          "total_bits", "ris_switching_energy"):
                self.assertEqual(cached_summary[field], reference[field], field)

    def test_shared_slot_builds_pool_once_and_bills_hits_as_zero(self):
        cfg = strong_channel_config()
        trajectory, estimates = strong_trajectory(cfg)
        calls = []
        original = __import__("journal_sim.design.candidate_pool", fromlist=["build_candidate_pool"]).build_candidate_pool

        def counting(channel, local_cfg, seed, t):
            calls.append(t)
            return original(channel, local_cfg, seed, t)

        with patch("journal_sim.design.pool_cache.build_candidate_pool", side_effect=counting):
            out = run_paired_policies(cfg, 60001, trajectory, estimates)
        triggered = [(r["policy"], r["time_index"]) for r in out["records"] if r["triggered"]]
        stats = out["pool_cache"]
        self.assertEqual(stats["candidate_pool_requests"], len(triggered))
        self.assertEqual(stats["candidate_pool_builds"], len({t for _, t in triggered}))
        self.assertEqual(stats["candidate_pool_cache_hits"],
                         len(triggered) - len({t for _, t in triggered}))
        # Every distinct redesigned slot built exactly one pool.
        self.assertEqual(sorted(calls), sorted({t for _, t in triggered}))
        # Cache hits carry zero cost; misses keep the original accounting.
        for record in out["records"]:
            if record["triggered"]:
                if record["candidate_pool_cache_hit"]:
                    self.assertEqual(record["candidate_pool_build_runtime"], 0.)
                    self.assertEqual(record["strict_oracle_calls"], 0)
                    self.assertGreaterEqual(record["candidate_pool_original_runtime"], 0.)
                else:
                    self.assertFalse(record["candidate_pool_cache_hit"])
                    self.assertGreaterEqual(record["candidate_pool_build_runtime"], 0.)

    def test_provider_key_covers_mobility_and_observation(self):
        cfg = strong_channel_config()
        trajectory, estimates = strong_trajectory(cfg)
        estimate = estimates[0]
        provider = CandidatePoolProvider()
        first = provider.get_or_build(estimate, cfg, 60001, 0, epsilon_est=.02)
        self.assertFalse(first.cache_hit)
        again = provider.get_or_build(estimate, cfg, 60001, 0, epsilon_est=.02)
        self.assertTrue(again.cache_hit)
        self.assertIs(again.candidates, first.candidates)
        # A different observation at the same slot must not reuse the pool.
        moved = estimate.with_links(h_bu=np.full_like(estimate.h_bu, 11.))
        self.assertFalse(provider.get_or_build(moved, cfg, 60001, 0, epsilon_est=.02).cache_hit)
        # A different mobility regime is a different key even with equal CSI.
        medium = cfg.with_overrides(mobility_level="medium")
        self.assertFalse(provider.get_or_build(estimate, medium, 60001, 0, epsilon_est=.02).cache_hit)
        # Disagreeing epsilon_est on a cached pool is a bug, not a restamp.
        with self.assertRaises(ValueError):
            provider.get_or_build(estimate, cfg, 60001, 0, epsilon_est=.03)

    def test_selection_reads_pooled_candidates_without_mutating_them(self):
        cfg = strong_channel_config()
        trajectory, estimates = strong_trajectory(cfg)
        provider = CandidatePoolProvider()
        supplied = provider.get_or_build(estimates[0], cfg, 60001, 0, epsilon_est=.02)
        pool = supplied.candidates
        ids = [c.configuration.configuration_id for c in pool]
        # Two policies with different previous thetas select independently.
        a = select_lifetime_aware(pool, np.ones(cfg.N, complex), cfg, epsilon_est=.02, drift_rate_nu=.05)
        b = select_lifetime_aware(pool, -np.ones(cfg.N, complex), cfg, epsilon_est=.02, drift_rate_nu=.05)
        self.assertEqual([s["candidate_index"] for s in a.scores],
                         [s["candidate_index"] for s in b.scores])
        self.assertEqual(ids, [c.configuration.configuration_id for c in pool])
        for candidate in pool:
            self.assertEqual(candidate.epsilon_est, .02)
            self.assertTrue(candidate.certificate.matches(candidate.configuration.w, candidate.H_hat,
                                                          candidate.configuration.theta, cfg))


class MobilityCheckpointTests(unittest.TestCase):
    def slow_result(self):
        return dict(records=[dict(value=1)], summaries=[dict(policy="static")], designs=[],
                    arrays=dict(a=np.ones(2)), pool_cache=dict(candidate_pool_requests=1),
                    end_to_end_runtime=1.)

    def test_run_seed_stages_completed_mobilities_atomically(self):
        from journal_sim.experiments import exp3_dynamic_reconfiguration as exp3
        cfg = tiny_config(mobility_regimes=("slow", "medium", "fast"), epsilon_est=.01)
        with TemporaryDirectory() as tmp:
            staging = Path(tmp) / "staging"
            with patch.object(exp3, "run_mobility", side_effect=[self.slow_result(), RuntimeError("boom")]) as run:
                with self.assertRaises(RuntimeError):
                    exp3.run_seed(cfg, 60001, staging_dir=str(staging))
                self.assertEqual([c.args[2] for c in run.call_args_list], ["slow", "medium"])
            seed_dir = staging / "seed_60001"
            self.assertTrue((seed_dir / "slow_result.json").exists())
            self.assertTrue((seed_dir / "slow_arrays.npz").exists())
            for partial in ("medium_result.json", "medium_result.json.tmp", "fast_result.json"):
                self.assertFalse((seed_dir / partial).exists(), partial)
            payload = json.loads((seed_dir / "slow_result.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["records"][0]["value"], 1)
            self.assertNotIn("arrays", payload)

    def test_interrupted_seed_reports_completed_and_pending_mobility(self):
        from journal_sim.experiments import exp3_dynamic_reconfiguration as exp3

        def plot(rows, details, output, local):
            (output / "plot.png").write_bytes(b"placeholder")

        with TemporaryDirectory() as tmp:
            cfg = tiny_config(output_root=tmp, mobility_regimes=("slow", "medium", "fast"),
                              epsilon_est=.01, seeds=(60001,))
            staging_holder = []
            original_run_seed = exp3.run_seed

            def staging_run_seed(local, seed, staging_dir=None):
                staging_holder.append(staging_dir)
                exp3._stage_mobility(staging_dir, seed, "slow", self.slow_result())
                raise KeyboardInterrupt

            # patch with a real function (not a MagicMock) so signature-based
            # staging_dir dispatch inside execute() sees the parameter.
            with patch.object(exp3, "run_seed", new=staging_run_seed):
                with self.assertRaises(KeyboardInterrupt):
                    execute("checkpoint_test", cfg, exp3.run_seed, plot, workers=1)
            runs = sorted((Path(tmp) / "checkpoint_test").glob("*/*/manifest.json"))
            manifest = json.loads(runs[-1].read_text(encoding="utf-8"))
            detail = manifest["seed_details"][0]
            self.assertEqual(detail["status"], "INTERRUPTED")
            self.assertEqual(detail["completed_mobility"], ["slow"])
            self.assertEqual(detail["pending_mobility"], ["medium", "fast"])
            self.assertFalse(manifest["run_complete"])
            self.assertFalse(manifest.get("success"))


if __name__ == "__main__":
    unittest.main()
