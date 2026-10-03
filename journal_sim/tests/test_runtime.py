import unittest
from unittest.mock import patch
from journal_sim.config import smoke_config
from journal_sim.experiments.exp4_runtime_scaling import run_seed


class RuntimeTests(unittest.TestCase):
    def test_representative_scaling_and_end_to_end(self):
        cfg = smoke_config()
        row = dict(candidate_generation_runtime=1., certificate_runtime=3., strict_validation_runtime=2.,
                   calibration_runtime=.5, runtime=5., fast_oracle_calls=4, strict_oracle_calls=2,
                   algorithm_calls=1, design_status="NO_ELIGIBLE_CANDIDATE")
        result = dict(records=[row, row], end_to_end_runtime=12., summaries=[], designs=[], arrays={})
        with patch("journal_sim.experiments.exp4_runtime_scaling.run_paired_policies", return_value=result) as run:
            output = run_seed(cfg, 60001)
        self.assertEqual(run.call_count, 4)
        self.assertEqual([(r["N"], r["bits"]) for r in output["records"]], list(cfg.scaling_cases))
        for r in output["records"]:
            self.assertEqual(r["end_to_end_runtime"], 12.)
            self.assertEqual(r["dynamic_average_computation_cost"], 5.)
            self.assertEqual(r["failed_design_steps"], 2)
            self.assertEqual(r["strict_validation_runtime"], 4.)
        self.assertEqual(len({r["local_fingerprint"] for r in output["records"]}), 4)
