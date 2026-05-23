import unittest
from pathlib import Path

from scripts import build_review_tau_sweep as sweep


class ReviewTauSweepTests(unittest.TestCase):
    def test_tau_from_path(self):
        path = Path("results_v2/review_tau_sweep/tau4/synthetic/run.json")

        self.assertEqual(sweep.tau_from_path(path), 4.0)

    def test_summarize_groups_by_tau(self):
        rows = [
            {"tau": 1.0, "test_mse": 1.0, "ood_mse": 4.0, "nan_steps": 0, "gate_mean": 0.2},
            {"tau": 1.0, "test_mse": 3.0, "ood_mse": 6.0, "nan_steps": 1, "gate_mean": 0.4},
        ]

        summary = sweep.summarize(rows)

        self.assertEqual(summary[0]["n"], 2)
        self.assertEqual(summary[0]["test_mse_mean"], 2.0)
        self.assertEqual(summary[0]["nan_steps_sum"], 1)


if __name__ == "__main__":
    unittest.main()
