import math
import tempfile
from pathlib import Path
import unittest

from scripts import build_review_supplement as brs
from scripts import run_review_supplement as rrs


class ReviewSupplementHelperTests(unittest.TestCase):
    def test_finite_filters_nan_and_infinite_values(self):
        self.assertEqual(brs.finite([1.0, math.nan, 2.5, math.inf, -math.inf]), [1.0, 2.5])

    def test_mean_finite_ignores_nonfinite_values(self):
        self.assertAlmostEqual(brs.mean_finite([1.0, math.nan, 3.0]), 2.0)
        self.assertTrue(math.isnan(brs.mean_finite([math.nan, math.inf])))

    def test_geomean_finite_handles_positive_error_ratios(self):
        self.assertAlmostEqual(brs.geomean_finite([0.25, 1.0, 4.0]), 1.0)
        self.assertTrue(math.isnan(brs.geomean_finite([0.0, -1.0, math.nan])))

    def test_sign_test_p_value_is_two_sided_and_symmetric(self):
        self.assertAlmostEqual(brs.sign_test_p_value(3, 0), 0.25)
        self.assertAlmostEqual(brs.sign_test_p_value(0, 3), 0.25)
        self.assertAlmostEqual(brs.sign_test_p_value(1, 1), 1.0)

    def test_paired_summary_counts_wins_and_ratios(self):
        rows = [
            {"baseline": 10.0, "candidate": 1.0},
            {"baseline": 3.0, "candidate": 6.0},
            {"baseline": 8.0, "candidate": 2.0},
        ]
        summary = brs.paired_summary(rows, "baseline", "candidate")
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["candidate_wins"], 2)
        self.assertEqual(summary["baseline_wins"], 1)
        self.assertAlmostEqual(summary["mean_ratio"], (0.1 + 2.0 + 0.25) / 3.0)
        self.assertAlmostEqual(summary["geomean_ratio"], (0.1 * 2.0 * 0.25) ** (1.0 / 3.0))

    def test_make_pairwise_row_maps_columns_and_direction(self):
        rows = [
            {"mlp_test_mse": "4.0", "eml_kan_test_mse": "1.0"},
            {"mlp_test_mse": "2.0", "eml_kan_test_mse": "3.0"},
            {"mlp_test_mse": "9.0", "eml_kan_test_mse": "3.0"},
        ]
        result = brs.make_pairwise_row("Synthetic test", rows, "mlp_test_mse", "eml_kan_test_mse")
        self.assertEqual(result["comparison"], "Synthetic test")
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["eml_wins"], 2)
        self.assertEqual(result["mlp_wins"], 1)
        self.assertAlmostEqual(result["geomean_eml_over_mlp"], (0.25 * 1.5 * (1.0 / 3.0)) ** (1.0 / 3.0))

    def test_add_label_noise_is_deterministic_and_scales_to_target_std(self):
        y = [1.0, 2.0, 3.0, 4.0]
        noisy_a = rrs.add_label_noise(y, noise_frac=0.1, seed=7)
        noisy_b = rrs.add_label_noise(y, noise_frac=0.1, seed=7)
        self.assertEqual(noisy_a.tolist(), noisy_b.tolist())
        self.assertNotEqual(noisy_a.tolist(), list(y))
        self.assertAlmostEqual(float((noisy_a - y).std()), 0.1 * float(rrs.np.asarray(y).std()), delta=0.08)

    def test_group_metric_summary_reports_means_and_nan_steps(self):
        records = [
            {"method": "a", "test_mse": 1.0, "ood_mse": 2.0, "nan_steps": 0},
            {"method": "a", "test_mse": 3.0, "ood_mse": 4.0, "nan_steps": 2},
            {"method": "b", "test_mse": 10.0, "ood_mse": 20.0, "nan_steps": 0},
        ]
        rows = brs.group_metric_summary(records, ["method"])
        by_method = {row["method"]: row for row in rows}
        self.assertEqual(by_method["a"]["n"], 2)
        self.assertAlmostEqual(by_method["a"]["test_mse_mean"], 2.0)
        self.assertAlmostEqual(by_method["a"]["ood_mse_mean"], 3.0)
        self.assertEqual(by_method["a"]["nan_steps_sum"], 2)

    def test_write_csv_projects_extra_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.csv"
            brs.write_csv(path, [{"a": 1, "b": 2, "extra": 3}], ["a", "b"])
            self.assertEqual(path.read_text(encoding="utf-8").strip().splitlines(), ["a,b", "1,2"])


if __name__ == "__main__":
    unittest.main()
