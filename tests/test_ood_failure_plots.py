import unittest
from pathlib import Path

from scripts import build_ood_failure_plots as plots


class OODFailurePlotTests(unittest.TestCase):
    def test_latest_result_prefers_newest_timestamp(self):
        files = [
            Path("synthetic_inverse_quadratic_mlp_seed0_20260508_204408.json"),
            Path("synthetic_inverse_quadratic_mlp_seed0_20260509_010000.json"),
        ]

        self.assertEqual(plots.latest_result(files).name, files[1].name)

    def test_dataset_dimension_from_csv_header(self):
        self.assertEqual(plots.input_columns(["x1", "y"]), ["x1"])
        self.assertEqual(plots.input_columns(["x1", "x2", "y"]), ["x1", "x2"])

    def test_selected_cases_are_one_dimensional(self):
        self.assertTrue(plots.is_one_dimensional_case("inverse_quadratic", seed=0))
        self.assertFalse(plots.is_one_dimensional_case("poly_2d_x2_2y_1", seed=0))


if __name__ == "__main__":
    unittest.main()
