import math
import unittest

from scripts import build_export_fidelity_details as fidelity
from symbolic_export import graph_export


class ExportFidelityDetailsTests(unittest.TestCase):
    def test_summarize_errors_reports_mse_mae_and_max_abs(self):
        summary = fidelity.summarize_errors(
            y_checkpoint=[1.0, 2.0, 3.0],
            y_export=[1.0, 1.5, 4.0],
        )

        self.assertEqual(summary["n_total"], 3)
        self.assertEqual(summary["n_finite"], 3)
        self.assertAlmostEqual(summary["mse"], (0.0**2 + 0.5**2 + 1.0**2) / 3.0)
        self.assertAlmostEqual(summary["mae"], (0.0 + 0.5 + 1.0) / 3.0)
        self.assertAlmostEqual(summary["max_abs"], 1.0)

    def test_summarize_errors_ignores_nonfinite_pairs(self):
        summary = fidelity.summarize_errors(
            y_checkpoint=[1.0, math.nan, 3.0, 5.0],
            y_export=[1.5, 2.0, math.inf, 4.0],
        )

        self.assertEqual(summary["n_total"], 4)
        self.assertEqual(summary["n_finite"], 2)
        self.assertAlmostEqual(summary["mse"], (0.5**2 + 1.0**2) / 2.0)
        self.assertAlmostEqual(summary["mae"], (0.5 + 1.0) / 2.0)
        self.assertAlmostEqual(summary["max_abs"], 1.0)

    def test_exact_export_scope_excludes_non_exact_references(self):
        self.assertTrue(fidelity.is_exact_neural_export("mlp"))
        self.assertTrue(fidelity.is_exact_neural_export("kan"))
        self.assertTrue(fidelity.is_exact_neural_export("eml_kan"))
        self.assertFalse(fidelity.is_exact_neural_export("stable_eml"))
        self.assertFalse(fidelity.is_exact_neural_export("pysr"))

    def test_status_from_tolerance_distinguishes_rounded_zero(self):
        self.assertEqual(fidelity.status_from_tolerance(9.9e-13, 1e-12), "within_tolerance")
        self.assertEqual(fidelity.status_from_tolerance(1.1e-12, 1e-12), "exceeds_tolerance")
        self.assertEqual(fidelity.status_from_tolerance(math.nan, 1e-12), "not_evaluated")

    def test_evaluate_assignment_graph_uses_serialized_preprocessing(self):
        text = "x1_norm = (x1 - 1)/2;\ny_norm = 3*x1_norm;\ny = 4*y_norm + 5"

        values = fidelity.evaluate_assignment_graph(text, [[3.0], [5.0]])

        self.assertEqual(values.tolist(), [17.0, 29.0])

    def test_exported_float_format_preserves_audit_precision(self):
        self.assertEqual(graph_export.fmt(1.23456789), "1.2345679")
        self.assertEqual(graph_export.fmt(0.000123456789), "0.00012345679")


if __name__ == "__main__":
    unittest.main()
