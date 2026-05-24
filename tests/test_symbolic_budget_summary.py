import unittest

from scripts import build_symbolic_budget_summary as budget


class SymbolicBudgetSummaryTests(unittest.TestCase):
    def test_summarize_reports_ok_and_exact_counts(self):
        rows = [
            {"status": "ok", "exact_recovery": 1, "test_mse": 1.0, "ood_mse": 4.0, "complexity": 3, "runtime_sec": 2.0},
            {"status": "failed", "exact_recovery": 0, "test_mse": "", "ood_mse": "", "complexity": "", "runtime_sec": 1.0},
        ]

        summary = budget.summarize_runs(rows)

        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["ok"], 1)
        self.assertEqual(summary["exact_count"], 1)
        self.assertEqual(summary["test_mse_mean"], 1.0)

    def test_budget_label_is_explicit(self):
        label = budget.budget_label(generations=20, population_size=1000)

        self.assertIn("20 gen", label)
        self.assertIn("1000 pop", label)


if __name__ == "__main__":
    unittest.main()
