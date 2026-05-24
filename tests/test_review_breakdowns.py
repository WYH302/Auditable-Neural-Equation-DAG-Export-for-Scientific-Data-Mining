import unittest

from scripts import build_review_breakdowns as breakdowns


class ReviewBreakdownTests(unittest.TestCase):
    def test_sign_test_is_two_sided(self):
        self.assertAlmostEqual(breakdowns.sign_test_p(3, 0), 0.25)

    def test_ratio_summary_counts_winners(self):
        ratios = [0.5, 2.0, 1.0]

        summary = breakdowns.summarize_ratios("toy", ratios, bootstrap_reps=100)

        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["eml_wins"], 1)
        self.assertEqual(summary["mlp_wins"], 1)
        self.assertEqual(summary["ties"], 1)

    def test_select_worst_cases_sorts_descending(self):
        rows = [
            {"dataset": "a", "ratio": 2.0},
            {"dataset": "b", "ratio": 5.0},
            {"dataset": "c", "ratio": 1.5},
        ]

        worst = breakdowns.select_worst_cases(rows, n=2)

        self.assertEqual([row["dataset"] for row in worst], ["b", "a"])


if __name__ == "__main__":
    unittest.main()
