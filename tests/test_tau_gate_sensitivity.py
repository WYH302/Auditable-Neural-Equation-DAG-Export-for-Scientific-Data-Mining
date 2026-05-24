import unittest

from scripts import build_tau_gate_sensitivity as sens


class TauGateSensitivityTests(unittest.TestCase):
    def test_variant_family_labels_tau_and_gate(self):
        self.assertEqual(sens.variant_family("tau4"), "temperature")
        self.assertEqual(sens.variant_family("no_gate_penalty"), "gate")
        self.assertEqual(sens.variant_family("raw_exp_d3"), "stability")

    def test_pick_rows_keeps_requested_configs_in_order(self):
        rows = [
            {"config": "tau4", "label": "tau=4"},
            {"config": "t2_current", "label": "T=2"},
        ]

        picked = sens.pick_rows(rows, ["t2_current", "tau4"])

        self.assertEqual([row["config"] for row in picked], ["t2_current", "tau4"])

    def test_render_table_mentions_existing_results(self):
        rows = [
            {
                "config": "tau4",
                "label": "tau=4",
                "n": 24,
                "test_mse": 0.1,
                "ood_mse": 0.2,
                "tokens": 1000,
                "gate_mean": 0.18,
                "effective_depth": 3.0,
                "nonfinite": 0,
                "nan_steps": 0,
                "family": "temperature",
            }
        ]

        table = sens.render_latex(rows)

        self.assertIn("Existing ablation", table)
        self.assertIn("tau=4", table)


if __name__ == "__main__":
    unittest.main()
