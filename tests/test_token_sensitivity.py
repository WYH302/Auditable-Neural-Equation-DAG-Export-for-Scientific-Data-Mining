import unittest

from scripts import build_token_sensitivity as sensitivity


class TokenSensitivityTests(unittest.TestCase):
    def test_round_numeric_literals_keeps_token_count_stable(self):
        text = "v1 = 1.23456789*x1 + -0.000123456789;"

        rounded = sensitivity.round_numeric_literals(text, digits=4)

        self.assertEqual(rounded, "v1 = 1.235*x1 + -0.0001235;")
        self.assertEqual(sensitivity.token_count(text), sensitivity.token_count(rounded))

    def test_split_assignments_handles_semicolon_inside_rbf_call(self):
        text = "h = 0.1*rbf(x1;-3,2.5);\ny = h + 1"

        statements = sensitivity.split_assignments(text)

        self.assertEqual(statements, ["h = 0.1*rbf(x1;-3,2.5)", "y = h + 1"])

    def test_inline_assignments_expands_final_output(self):
        text = "a = x1 + 1;\nb = a*a;\ny = b + a"

        expanded = sensitivity.inline_final_expression(text)

        self.assertEqual(expanded, "(((x1 + 1)*(x1 + 1)) + (x1 + 1))")

    def test_global_cse_keeps_equivalent_repeated_subexpression_once(self):
        text = "a = x1 + 1;\nb = x1 + 1;\ny = a + b"

        cse_text, ok = sensitivity.global_cse_text(text)

        self.assertTrue(ok)
        self.assertIn("cse0 = x1 + 1", cse_text)

    def test_non_assignment_formula_is_not_canonicalized_to_empty_text(self):
        text = "sin(x1) + x2^2"

        canonical, depth, assignments, ok, _elapsed_ms = sensitivity.canonicalize_local(text)

        self.assertTrue(ok)
        self.assertGreater(sensitivity.token_count(canonical), 0)
        self.assertEqual(assignments, 0)
        self.assertGreater(depth, 0)

    def test_latex_table_excludes_symbolic_search_reference(self):
        summary = [
            {"method": "MLP", "setting": "raw_core", "tokens_mean": 10},
            {"method": "PySR", "setting": "raw_core", "tokens_mean": 2},
        ]

        text = sensitivity.render_latex_table(summary)

        self.assertIn("MLP", text)
        self.assertNotIn("PySR", text)


if __name__ == "__main__":
    unittest.main()
