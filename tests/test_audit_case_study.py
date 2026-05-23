import unittest

from scripts import build_audit_case_study as case_study


class AuditCaseStudyTests(unittest.TestCase):
    def test_parse_active_gate_notes(self):
        notes = "active_edges=16; active_terms=32"

        parsed = case_study.parse_active_notes(notes)

        self.assertEqual(parsed["active_edges"], 16)
        self.assertEqual(parsed["active_terms"], 32)

    def test_first_assignments_keeps_final_output_when_requested(self):
        text = "a = x1 + 1;\nb = a*a;\ny = b + a"

        snippet = case_study.first_assignments(text, n=1, include_final=True)

        self.assertEqual(snippet, ["a = x1 + 1", "y = b + a"])

    def test_render_latex_has_case_study_label(self):
        rows = [
            {
                "method_label": "MLP",
                "raw_tokens": 10,
                "local_tokens": 9,
                "global_cse_tokens": 8,
                "full_tokens": 12,
                "assignments": 3,
                "max_rhs_depth": 4,
                "canonical_ms": 1.2,
                "active_edges": 0,
                "active_terms": 0,
                "basis_terms": 0,
            }
        ]

        text = case_study.render_latex_table("toy", rows)

        self.assertIn(r"\label{tab:audit-case-study}", text)
        self.assertIn("MLP", text)


if __name__ == "__main__":
    unittest.main()
