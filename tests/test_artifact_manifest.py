import unittest

from scripts import build_artifact_manifest as manifest


class ArtifactManifestTests(unittest.TestCase):
    def test_manifest_contains_required_review_audits(self):
        rows = manifest.manifest_rows()
        names = {row["name"] for row in rows}

        self.assertIn("export_fidelity_details", names)
        self.assertIn("token_sensitivity", names)
        self.assertIn("ood_failure_curves", names)
        self.assertIn("tau_gate_sensitivity", names)

    def test_markdown_mentions_reproduction_command(self):
        rows = [{"name": "toy", "command": "python toy.py", "inputs": "in.csv", "outputs": "out.csv", "paper_items": "Table X"}]

        text = manifest.render_markdown(rows)

        self.assertIn("python toy.py", text)
        self.assertIn("Table X", text)


if __name__ == "__main__":
    unittest.main()
