import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class UiEvidenceTests(unittest.TestCase):
    def test_ui_embeds_verified_comparison_without_changing_claims(self):
        source = (ROOT / "ui" / "data.js").read_text(encoding="utf-8")
        prefix = "window.WING_SPECTRUM_EVIDENCE = "
        self.assertTrue(source.startswith(prefix))
        payload = json.loads(source[len(prefix):].removesuffix(";\n"))
        faulty = payload["faulty"]["history"][-1]
        fixed = payload["fixed"]["history"][-1]
        self.assertEqual(payload["version"], "1.11.0")
        self.assertTrue(payload["forwardEqual"])
        self.assertEqual(payload["faulty"]["contracts"]["nodes"]["uvs"], "FAIL")
        self.assertEqual(payload["fixed"]["contracts"]["nodes"]["uvs"], "PASS")
        self.assertGreater(fixed["uvs_visibility"], faulty["uvs_visibility"])
        self.assertGreater(fixed["vs_visibility"], faulty["vs_visibility"])
        self.assertLess(fixed["human_reflectance"], 0.20)


if __name__ == "__main__":
    unittest.main()
