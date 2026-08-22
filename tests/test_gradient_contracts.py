import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tesstrace.core import ContractStatus
from tesstrace.demo import wing_spectrum_contract_fixture


class GradientContractTests(unittest.TestCase):
    def test_fault_isolated_to_uvs_branch(self):
        dag, fixtures = wing_spectrum_contract_fixture(faulty_uvs=True)
        report = dag.scan(fixtures)

        self.assertEqual(report.node_status["uvs_vision"], ContractStatus.FAIL)
        self.assertEqual(report.node_status["vs_vision"], ContractStatus.PASS)
        self.assertEqual(report.node_status["human_vision"], ContractStatus.PASS)
        self.assertEqual(report.node_status["solar_thermal"], ContractStatus.PASS)
        self.assertEqual(report.node_status["optics"], ContractStatus.PARTIALLY_CONTAMINATED)
        self.assertEqual(
            report.edge_status[("optics", "uvs_vision")], ContractStatus.CONTAMINATED
        )
        self.assertEqual(report.edge_status[("optics", "vs_vision")], ContractStatus.PASS)
        self.assertGreater(report.checks["uvs_vision"].relative_error, 0.01)

    def test_corrected_dag_fully_passes(self):
        dag, fixtures = wing_spectrum_contract_fixture(faulty_uvs=False)
        report = dag.scan(fixtures)

        self.assertFalse(report.failed)
        self.assertTrue(
            all(status == ContractStatus.PASS for status in report.node_status.values())
        )
        self.assertTrue(
            all(status == ContractStatus.PASS for status in report.edge_status.values())
        )

    def test_cli_returns_ci_exit_code_and_saves_directional_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "report.json"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = "src"
            result = subprocess.run(
                [sys.executable, "-m", "tesstrace.cli", "--json", str(report_path)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            payload = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertNotEqual(result.returncode, 0)
        self.assertIsInstance(payload["checks"]["uvs_vision"]["analytic"], float)
        self.assertIsInstance(
            payload["checks"]["uvs_vision"]["finite_difference"], float
        )
        self.assertEqual(payload["nodes"]["optics"], "PARTIALLY_CONTAMINATED")



if __name__ == "__main__":
    unittest.main()
