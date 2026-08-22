from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    faulty = load(sys.argv[1])
    fixed = load(sys.argv[2])

    assert faulty["nodes"]["uvs_vision"] == "FAIL"
    assert faulty["nodes"]["vs_vision"] == "PASS"
    assert faulty["nodes"]["human_vision"] == "PASS"
    assert faulty["nodes"]["solar_thermal"] == "PASS"
    assert faulty["nodes"]["optics"] == "PARTIALLY_CONTAMINATED"
    assert faulty["edges"]["optics->uvs_vision"] == "CONTAMINATED"
    assert faulty["edges"]["optics->vs_vision"] == "PASS"
    assert faulty["edges"]["optics->human_vision"] == "PASS"
    assert faulty["edges"]["optics->solar_thermal"] == "PASS"
    assert all(status == "PASS" for status in fixed["nodes"].values())
    assert all(status == "PASS" for status in fixed["edges"].values())

    for report in (faulty, fixed):
        assert report["tesseract_core_version"] == "1.11.0"
        for evidence in report["endpoint_evidence"].values():
            endpoints = set(evidence["available_endpoints"])
            assert {"apply", "vector_jacobian_product"} <= endpoints
        for check in report["checks"].values():
            assert isinstance(check["analytic"], float)
            assert isinstance(check["finite_difference"], float)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

