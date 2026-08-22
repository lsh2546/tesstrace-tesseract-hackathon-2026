from __future__ import annotations

import argparse
import json
from pathlib import Path

from .demo import wing_spectrum_contract_fixture


def main() -> int:
    parser = argparse.ArgumentParser(description="TessTrace gradient-contract verifier")
    parser.add_argument("--fixed", action="store_true", help="Use the corrected UVS VJP")
    parser.add_argument("--json", type=Path, help="Write a machine-readable report")
    args = parser.parse_args()

    dag, fixtures = wing_spectrum_contract_fixture(faulty_uvs=not args.fixed)
    report = dag.scan(fixtures)
    payload = report.as_dict()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

