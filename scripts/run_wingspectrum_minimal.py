from __future__ import annotations

import argparse
import json
from pathlib import Path

from wingspectrum import optimize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = optimize()
    initial = result["history"][0]
    final = result["history"][-1]
    checks = {
        "loss_improved": final["loss"] < initial["loss"],
        "uvs_improved": final["uvs_visibility"] > initial["uvs_visibility"],
        "vs_improved": final["vs_visibility"] > initial["vs_visibility"],
        "human_reflectance_bounded": final["human_reflectance"] < 0.20,
        "solar_transmittance_improved": final["solar_transmittance"] < initial["solar_transmittance"],
    }
    payload = {"checks": checks, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "initial": initial, "final": final}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

