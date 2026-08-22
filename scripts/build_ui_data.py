from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    faulty = json.loads((args.artifact_dir / "faulty.json").read_text(encoding="utf-8"))
    fixed = json.loads((args.artifact_dir / "fixed.json").read_text(encoding="utf-8"))
    settings = json.loads((args.artifact_dir / "settings.json").read_text(encoding="utf-8"))
    summary = json.loads((args.artifact_dir / "summary.json").read_text(encoding="utf-8"))
    environment = json.loads((args.artifact_dir / "environment.json").read_text(encoding="utf-8"))
    wavelengths = list(range(300, 1101, 10))

    def compact_run(run: dict) -> dict:
        spectrum = run["final_spectrum"]
        initial = run["initial_spectrum"]
        return {
            "history": run["history"],
            "finalDesign": run["final_design"],
            "initialReflectance": initial[:81],
            "initialTransmittance": initial[81:],
            "reflectance": spectrum[:81],
            "transmittance": spectrum[81:],
            "contracts": run["contracts"],
        }

    payload = {
        "runUrl": environment["workflow_run_url"],
        "artifact": "wingspectrum-container-comparison-32576243257",
        "version": summary["tesseract_core_version"],
        "forwardEqual": summary["forward_equal"],
        "settings": settings,
        "imageShas": environment["images"],
        "wavelengths": wavelengths,
        "faulty": compact_run(faulty),
        "fixed": compact_run(fixed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "window.WING_SPECTRUM_EVIDENCE = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
