from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from tesseract_core import Tesseract, __version__ as tesseract_version

from tesstrace.core import DAG, Node


IMAGES = {
    "optics": "tesstrace-optics:latest",
    "uvs_faulty": "tesstrace-uvs-faulty:latest",
    "uvs_fixed": "tesstrace-uvs-fixed:latest",
    "vs_vision": "tesstrace-vs:latest",
    "human_vision": "tesstrace-human:latest",
    "solar_thermal": "tesstrace-thermal:latest",
}


def remote_node(name: str, tesseract: Tesseract) -> Node:
    def forward(x: np.ndarray) -> np.ndarray:
        return np.asarray(tesseract.apply({"x": x})["y"], dtype=float)

    def vjp(x: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
        result = tesseract.vector_jacobian_product(
            {"x": x},
            vjp_inputs=["x"],
            vjp_outputs=["y"],
            cotangent_vector={"y": cotangent},
        )
        return np.asarray(result["x"], dtype=float)

    return Node(name, forward, vjp)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("faulty", "fixed"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    selected = {
        "optics": IMAGES["optics"],
        "uvs_vision": IMAGES[f"uvs_{args.variant}"],
        "vs_vision": IMAGES["vs_vision"],
        "human_vision": IMAGES["human_vision"],
        "solar_thermal": IMAGES["solar_thermal"],
    }

    endpoint_evidence = {}
    with ExitStack() as stack:
        clients = {
            name: stack.enter_context(Tesseract.from_image(image, timeout=120))
            for name, image in selected.items()
        }
        dag = DAG()
        for name, client in clients.items():
            dag.add_node(remote_node(name, client))
            endpoint_evidence[name] = {
                "image": selected[name],
                "available_endpoints": sorted(client.available_endpoints),
            }
        for branch in ("uvs_vision", "vs_vision", "human_vision", "solar_thermal"):
            dag.add_edge("optics", branch)

        fixtures = {
            "optics": (np.array([0.4, 0.6]), np.array([0.3, -0.2]), np.array([0.5, 0.7])),
            "uvs_vision": (np.array([0.5, 0.5]), np.array([0.4, -0.3]), np.array([0.8, 0.2])),
            "vs_vision": (np.array([0.5, 0.5]), np.array([0.2, 0.4]), np.array([0.1, 0.9])),
            "human_vision": (np.array([0.5, 0.5]), np.array([0.7]), np.array([0.6, 0.4])),
            "solar_thermal": (np.array([0.5, 0.5]), np.array([-0.5]), np.array([0.3, 0.7])),
        }
        report = dag.scan(fixtures)

    payload = report.as_dict()
    payload["fixture_variant"] = args.variant
    payload["tesseract_core_version"] = tesseract_version
    payload["endpoint_evidence"] = endpoint_evidence
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

