from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from tesseract_core import Tesseract, __version__ as tesseract_version

from tesstrace.core import DAG, Node


IMAGES = {
    "optics": "wingspectrum-optics:latest",
    "uvs_faulty": "wingspectrum-uvs-faulty:latest",
    "uvs_fixed": "wingspectrum-uvs-fixed:latest",
    "vs": "wingspectrum-vs:latest",
    "human": "wingspectrum-human:latest",
    "thermal": "wingspectrum-thermal:latest",
}
SETTINGS = {
    "initial_design": [-3.0] * 10,
    "optimizer": "Adam",
    "learning_rate": 0.16,
    "iterations": 180,
    "seed": 20260822,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "contract_epsilon": 1e-6,
    "contract_rtol": 1e-4,
    "contract_atol": 1e-7,
}


def apply(client, x):
    return np.asarray(client.apply({"x": x})["y"], dtype=float)


def vjp(client, x, cotangent):
    result = client.vector_jacobian_product(
        {"x": x}, vjp_inputs=["x"], vjp_outputs=["y"],
        cotangent_vector={"y": cotangent},
    )
    return np.asarray(result["x"], dtype=float)


def objective_and_gradient(design, clients, uvs_key):
    spectrum = apply(clients["optics"], design)
    uvs = float(apply(clients[uvs_key], spectrum)[0])
    vs = float(apply(clients["vs"], spectrum)[0])
    human = apply(clients["human"], spectrum)
    solar = float(apply(clients["thermal"], spectrum)[0])
    temperature = 0.035
    e_u, e_v = np.exp(-uvs / temperature), np.exp(-vs / temperature)
    smooth_min = -temperature * np.log(e_u + e_v)
    du, dv = e_u / (e_u + e_v), e_v / (e_u + e_v)
    regularization = float(np.mean(design**2))
    loss = (-2.4*smooth_min + 15.0*human[0]**2 + 1.5*human[1]
            + 0.22*solar + 0.003*regularization)
    spectrum_cotangent = vjp(clients[uvs_key], spectrum, np.array([-2.4*du]))
    spectrum_cotangent += vjp(clients["vs"], spectrum, np.array([-2.4*dv]))
    spectrum_cotangent += vjp(
        clients["human"], spectrum, np.array([30.0*human[0], 1.5])
    )
    spectrum_cotangent += vjp(clients["thermal"], spectrum, np.array([0.22]))
    gradient = vjp(clients["optics"], design, spectrum_cotangent)
    gradient += 0.006 * design / len(design)
    metrics = {
        "loss": float(loss), "uvs_visibility": uvs, "vs_visibility": vs,
        "human_reflectance": float(human[0]),
        "human_color_variation": float(human[1]), "solar_transmittance": solar,
    }
    return metrics, gradient, spectrum


def optimize(clients, uvs_key):
    design = np.asarray(SETTINGS["initial_design"], dtype=float)
    m = np.zeros_like(design); v = np.zeros_like(design); history = []
    initial_spectrum = None
    for step in range(SETTINGS["iterations"] + 1):
        metrics, gradient, spectrum = objective_and_gradient(design, clients, uvs_key)
        history.append({"step": step, **metrics})
        if initial_spectrum is None: initial_spectrum = spectrum.copy()
        if step == SETTINGS["iterations"]: break
        t = step + 1
        m = SETTINGS["adam_beta1"]*m + (1-SETTINGS["adam_beta1"])*gradient
        v = SETTINGS["adam_beta2"]*v + (1-SETTINGS["adam_beta2"])*gradient**2
        mh = m/(1-SETTINGS["adam_beta1"]**t)
        vh = v/(1-SETTINGS["adam_beta2"]**t)
        design -= SETTINGS["learning_rate"]*mh/(np.sqrt(vh)+SETTINGS["adam_epsilon"])
    return {
        "history": history, "final_design": design.tolist(),
        "initial_spectrum": initial_spectrum.tolist(), "final_spectrum": spectrum.tolist(),
    }


def remote_node(name, client):
    return Node(name, lambda x: apply(client, x), lambda x, c: vjp(client, x, c))


def contract_report(clients, uvs_key):
    design = np.asarray(SETTINGS["initial_design"])
    metrics, _, spectrum = objective_and_gradient(design, clients, uvs_key)
    uvs, vs = metrics["uvs_visibility"], metrics["vs_visibility"]
    temp = 0.035; eu, ev = np.exp(-uvs/temp), np.exp(-vs/temp)
    fixtures = {
        "optics": (design, np.linspace(-0.2, 0.3, 162), np.linspace(0.4, -0.3, 10)),
        "uvs": (spectrum, np.array([-2.4*eu/(eu+ev)]), np.linspace(0.3, -0.2, 162)),
        "vs": (spectrum, np.array([-2.4*ev/(eu+ev)]), np.linspace(-0.1, 0.25, 162)),
        "human": (spectrum, np.array([30*metrics["human_reflectance"], 1.5]), np.linspace(0.2, -0.15, 162)),
        "thermal": (spectrum, np.array([0.22]), np.linspace(-0.25, 0.1, 162)),
    }
    dag = DAG()
    mapping = {"optics":"optics", "uvs":uvs_key, "vs":"vs", "human":"human", "thermal":"thermal"}
    for name, key in mapping.items(): dag.add_node(remote_node(name, clients[key]))
    for branch in ("uvs", "vs", "human", "thermal"): dag.add_edge("optics", branch)
    return dag.scan(
        fixtures, epsilon=SETTINGS["contract_epsilon"],
        rtol=SETTINGS["contract_rtol"], atol=SETTINGS["contract_atol"],
    ).as_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        clients = {k: stack.enter_context(Tesseract.from_image(v, timeout=120)) for k,v in IMAGES.items()}
        faulty = optimize(clients, "uvs_faulty")
        fixed = optimize(clients, "uvs_fixed")
        faulty["contracts"] = contract_report(clients, "uvs_faulty")
        fixed["contracts"] = contract_report(clients, "uvs_fixed")
        probe = apply(clients["optics"], np.asarray(SETTINGS["initial_design"]))
        forward_equal = bool(np.array_equal(apply(clients["uvs_faulty"], probe), apply(clients["uvs_fixed"], probe)))
        endpoints = {k: sorted(c.available_endpoints) for k,c in clients.items()}
    faulty_exit = 1 if faulty["contracts"]["failed"] else 0
    fixed_exit = 1 if fixed["contracts"]["failed"] else 0
    summary = {
        "forward_equal": forward_equal, "faulty_exit_code": faulty_exit,
        "fixed_exit_code": fixed_exit, "tesseract_core_version": tesseract_version,
        "images": IMAGES, "endpoints": endpoints,
        "faulty_final": faulty["history"][-1], "fixed_final": fixed["history"][-1],
    }
    for name, payload in (("settings", SETTINGS), ("faulty", faulty), ("fixed", fixed), ("summary", summary)):
        (args.output_dir/f"{name}.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    expected_faulty = {"uvs":"FAIL", "vs":"PASS", "human":"PASS", "thermal":"PASS", "optics":"PARTIALLY_CONTAMINATED"}
    expected_fixed = {name:"PASS" for name in ("uvs","vs","human","thermal","optics")}
    passed = (forward_equal and faulty_exit == 1 and fixed_exit == 0
              and faulty["contracts"]["nodes"] == expected_faulty
              and fixed["contracts"]["nodes"] == expected_fixed
              and fixed["history"][-1]["uvs_visibility"] > faulty["history"][-1]["uvs_visibility"]
              and fixed["history"][-1]["human_reflectance"] < 0.20)
    return 0 if passed else 1


if __name__ == "__main__": raise SystemExit(main())
