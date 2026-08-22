from __future__ import annotations

import numpy as np

from .core import DAG, Node


def _linear(name: str, matrix: np.ndarray, *, faulty: bool = False) -> Node:
    matrix = np.asarray(matrix, dtype=float)

    def forward(x: np.ndarray) -> np.ndarray:
        return matrix @ x

    def vjp(_: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
        result = matrix.T @ cotangent
        if faulty:
            result = result[::-1]
        return result

    return Node(name, forward, vjp)


def wing_spectrum_contract_fixture(*, faulty_uvs: bool = True):
    dag = DAG()
    dag.add_node(_linear("optics", np.array([[1.0, 0.2], [0.1, 0.9]])))
    dag.add_node(_linear("uvs_vision", np.array([[1.4, 0.2], [0.0, 0.6]]), faulty=faulty_uvs))
    dag.add_node(_linear("vs_vision", np.array([[0.7, 0.3], [0.2, 0.8]])))
    dag.add_node(_linear("human_vision", np.array([[0.3, 0.7]])))
    dag.add_node(_linear("solar_thermal", np.array([[0.8, 0.2]])))
    for branch in ("uvs_vision", "vs_vision", "human_vision", "solar_thermal"):
        dag.add_edge("optics", branch)

    fixtures = {
        "optics": (np.array([0.4, 0.6]), np.array([0.3, -0.2]), np.array([0.5, 0.7])),
        "uvs_vision": (np.array([0.5, 0.5]), np.array([0.4, -0.3]), np.array([0.8, 0.2])),
        "vs_vision": (np.array([0.5, 0.5]), np.array([0.2, 0.4]), np.array([0.1, 0.9])),
        "human_vision": (np.array([0.5, 0.5]), np.array([0.7]), np.array([0.6, 0.4])),
        "solar_thermal": (np.array([0.5, 0.5]), np.array([-0.5]), np.array([0.3, 0.7])),
    }
    return dag, fixtures

