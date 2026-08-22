from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Forward = Callable[[Array], Array]
VJP = Callable[[Array, Array], Array]


class ContractStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    CONTAMINATED = "CONTAMINATED"
    PARTIALLY_CONTAMINATED = "PARTIALLY_CONTAMINATED"
    UNTESTED = "UNTESTED"


@dataclass(frozen=True)
class Node:
    name: str
    forward: Forward
    vjp: VJP


@dataclass(frozen=True)
class LocalCheck:
    node: str
    status: ContractStatus
    analytic: float | None
    finite_difference: float | None
    absolute_error: float | None
    relative_error: float | None


@dataclass
class GradientContractReport:
    checks: dict[str, LocalCheck]
    node_status: dict[str, ContractStatus]
    edge_status: dict[tuple[str, str], ContractStatus]

    @property
    def failed(self) -> bool:
        return any(check.status == ContractStatus.FAIL for check in self.checks.values())

    def as_dict(self) -> dict[str, object]:
        return {
            "failed": self.failed,
            "checks": {
                name: {
                    "status": check.status.value,
                    "analytic": check.analytic,
                    "finite_difference": check.finite_difference,
                    "absolute_error": check.absolute_error,
                    "relative_error": check.relative_error,
                }
                for name, check in self.checks.items()
            },
            "nodes": {name: status.value for name, status in self.node_status.items()},
            "edges": {
                f"{source}->{target}": status.value
                for (source, target), status in self.edge_status.items()
            },
        }


@dataclass
class DAG:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: set[tuple[str, str]] = field(default_factory=set)

    def add_node(self, node: Node) -> None:
        self.nodes[node.name] = node

    def add_edge(self, source: str, target: str) -> None:
        if source not in self.nodes or target not in self.nodes:
            raise KeyError("Both edge endpoints must exist before adding an edge")
        self.edges.add((source, target))

    def children(self, name: str) -> set[str]:
        return {target for source, target in self.edges if source == name}

    def parents(self, name: str) -> set[str]:
        return {source for source, target in self.edges if target == name}

    def local_check(
        self,
        name: str,
        x: Array,
        cotangent: Array,
        direction: Array,
        *,
        epsilon: float = 1e-6,
        rtol: float = 1e-4,
        atol: float = 1e-7,
    ) -> LocalCheck:
        node = self.nodes[name]
        try:
            analytic_vjp = np.asarray(node.vjp(x, cotangent), dtype=float)
            analytic = float(np.vdot(analytic_vjp, direction))
            plus = np.asarray(node.forward(x + epsilon * direction), dtype=float)
            minus = np.asarray(node.forward(x - epsilon * direction), dtype=float)
            finite_difference = float(np.vdot(cotangent, (plus - minus) / (2 * epsilon)))
        except Exception:
            return LocalCheck(name, ContractStatus.UNTESTED, None, None, None, None)

        absolute_error = abs(analytic - finite_difference)
        scale = max(abs(analytic), abs(finite_difference), atol)
        relative_error = absolute_error / scale
        passed = bool(np.isfinite([analytic, finite_difference]).all()) and bool(
            np.isclose(analytic, finite_difference, rtol=rtol, atol=atol)
        )
        status = ContractStatus.PASS if passed else ContractStatus.FAIL
        return LocalCheck(
            name,
            status,
            analytic,
            finite_difference,
            absolute_error,
            relative_error,
        )

    def scan(
        self,
        fixtures: dict[str, tuple[Array, Array, Array]],
        *,
        epsilon: float = 1e-6,
        rtol: float = 1e-4,
        atol: float = 1e-7,
    ) -> GradientContractReport:
        checks = {
            name: self.local_check(
                name,
                *fixture,
                epsilon=epsilon,
                rtol=rtol,
                atol=atol,
            )
            for name, fixture in fixtures.items()
        }
        node_status = {
            name: checks.get(
                name,
                LocalCheck(name, ContractStatus.UNTESTED, None, None, None, None),
            ).status
            for name in self.nodes
        }
        edge_status = {edge: ContractStatus.PASS for edge in self.edges}

        failed_nodes = {name for name, check in checks.items() if check.status == ContractStatus.FAIL}
        affected_children: dict[str, set[str]] = {}
        frontier = list(failed_nodes)
        visited = set(failed_nodes)
        while frontier:
            child = frontier.pop()
            for parent in self.parents(child):
                affected_children.setdefault(parent, set()).add(child)
                edge_status[(parent, child)] = ContractStatus.CONTAMINATED
                if parent not in visited:
                    visited.add(parent)
                    frontier.append(parent)

        for parent, directly_affected in affected_children.items():
            if node_status[parent] == ContractStatus.FAIL:
                continue
            all_children = self.children(parent)
            affected = {child for child in all_children if child in visited}
            if affected and affected != all_children:
                node_status[parent] = ContractStatus.PARTIALLY_CONTAMINATED
            elif affected:
                node_status[parent] = ContractStatus.CONTAMINATED

        return GradientContractReport(checks, node_status, edge_status)

