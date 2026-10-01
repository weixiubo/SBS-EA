from __future__ import annotations

import gzip
import hashlib
import math
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graphprize import GraphPrizeInstance, GraphPrizeNode


SNAP_GRAPH_SOURCES: dict[str, dict[str, str]] = {
    "snap_email_eu_core": {
        "filename": "email-Eu-core.txt.gz",
        "url": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
        "info_url": "https://snap.stanford.edu/data/email-Eu-core.html",
    },
    "snap_ca_grqc": {
        "filename": "ca-GrQc.txt.gz",
        "url": "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
        "info_url": "https://snap.stanford.edu/data/ca-GrQc.html",
    },
}


@dataclass(frozen=True)
class PublicGraphCoverConfig:
    roots_per_family: int = 20
    smoke_per_family: int = 3
    val_per_family: int = 3
    selector_per_family: int = 6
    smoke_node_limit: int = 30
    val_node_limit: int = 30
    selector_node_limit: int = 34
    test_node_limit: int = 40
    smoke_budget: int = 11
    val_budget: int = 11
    selector_budget: int = 12
    test_budget: int = 14
    root_rank_fraction_cap: float = 0.85
    root_window: int = 25

    @property
    def test_per_family(self) -> int:
        return self.roots_per_family - self.smoke_per_family - self.val_per_family - self.selector_per_family


def ensure_snap_graph_files(root_dir: str | Path, *, download: bool = False) -> dict[str, Path]:
    data_dir = Path(root_dir) / "benchmarks" / "snap_graphs"
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for family, spec in SNAP_GRAPH_SOURCES.items():
        path = data_dir / spec["filename"]
        if not path.exists():
            missing.append(family)
            if download:
                urllib.request.urlretrieve(spec["url"], path)
        if path.exists():
            paths[family] = path
    if missing and not download:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"missing_snap_graph_files:{joined}; rerun with download=True")
    return paths


def build_snap_graphcover_dataset(
    root_dir: str | Path,
    *,
    config: PublicGraphCoverConfig | None = None,
    download: bool = False,
) -> tuple[dict[str, list[GraphPrizeInstance]], dict[str, Any]]:
    resolved = config or PublicGraphCoverConfig()
    if resolved.test_per_family <= 0:
        raise ValueError("roots_per_family must exceed smoke+val+selector counts")
    files = ensure_snap_graph_files(root_dir, download=download)
    dataset: dict[str, list[GraphPrizeInstance]] = {"smoke": [], "val": [], "selector": [], "test": []}
    graph_metadata: dict[str, Any] = {}
    cursor = 0
    for family, path in sorted(files.items()):
        adjacency = _largest_component(_load_edge_list(path))
        roots = _stratified_roots(adjacency, resolved)
        graph_metadata[family] = {
            "path": str(path),
            "sha256": _sha256(path),
            "node_count_largest_component": len(adjacency),
            "edge_count_largest_component": sum(len(values) for values in adjacency.values()) // 2,
            "root_count": len(roots),
            "roots": roots,
            "source": SNAP_GRAPH_SOURCES[family],
        }
        for offset, root in enumerate(roots):
            split, node_limit, budget = _split_params(offset, resolved)
            dataset[split].append(
                _build_instance(
                    family=family,
                    adjacency=adjacency,
                    root=root,
                    split=split,
                    seed=cursor,
                    node_limit=node_limit,
                    budget=budget,
                )
            )
            cursor += 1
    metadata = {
        "source": "SNAP public graph edge lists",
        "config": resolved.__dict__,
        "split_counts": {split: len(items) for split, items in dataset.items()},
        "graphs": graph_metadata,
        "objective": "budgeted connected prize collection on deterministic root-neighborhood subgraphs",
        "upper_bound": "sum of the top budget node prizes, ignoring connectivity and edge-cost constraints",
        "split_protocol": "smoke and val tune EA expressions; selector chooses the source; test is held out for final reporting",
    }
    return dataset, metadata


def _split_params(offset: int, config: PublicGraphCoverConfig) -> tuple[str, int, int]:
    if offset < config.smoke_per_family:
        return "smoke", config.smoke_node_limit, config.smoke_budget
    if offset < config.smoke_per_family + config.val_per_family:
        return "val", config.val_node_limit, config.val_budget
    if offset < config.smoke_per_family + config.val_per_family + config.selector_per_family:
        return "selector", config.selector_node_limit, config.selector_budget
    return "test", config.test_node_limit, config.test_budget


def _load_edge_list(path: Path) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            left, right = int(parts[0]), int(parts[1])
            if left == right:
                continue
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)
    return adjacency


def _largest_component(adjacency: dict[int, set[int]]) -> dict[int, set[int]]:
    seen: set[int] = set()
    best: list[int] = []
    for start in adjacency:
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        component: list[int] = []
        for node in queue:
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        if len(component) > len(best):
            best = component
    allowed = set(best)
    return {node: adjacency[node] & allowed for node in best}


def _stratified_roots(adjacency: dict[int, set[int]], config: PublicGraphCoverConfig) -> list[int]:
    ranked = sorted(adjacency, key=lambda node: (-len(adjacency[node]), node))
    root_count = config.roots_per_family
    roots: list[int] = []
    for index in range(root_count):
        fraction = (index + 1) / (root_count + 1)
        position = min(len(ranked) - 1, int(fraction * config.root_rank_fraction_cap * len(ranked)))
        left = max(0, position - config.root_window)
        right = min(len(ranked), position + config.root_window)
        window = ranked[left:right] or ranked
        roots.append(window[(index * 11 + 7) % len(window)])
    return roots


def _neighborhood(adjacency: dict[int, set[int]], root: int, limit: int) -> list[int]:
    selected: list[int] = []
    seen = {root}
    queue: deque[int] = deque([root])
    while queue and len(selected) < limit:
        node = queue.popleft()
        selected.append(node)
        ordered = sorted(adjacency[node], key=lambda item: (-len(adjacency[item]), item))
        if ordered:
            shift = (root + len(selected)) % len(ordered)
            ordered = [*ordered[shift:], *ordered[:shift]]
        for neighbor in ordered:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return selected[:limit]


def _build_instance(
    *,
    family: str,
    adjacency: dict[int, set[int]],
    root: int,
    split: str,
    seed: int,
    node_limit: int,
    budget: int,
) -> GraphPrizeInstance:
    old_nodes = [root, *(node for node in _neighborhood(adjacency, root, node_limit) if node != root)]
    old_to_new = {old: index for index, old in enumerate(old_nodes)}
    degrees = {old: len(adjacency[old]) for old in old_nodes}
    max_degree = max(degrees.values()) or 1
    nodes: list[GraphPrizeNode] = [GraphPrizeNode(0, 0.0, -1)]
    for old in old_nodes[1:]:
        local_neighbors = sum(1 for neighbor in adjacency[old] if neighbor in old_to_new)
        external_neighbors = max(0, len(adjacency[old]) - local_neighbors)
        prize = 1.0 + math.log1p(len(adjacency[old])) + 0.35 * math.sqrt(external_neighbors) + 0.15 * local_neighbors
        topic = min(7, int(8 * degrees[old] / (max_degree + 1)))
        nodes.append(GraphPrizeNode(old_to_new[old], round(prize, 4), topic))
    edges: list[tuple[int, int, int]] = []
    for old_left in old_nodes:
        for old_right in adjacency[old_left]:
            if old_right not in old_to_new:
                continue
            left = old_to_new[old_left]
            right = old_to_new[old_right]
            if left < right:
                degree_gap = abs(degrees[old_left] - degrees[old_right])
                cost = 1 + int((degree_gap + old_left + old_right + seed) % 4 == 0)
                edges.append((left, right, cost))
    return GraphPrizeInstance(
        family=family,
        split=split,
        seed=seed,
        instance_id=f"{split}-{family}-r{root}-n{len(nodes)}-b{budget}",
        nodes=tuple(nodes),
        adjacency=_build_adjacency(len(nodes), edges),
        budget=budget,
        upper_bound=_top_budget_prize_bound(nodes, budget),
    )


def _build_adjacency(node_count: int, edges: list[tuple[int, int, int]]) -> tuple[tuple[tuple[int, int], ...], ...]:
    rows: list[list[tuple[int, int]]] = [[] for _ in range(node_count)]
    for left, right, cost in edges:
        rows[left].append((right, cost))
        rows[right].append((left, cost))
    return tuple(tuple(sorted(row)) for row in rows)


def _top_budget_prize_bound(nodes: list[GraphPrizeNode], budget: int) -> float:
    prizes = sorted((node.prize for node in nodes if node.node_id != 0), reverse=True)
    return max(sum(prizes[:budget]), 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
