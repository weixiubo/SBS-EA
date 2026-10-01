from __future__ import annotations

import json
import re
import zipfile
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig, default_config
from .download_utils import download_file
from .jssp import JSSPInstance, parse_jsplib_operations, rollout_dispatch_policy
from .jssp_heuristics import dispatch_seed_policies
from .ranking import EloState, kendall_tau, spearman_correlation, update_elo


JSPLIB_ZIP_URL = "https://github.com/tamy0612/JSPLIB/archive/refs/heads/master.zip"
JSPLIB_OVERVIEW_URL = "https://scheduleopt.github.io/benchmarks/jsplib/"
FAMILY_RE = re.compile(r"^[A-Za-z]+")


@dataclass(frozen=True)
class JSPLibInstanceMeta:
    name: str
    jobs: int
    machines: int
    optimum: int | None
    lower_bound: int | None
    upper_bound: int | None
    family: str
    source_path: str


@dataclass(frozen=True)
class JSPLibInstance:
    name: str
    family: str
    jobs: int
    machines: int
    operations: tuple
    optimum: int | None
    lower_bound: int | None
    upper_bound: int | None
    source_path: str


def jsplib_root(config: AppConfig) -> Path:
    return (
        config.runtime.root_dir
        / config.public_benchmarks.root_dir_name
        / config.public_benchmarks.jsplib_dir_name
    )


def ensure_jsplib_repo(target_dir: str | Path) -> Path:
    root = Path(target_dir)
    if (root / "instances.json").exists():
        return root
    extracted = root / "JSPLIB-master"
    if (extracted / "instances.json").exists():
        return extracted
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "jsplib-master.zip"
    download_file(JSPLIB_ZIP_URL, archive_path)
    with zipfile.ZipFile(archive_path) as bundle:
        bundle.extractall(root)
    extracted = root / "JSPLIB-master"
    if (extracted / "instances.json").exists():
        return extracted
    raise FileNotFoundError("downloaded JSPLIB archive does not contain instances.json")


def _family_from_name(name: str) -> str:
    match = FAMILY_RE.match(name)
    if match is None:
        return "unknown"
    return match.group(0).lower()


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def load_jsplib_metadata(
    target_dir: str | Path,
    *,
    limit: int | None = None,
) -> list[JSPLibInstanceMeta]:
    root = Path(target_dir)
    if not (root / "instances.json").exists():
        root = ensure_jsplib_repo(root)
    payload = json.loads((root / "instances.json").read_text(encoding="utf-8"))
    metadata: list[JSPLibInstanceMeta] = []
    for raw in payload:
        name = str(raw.get("name", "")).strip()
        jobs = _coerce_int(raw.get("jobs"))
        machines = _coerce_int(raw.get("machines"))
        relative_path = str(raw.get("path", "")).strip()
        if not name or jobs is None or machines is None or jobs <= 0 or machines <= 0:
            continue
        optimum = _coerce_int(raw.get("optimum"))
        bounds = raw.get("bounds") or {}
        metadata.append(
            JSPLibInstanceMeta(
                name=name,
                jobs=jobs,
                machines=machines,
                optimum=optimum,
                lower_bound=_coerce_int(bounds.get("lower")),
                upper_bound=_coerce_int(bounds.get("upper")),
                family=_family_from_name(name),
                source_path=str((root / relative_path).resolve()) if relative_path else "",
            )
        )
        if limit is not None and len(metadata) >= limit:
            break
    return metadata


def load_jsplib_instances(
    target_dir: str | Path,
    *,
    include_families: tuple[str, ...] | None = None,
    limit_per_family: int = 3,
    return_errors: bool = False,
) -> list[JSSPInstance] | tuple[list[JSSPInstance], list[str]]:
    root = Path(target_dir)
    if not (root / "instances.json").exists():
        root = ensure_jsplib_repo(root)
    picked: dict[str, list[JSSPInstance]] = {}
    errors: list[str] = []
    for meta in load_jsplib_metadata(root):
        if include_families is not None and meta.family not in include_families:
            continue
        if len(picked.setdefault(meta.family, [])) >= limit_per_family:
            continue
        try:
            parsed = parse_jsplib_instance(meta)
        except Exception as exc:
            errors.append(f"{meta.name}:{type(exc).__name__}:{exc}")
            continue
        picked[meta.family].append(parsed)
    instances: list[JSSPInstance] = []
    for family in sorted(picked):
        instances.extend(picked[family])
    if return_errors:
        return instances, errors
    return instances


def load_project_jsplib_instances(
    root_dir: str | Path,
    *,
    limit_per_family: int | None = None,
) -> list[JSSPInstance]:
    config = default_config(root_dir)
    resolved_root = ensure_jsplib_repo(jsplib_root(config))
    include_families = getattr(config.public_benchmarks, "jsplib_families", None)
    if limit_per_family is None:
        limit_per_family = (
            config.public_benchmarks.jsplib_smoke_per_family
            + config.public_benchmarks.jsplib_val_per_family
            + config.public_benchmarks.jsplib_test_per_family
        )
    return load_jsplib_instances(
        resolved_root,
        include_families=include_families,
        limit_per_family=limit_per_family,
    )


def parse_jsplib_instance(
    meta: JSPLibInstanceMeta,
    *,
    allow_zero_duration: bool = False,
) -> JSSPInstance:
    source_path = Path(meta.source_path)
    jobs, machines, operations = parse_jsplib_operations(
        source_path.read_text(encoding="utf-8", errors="ignore"),
        expected_jobs=meta.jobs,
        expected_machines=meta.machines,
        allow_zero_duration=allow_zero_duration,
    )
    return JSSPInstance(
        family=meta.family,
        split="public",
        seed=0,
        instance_id=meta.name,
        jobs=jobs,
        machines=machines,
        operations=operations,
        optimum=meta.optimum,
        lower_bound=meta.lower_bound,
        upper_bound=meta.upper_bound,
    )


def build_jsplib_dataset(
    instances: list[JSSPInstance],
    *,
    smoke_per_family: int = 1,
    val_per_family: int = 1,
    test_per_family: int = 1,
    family_rotation: int = 0,
) -> dict[str, list[JSSPInstance]]:
    grouped: dict[str, list[JSSPInstance]] = {}
    for instance in instances:
        grouped.setdefault(instance.family, []).append(instance)
    dataset: dict[str, list[JSSPInstance]] = {"smoke": [], "val": [], "test": []}
    cursor = 0
    for family in sorted(grouped):
        family_instances = list(grouped[family])
        if family_instances and family_rotation:
            offset = family_rotation % len(family_instances)
            family_instances = family_instances[offset:] + family_instances[:offset]
        smoke = family_instances[:smoke_per_family]
        val = family_instances[smoke_per_family : smoke_per_family + val_per_family]
        test = family_instances[
            smoke_per_family + val_per_family : smoke_per_family + val_per_family + test_per_family
        ]
        for split_name, selected in (("smoke", smoke), ("val", val), ("test", test)):
            for instance in selected:
                dataset[split_name].append(
                    replace(
                        instance,
                        split=split_name,
                        seed=cursor,
                        instance_id=f"{split_name}-{instance.family}-{instance.instance_id}",
                    )
                )
                cursor += 1
    return dataset


def load_project_jsplib_dataset(
    root_dir: str | Path,
    *,
    family_rotation: int = 0,
) -> dict[str, list[JSSPInstance]]:
    config = default_config(root_dir)
    instances = load_project_jsplib_instances(root_dir)
    return build_jsplib_dataset(
        instances,
        smoke_per_family=config.public_benchmarks.jsplib_smoke_per_family,
        val_per_family=config.public_benchmarks.jsplib_val_per_family,
        test_per_family=config.public_benchmarks.jsplib_test_per_family,
        family_rotation=family_rotation,
    )


def jsplib_rotation_capacity(
    instances: list[JSSPInstance],
    *,
    smoke_per_family: int = 1,
    val_per_family: int = 1,
    test_per_family: int = 1,
) -> int:
    required = smoke_per_family + val_per_family + test_per_family
    if required <= 0:
        return 0
    grouped: dict[str, int] = {}
    for instance in instances:
        grouped[instance.family] = grouped.get(instance.family, 0) + 1
    if not grouped:
        return 0
    return min(grouped.values())


def project_jsplib_rotation_capacity(root_dir: str | Path) -> int:
    config = default_config(root_dir)
    instances = load_project_jsplib_instances(root_dir)
    return jsplib_rotation_capacity(
        instances,
        smoke_per_family=config.public_benchmarks.jsplib_smoke_per_family,
        val_per_family=config.public_benchmarks.jsplib_val_per_family,
        test_per_family=config.public_benchmarks.jsplib_test_per_family,
    )


def build_jsplib_rotation_datasets(
    instances: list[JSSPInstance],
    *,
    smoke_per_family: int = 1,
    val_per_family: int = 1,
    test_per_family: int = 1,
    num_rotations: int | None = None,
) -> list[dict[str, list[JSSPInstance]]]:
    capacity = jsplib_rotation_capacity(
        instances,
        smoke_per_family=smoke_per_family,
        val_per_family=val_per_family,
        test_per_family=test_per_family,
    )
    if capacity <= 0:
        return []
    rotation_count = capacity if num_rotations is None else max(0, min(int(num_rotations), capacity))
    return [
        build_jsplib_dataset(
            instances,
            smoke_per_family=smoke_per_family,
            val_per_family=val_per_family,
            test_per_family=test_per_family,
            family_rotation=rotation,
        )
        for rotation in range(rotation_count)
    ]


def _collect_dispatch_rule_rows(
    dataset: dict[str, list[JSSPInstance]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, float]]]:
    policies = dispatch_seed_policies()
    split_scores: dict[str, dict[str, float]] = {"smoke": {}, "val": {}, "test": {}}
    rule_rows: dict[str, dict[str, Any]] = {}
    for rule_name, policy in policies.items():
        row: dict[str, Any] = {}
        for split_name, instances in dataset.items():
            makespans: list[int] = []
            family_scores: dict[str, list[int]] = {}
            for instance in instances:
                result = rollout_dispatch_policy(policy, instance)
                if not result.success or result.makespan is None:
                    raise RuntimeError(f"{rule_name} failed on {instance.instance_id}: {result.error}")
                makespans.append(result.makespan)
                family_scores.setdefault(instance.family, []).append(result.makespan)
            mean_makespan = round(sum(makespans) / len(makespans), 4) if makespans else None
            split_scores[split_name][rule_name] = 10**9 if mean_makespan is None else mean_makespan
            row[f"{split_name}_mean_makespan"] = mean_makespan
            row[f"{split_name}_family_means"] = {
                family: round(sum(values) / len(values), 4)
                for family, values in family_scores.items()
            }
        rule_rows[rule_name] = row
    return rule_rows, split_scores


def jsplib_dispatch_preview(dataset: dict[str, list[JSSPInstance]]) -> dict[str, Any]:
    rule_rows, split_scores = _collect_dispatch_rule_rows(dataset)
    smoke_scores = split_scores["smoke"]
    test_scores = split_scores["test"]
    best_rule = min(test_scores, key=test_scores.get)
    worst_rule = max(test_scores, key=test_scores.get)
    return {
        "rule_count": len(rule_rows),
        "rules": rule_rows,
        "smoke_test_spearman": spearman_correlation(smoke_scores, test_scores),
        "smoke_test_kendall": kendall_tau(smoke_scores, test_scores),
        "test_headroom_gap": round(test_scores[worst_rule] - test_scores[best_rule], 4),
        "best_rule_by_test": best_rule,
        "worst_rule_by_test": worst_rule,
    }


def _proxy_pointwise_scores(rule_rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for rule_name, row in rule_rows.items():
        smoke_mean = row.get("smoke_mean_makespan")
        val_mean = row.get("val_mean_makespan")
        if smoke_mean is None and val_mean is None:
            scores[rule_name] = 10**9
            continue
        if val_mean is None:
            scores[rule_name] = float(smoke_mean)
            continue
        if smoke_mean is None:
            scores[rule_name] = float(val_mean)
            continue
        scores[rule_name] = round(0.7 * float(smoke_mean) + 0.3 * float(val_mean), 4)
    return scores


def _pairwise_vote(left: float | None, right: float | None) -> int:
    if left is None or right is None:
        return 0
    if abs(float(left) - float(right)) <= 1e-9:
        return 0
    return 1 if float(left) < float(right) else -1


def _proxy_match_summary(
    left_name: str,
    left_row: dict[str, Any],
    right_name: str,
    right_row: dict[str, Any],
) -> dict[str, Any]:
    margin = 0.0
    evidence_used = 0
    split_wins = {"smoke": "tie", "val": "tie"}
    family_wins = {"smoke": {"left": 0, "right": 0, "tie": 0}, "val": {"left": 0, "right": 0, "tie": 0}}
    for split_name, weight in (("smoke", 2.0), ("val", 2.0)):
        vote = _pairwise_vote(
            left_row.get(f"{split_name}_mean_makespan"),
            right_row.get(f"{split_name}_mean_makespan"),
        )
        if vote > 0:
            split_wins[split_name] = left_name
            margin += weight
            evidence_used += 1
        elif vote < 0:
            split_wins[split_name] = right_name
            margin -= weight
            evidence_used += 1
        left_family = left_row.get(f"{split_name}_family_means", {})
        right_family = right_row.get(f"{split_name}_family_means", {})
        for family in sorted(set(left_family) | set(right_family)):
            family_vote = _pairwise_vote(left_family.get(family), right_family.get(family))
            if family_vote > 0:
                family_wins[split_name]["left"] += 1
                margin += 1.0
                evidence_used += 1
            elif family_vote < 0:
                family_wins[split_name]["right"] += 1
                margin -= 1.0
                evidence_used += 1
            else:
                family_wins[split_name]["tie"] += 1
    if margin > 0.0:
        winner = left_name
    elif margin < 0.0:
        winner = right_name
    else:
        winner = "tie"
    return {
        "left": left_name,
        "right": right_name,
        "winner": winner,
        "margin": round(margin, 4),
        "evidence_used": evidence_used,
        "split_wins": split_wins,
        "family_wins": family_wins,
    }


def jsplib_stage_a_proxy(dataset: dict[str, list[JSSPInstance]]) -> dict[str, Any]:
    if not dataset.get("test"):
        return {}
    rule_rows, split_scores = _collect_dispatch_rule_rows(dataset)
    pointwise_scores = _proxy_pointwise_scores(rule_rows)
    oracle_scores = split_scores["test"]
    ratings: dict[str, EloState] = {
        rule_name: EloState()
        for rule_name in sorted(rule_rows)
    }
    pairwise_matches: list[dict[str, Any]] = []
    ordered_rules = sorted(rule_rows)
    for index, left_name in enumerate(ordered_rules):
        for right_name in ordered_rules[index + 1 :]:
            match = _proxy_match_summary(
                left_name,
                rule_rows[left_name],
                right_name,
                rule_rows[right_name],
            )
            ratings = update_elo(
                ratings,
                left_name,
                right_name,
                match["winner"],
                k=24.0,
            )
            pairwise_matches.append(match)
    pairwise_scores = {
        rule_name: -state.rating
        for rule_name, state in ratings.items()
    }
    return {
        "candidate_count": len(rule_rows),
        "pointwise_scores": pointwise_scores,
        "pairwise_final_scores": pairwise_scores,
        "oracle_test_scores": oracle_scores,
        "pointwise_kendall": kendall_tau(pointwise_scores, oracle_scores),
        "pointwise_spearman": spearman_correlation(pointwise_scores, oracle_scores),
        "pairwise_final_kendall": kendall_tau(pairwise_scores, oracle_scores),
        "pairwise_final_spearman": spearman_correlation(pairwise_scores, oracle_scores),
        "kendall_gain": round(
            kendall_tau(pairwise_scores, oracle_scores)
            - kendall_tau(pointwise_scores, oracle_scores),
            6,
        ),
        "spearman_gain": round(
            spearman_correlation(pairwise_scores, oracle_scores)
            - spearman_correlation(pointwise_scores, oracle_scores),
            6,
        ),
        "pointwise_best_rule": min(pointwise_scores, key=pointwise_scores.get),
        "pairwise_best_rule": min(pairwise_scores, key=pairwise_scores.get),
        "oracle_best_rule": min(oracle_scores, key=oracle_scores.get),
        "pairwise_matches": pairwise_matches,
    }


def jsplib_corpus_summary(config: AppConfig, *, download: bool = False) -> dict[str, Any]:
    root = jsplib_root(config)
    if download:
        try:
            resolved_root = ensure_jsplib_repo(root)
        except Exception as exc:
            return {
                "suite": "jsplib",
                "official_corpus_available": False,
                "corpus_root": str(root),
                "source_url": JSPLIB_ZIP_URL,
                "overview_url": JSPLIB_OVERVIEW_URL,
                "paper_ready": False,
                "reason": "download_failed",
                "download_error": f"{type(exc).__name__}: {exc}",
            }
    elif (root / "instances.json").exists():
        resolved_root = root
    elif (root / "JSPLIB-master" / "instances.json").exists():
        resolved_root = root / "JSPLIB-master"
    else:
        return {
            "suite": "jsplib",
            "official_corpus_available": False,
            "corpus_root": str(root),
            "source_url": JSPLIB_ZIP_URL,
            "overview_url": JSPLIB_OVERVIEW_URL,
            "paper_ready": False,
            "reason": "corpus_missing",
        }
    metadata = load_jsplib_metadata(
        resolved_root,
        limit=config.public_benchmarks.jsplib_limit,
    )
    family_counts: dict[str, int] = {}
    optimum_known_count = 0
    bounded_only_count = 0
    missing_file_count = 0
    for instance in metadata:
        family_counts[instance.family] = family_counts.get(instance.family, 0) + 1
        if instance.optimum is not None:
            optimum_known_count += 1
        elif instance.lower_bound is not None or instance.upper_bound is not None:
            bounded_only_count += 1
        if not instance.source_path or not Path(instance.source_path).exists():
            missing_file_count += 1
    sample_instances = [
        {
            "name": instance.name,
            "family": instance.family,
            "jobs": instance.jobs,
            "machines": instance.machines,
            "optimum": instance.optimum,
            "lower_bound": instance.lower_bound,
            "upper_bound": instance.upper_bound,
        }
        for instance in metadata[:5]
    ]
    parsed_instances, parse_errors = load_jsplib_instances(
        resolved_root,
        limit_per_family=(
            config.public_benchmarks.jsplib_smoke_per_family
            + config.public_benchmarks.jsplib_val_per_family
            + config.public_benchmarks.jsplib_test_per_family
        ),
        return_errors=True,
    )
    dataset = build_jsplib_dataset(
        parsed_instances,
        smoke_per_family=config.public_benchmarks.jsplib_smoke_per_family,
        val_per_family=config.public_benchmarks.jsplib_val_per_family,
        test_per_family=config.public_benchmarks.jsplib_test_per_family,
    )
    rotation_capacity = jsplib_rotation_capacity(
        parsed_instances,
        smoke_per_family=config.public_benchmarks.jsplib_smoke_per_family,
        val_per_family=config.public_benchmarks.jsplib_val_per_family,
        test_per_family=config.public_benchmarks.jsplib_test_per_family,
    )
    return {
        "suite": "jsplib",
        "official_corpus_available": bool(metadata),
        "corpus_root": str(resolved_root),
        "source_url": JSPLIB_ZIP_URL,
        "overview_url": JSPLIB_OVERVIEW_URL,
        "instance_count": len(metadata),
        "family_counts": dict(sorted(family_counts.items())),
        "optimum_known_count": optimum_known_count,
        "bounded_only_count": bounded_only_count,
        "missing_file_count": missing_file_count,
        "parse_failure_count": len(parse_errors),
        "parse_failure_examples": parse_errors[:5],
        "default_dataset_counts": {split: len(rows) for split, rows in dataset.items()},
        "default_rotation_capacity": rotation_capacity,
        "dispatch_preview": jsplib_dispatch_preview(dataset) if dataset["test"] else {},
        "stage_a_proxy": jsplib_stage_a_proxy(dataset) if dataset["test"] else {},
        "sample_instances": sample_instances,
        "paper_ready": bool(metadata) and missing_file_count == 0,
        "reason": "ok" if metadata else "corpus_empty",
    }
