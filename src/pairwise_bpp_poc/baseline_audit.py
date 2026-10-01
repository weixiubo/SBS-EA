from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from .config import AppConfig, default_config
from .io_utils import ensure_dir, write_json
from .remote import RemoteManager, default_remote_connection


REQUIRED_RESULT_FIELDS = (
    "objective_mean",
    "relative_gap_to_best_known",
    "llm_call_count",
    "wall_clock_seconds",
    "gpu_hours",
    "peak_vram_gb",
    "seed",
    "run_tag",
)
DEFAULT_FRONTIER_COST_KEY = "wall_clock_seconds"
DEFAULT_FRONTIER_OBJECTIVE_KEY = "objective_mean"
COMPARISON_TASK = "bpp"
COMPARISON_BENCHMARK_SUITE = "bpplib"


@dataclass(frozen=True)
class BaselineSpec:
    baseline_name: str
    paper_title: str
    paper_url: str
    venue: str
    accepted_year: int
    repo_url: str
    task: str
    model_name: str
    budget_mode: str
    integration_mode: str
    notes: str = ""


BASELINE_SPECS: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        baseline_name="calm_iclr2026",
        paper_title="CALM: Co-evolution of Algorithms and Language Model for Automatic Heuristic Design",
        paper_url="https://openreview.net/forum?id=x6bG2Hoqdf",
        venue="ICLR",
        accepted_year=2026,
        repo_url="https://github.com/whxru/CALM.git",
        task="bpp",
        model_name="pending",
        budget_mode="small",
        integration_mode="official_repo",
        notes="Primary latest-tier baseline. Runs in an isolated environment.",
    ),
    BaselineSpec(
        baseline_name="mcts_ahd_icml2025",
        paper_title="Monte Carlo Tree Search for Comprehensive Exploration in LLM-Based Automatic Heuristic Design",
        paper_url="https://proceedings.mlr.press/v267/zheng25o.html",
        venue="ICML",
        accepted_year=2025,
        repo_url="https://github.com/Optima-CityU/llm4ad.git",
        task="bpp",
        model_name="pending",
        budget_mode="small",
        integration_mode="llm4ad_mcts_ahd",
        notes="Preferred via LLM4AD to reduce one-off integration work.",
    ),
    BaselineSpec(
        baseline_name="reevo_neurips2024",
        paper_title="ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution",
        paper_url="https://openreview.net/pdf?id=483IPG0HWL",
        venue="NeurIPS",
        accepted_year=2024,
        repo_url="https://github.com/ai4co/reevo.git",
        task="bpp",
        model_name="pending",
        budget_mode="small",
        integration_mode="official_repo",
        notes="Strong older A-tier baseline kept in the main pool.",
    ),
    BaselineSpec(
        baseline_name="partevo_neurips2025",
        paper_title="Partition to Evolve: Niching-enhanced Evolution with LLMs for Automated Algorithm Discovery",
        paper_url="https://openreview.net/forum?id=OEawM2coNT",
        venue="NeurIPS",
        accepted_year=2025,
        repo_url="https://github.com/Optima-CityU/llm4ad.git",
        task="bpp",
        model_name="pending",
        budget_mode="small",
        integration_mode="llm4ad_partevo",
        notes="Preferred backup latest-tier baseline.",
    ),
    BaselineSpec(
        baseline_name="eoh_icml2024",
        paper_title="Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model",
        paper_url="https://proceedings.mlr.press/v235/liu24bs.html",
        venue="ICML",
        accepted_year=2024,
        repo_url="https://github.com/Optima-CityU/llm4ad.git",
        task="bpp",
        model_name="pending",
        budget_mode="small",
        integration_mode="llm4ad_eoh",
        notes="Fallback if PartEvo integration stalls.",
    ),
)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _proxy_free_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "all_proxy",
    ):
        env.pop(key, None)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    return env


def baseline_audit_root(root_dir: str | Path) -> Path:
    return ensure_dir(Path(root_dir) / "_baseline_audit")


def local_baseline_dir(root_dir: str | Path, baseline_name: str) -> Path:
    return ensure_dir(baseline_audit_root(root_dir) / baseline_name)


def remote_baseline_dir(config: AppConfig, baseline_name: str) -> str:
    remote_root = os.environ.get("POC_BASELINE_AUDIT_REMOTE_ROOT", "baseline_audit").strip()
    if not remote_root:
        remote_root = "baseline_audit"
    return f"{remote_root.rstrip('/')}/{baseline_name}"


def manifest_path(root_dir: str | Path, baseline_name: str) -> Path:
    return local_baseline_dir(root_dir, baseline_name) / "manifest.json"


def status_path(root_dir: str | Path, baseline_name: str) -> Path:
    return local_baseline_dir(root_dir, baseline_name) / "status.json"


def results_dir(root_dir: str | Path, baseline_name: str) -> Path:
    return ensure_dir(local_baseline_dir(root_dir, baseline_name) / "results")


def repo_dir(root_dir: str | Path, baseline_name: str) -> Path:
    return local_baseline_dir(root_dir, baseline_name) / "repo"


def baseline_specs_by_name() -> dict[str, BaselineSpec]:
    return {spec.baseline_name: spec for spec in BASELINE_SPECS}


def probe_repo_head_sha(repo_url: str) -> str:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            proc = subprocess.run(
                ["git", "ls-remote", repo_url, "HEAD"],
                capture_output=True,
                text=True,
                env=_proxy_free_env(),
                check=True,
            )
            line = proc.stdout.strip().splitlines()[0]
            return line.split()[0]
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 + attempt)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"failed to resolve HEAD for {repo_url}")


def _github_tarball_parts(repo_url: str) -> tuple[str, str] | None:
    parsed = urlparse(repo_url)
    if parsed.netloc not in {"github.com", "www.github.com"}:
        return None
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        return None
    owner = parts[0]
    repo = parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo:
        return None
    return owner, repo


def _download_github_tarball(repo_url: str, target_dir: Path, ref: str) -> str:
    parts = _github_tarball_parts(repo_url)
    if parts is None:
        raise RuntimeError(f"unsupported_repo_archive_source:{repo_url}")
    owner, repo = parts
    archive_url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/{ref}"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(archive_url, timeout=120) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as bundle:
            extract_root: Path | None = None
            temp_extract_dir = target_dir.parent / f".{target_dir.name}_extract"
            if temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir)
            temp_extract_dir.mkdir(parents=True, exist_ok=True)
            try:
                bundle.extractall(temp_extract_dir)
                extracted_children = [child for child in temp_extract_dir.iterdir() if child.is_dir()]
                if len(extracted_children) != 1:
                    raise RuntimeError(f"unexpected_archive_layout:{archive_url}")
                extract_root = extracted_children[0]
                extract_root.rename(target_dir)
            finally:
                if temp_extract_dir.exists():
                    shutil.rmtree(temp_extract_dir)
    (target_dir / ".repo_source").write_text(
        json.dumps(
            {
                "repo_url": repo_url,
                "ref": ref,
                "source": "github_tarball",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ref


def clone_or_update_repo(repo_url: str, target_dir: Path) -> str:
    try:
        if (target_dir / ".git").exists():
            subprocess.run(
                ["git", "-C", str(target_dir), "fetch", "--depth", "1", "origin", "HEAD"],
                capture_output=True,
                text=True,
                env=_proxy_free_env(),
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(target_dir), "checkout", "--detach", "FETCH_HEAD"],
                capture_output=True,
                text=True,
                env=_proxy_free_env(),
                check=True,
            )
        else:
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
                capture_output=True,
                text=True,
                env=_proxy_free_env(),
                check=True,
            )
        proc = subprocess.run(
            ["git", "-C", str(target_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            env=_proxy_free_env(),
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        ref = probe_repo_head_sha(repo_url)
        return _download_github_tarball(repo_url, target_dir, ref)


def build_manifest_payload(
    spec: BaselineSpec,
    *,
    root_dir: str | Path,
    config: AppConfig | None = None,
    commit_sha: str = "",
) -> dict[str, Any]:
    effective_config = default_config(root_dir) if config is None else config
    baseline_dir = local_baseline_dir(effective_config.runtime.root_dir, spec.baseline_name)
    return {
        "baseline_name": spec.baseline_name,
        "paper_title": spec.paper_title,
        "paper_url": spec.paper_url,
        "venue": spec.venue,
        "accepted_year": spec.accepted_year,
        "repo_url": spec.repo_url,
        "commit_sha": commit_sha,
        "task": spec.task,
        "model_name": spec.model_name,
        "budget_mode": spec.budget_mode,
        "integration_mode": spec.integration_mode,
        "local_dir": str(baseline_dir),
        "remote_dir": remote_baseline_dir(effective_config, spec.baseline_name),
        "notes": spec.notes,
        "result_schema_fields": list(REQUIRED_RESULT_FIELDS),
        "updated_at": _utc_now(),
    }


def build_status_payload(
    spec: BaselineSpec,
    *,
    repo_ready: bool,
    commit_sha: str,
    state: str,
    last_error: str = "",
) -> dict[str, Any]:
    return {
        "baseline_name": spec.baseline_name,
        "state": state,
        "repo_ready": repo_ready,
        "smoke_status": "pending",
        "medium_status": "pending",
        "seed_repeat_status": "pending",
        "commit_sha": commit_sha,
        "last_error": last_error,
        "updated_at": _utc_now(),
    }


def initialize_baseline_layout(
    root_dir: str | Path,
    *,
    clone_repos: bool = False,
) -> dict[str, Any]:
    config = default_config(root_dir)
    created: list[dict[str, Any]] = []
    for spec in BASELINE_SPECS:
        baseline_dir = local_baseline_dir(config.runtime.root_dir, spec.baseline_name)
        ensure_dir(baseline_dir / "results")
        commit_sha = ""
        repo_ready = False
        state = "initialized"
        last_error = ""
        if clone_repos:
            try:
                commit_sha = clone_or_update_repo(spec.repo_url, repo_dir(config.runtime.root_dir, spec.baseline_name))
                repo_ready = True
                state = "repo_ready"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                state = "repo_error"
        else:
            try:
                commit_sha = probe_repo_head_sha(spec.repo_url)
                state = "manifest_only"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                state = "manifest_only"
        manifest = build_manifest_payload(
            spec,
            root_dir=config.runtime.root_dir,
            config=config,
            commit_sha=commit_sha,
        )
        status = build_status_payload(
            spec,
            repo_ready=repo_ready,
            commit_sha=commit_sha,
            state=state,
            last_error=last_error,
        )
        write_json(manifest_path(config.runtime.root_dir, spec.baseline_name), manifest)
        write_json(status_path(config.runtime.root_dir, spec.baseline_name), status)
        created.append(
            {
                "baseline_name": spec.baseline_name,
                "manifest_path": str(manifest_path(config.runtime.root_dir, spec.baseline_name)),
                "status_path": str(status_path(config.runtime.root_dir, spec.baseline_name)),
                "commit_sha": commit_sha,
                "state": state,
            }
        )
    return {
        "root_dir": str(config.runtime.root_dir),
        "baseline_audit_root": str(baseline_audit_root(config.runtime.root_dir)),
        "created": created,
    }


def sync_manifests_to_remote(root_dir: str | Path) -> dict[str, Any]:
    config = default_config(root_dir)
    uploaded: list[dict[str, str]] = []
    with RemoteManager(default_remote_connection(config)) as manager:
        manager.run(f"mkdir -p {shlex.quote(remote_baseline_dir(config, '.')).rstrip('/')}")
        for spec in BASELINE_SPECS:
            remote_dir = remote_baseline_dir(config, spec.baseline_name)
            manager.run(f"mkdir -p {remote_dir}")
            local_manifest = manifest_path(config.runtime.root_dir, spec.baseline_name)
            local_status = status_path(config.runtime.root_dir, spec.baseline_name)
            manager.upload_file(local_manifest, f"{remote_dir}/manifest.json")
            manager.upload_file(local_status, f"{remote_dir}/status.json")
            uploaded.append(
                {
                    "baseline_name": spec.baseline_name,
                    "remote_dir": remote_dir,
                }
            )
    return {
        "uploaded": uploaded,
    }


def validate_result_payload(payload: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_RESULT_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing_result_fields:{','.join(missing)}")


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_internal_unified_results(artifacts_root: str | Path) -> list[dict[str, Any]]:
    root = Path(artifacts_root)
    rows: list[dict[str, Any]] = []
    for result_path in sorted(root.glob("*/unified_result.json")):
        payload = load_json_if_exists(result_path)
        if not payload:
            continue
        validate_result_payload(payload)
        payload = dict(payload)
        payload["source_kind"] = "internal"
        payload["result_path"] = str(result_path)
        rows.append(payload)
    return rows


def collect_external_results(root_dir: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in BASELINE_SPECS:
        manifest = load_json_if_exists(manifest_path(root_dir, spec.baseline_name)) or {}
        for result_path in sorted(results_dir(root_dir, spec.baseline_name).glob("*.json")):
            payload = load_json_if_exists(result_path)
            if not payload:
                continue
            validate_result_payload(payload)
            payload = dict(payload)
            payload.setdefault("baseline_name", spec.baseline_name)
            payload.setdefault("task", manifest.get("task", spec.task))
            payload.setdefault("paper_title", manifest.get("paper_title", spec.paper_title))
            payload.setdefault("paper_url", manifest.get("paper_url", spec.paper_url))
            payload.setdefault("venue", manifest.get("venue", spec.venue))
            payload.setdefault("accepted_year", manifest.get("accepted_year", spec.accepted_year))
            payload.setdefault("repo_url", manifest.get("repo_url", spec.repo_url))
            payload.setdefault("commit_sha", manifest.get("commit_sha", ""))
            payload["source_kind"] = "external"
            payload["result_path"] = str(result_path)
            rows.append(payload)
    return rows


def _numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return float(text)


def _frontier(rows: list[dict[str, Any]], *, cost_key: str, objective_key: str) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            row for row in rows
            if _numeric_or_none(row.get(cost_key)) is not None
            and _numeric_or_none(row.get(objective_key)) is not None
        ),
        key=lambda row: (
            float(row[cost_key]),
            float(row[objective_key]),
            str(row.get("method_name") or row.get("baseline_name") or ""),
        ),
    )
    frontier: list[dict[str, Any]] = []
    best_objective = float("inf")
    for row in ranked:
        objective = float(row[objective_key])
        if objective < best_objective:
            frontier.append(row)
            best_objective = objective
    return frontier


def _format_number(value: Any, digits: int = 4) -> str:
    numeric = _numeric_or_none(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.{digits}f}"


def _row_in_comparison_scope(row: dict[str, Any]) -> bool:
    task_name = str(row.get("task") or "").strip().lower()
    if task_name != COMPARISON_TASK:
        return False
    benchmark_suite = str(row.get("benchmark_suite") or "").strip().lower()
    if benchmark_suite and benchmark_suite != COMPARISON_BENCHMARK_SUITE:
        return False
    return True


def _render_frontier_svg(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    cost_key: str = DEFAULT_FRONTIER_COST_KEY,
    objective_key: str = DEFAULT_FRONTIER_OBJECTIVE_KEY,
) -> None:
    eligible = [
        row for row in rows
        if _numeric_or_none(row.get(cost_key)) is not None
        and _numeric_or_none(row.get(objective_key)) is not None
    ]
    width = 960
    height = 540
    margin_left = 80
    margin_bottom = 70
    margin_top = 30
    margin_right = 30
    if not eligible:
        svg = (
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
            "<rect width='100%' height='100%' fill='white'/>"
            "<text x='40' y='60' font-size='22' font-family='monospace'>No eligible results yet.</text>"
            "</svg>"
        )
        output_path.write_text(svg, encoding="utf-8")
        return
    x_values = [float(row[cost_key]) for row in eligible]
    y_values = [float(row[objective_key]) for row in eligible]
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)
    if min_x == max_x:
        max_x = min_x + 1.0
    if min_y == max_y:
        max_y = min_y + 1.0

    def scale_x(value: float) -> float:
        return margin_left + (value - min_x) / (max_x - min_x) * (width - margin_left - margin_right)

    def scale_y(value: float) -> float:
        return height - margin_bottom - (value - min_y) / (max_y - min_y) * (height - margin_top - margin_bottom)

    frontier_rows = _frontier(eligible, cost_key=cost_key, objective_key=objective_key)
    polyline = " ".join(
        f"{scale_x(float(row[cost_key])):.2f},{scale_y(float(row[objective_key])):.2f}"
        for row in frontier_rows
    )
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<line x1='{margin_left}' y1='{height - margin_bottom}' x2='{width - margin_right}' y2='{height - margin_bottom}' stroke='#333' stroke-width='2'/>",
        f"<line x1='{margin_left}' y1='{height - margin_bottom}' x2='{margin_left}' y2='{margin_top}' stroke='#333' stroke-width='2'/>",
        f"<text x='{width / 2:.0f}' y='{height - 20}' text-anchor='middle' font-size='16' font-family='monospace'>{cost_key}</text>",
        f"<text x='24' y='{height / 2:.0f}' text-anchor='middle' transform='rotate(-90 24 {height / 2:.0f})' font-size='16' font-family='monospace'>{objective_key}</text>",
        "<text x='40' y='22' font-size='18' font-family='monospace'>BPP performance-cost frontier</text>",
    ]
    if len(frontier_rows) >= 2:
        parts.append(
            f"<polyline fill='none' stroke='#0b6efd' stroke-width='2.5' points='{polyline}'/>"
        )
    for row in eligible:
        x = scale_x(float(row[cost_key]))
        y = scale_y(float(row[objective_key]))
        color = "#0b6efd" if row.get("source_kind") == "internal" else "#d97706"
        label = str(row.get("method_name") or row.get("baseline_name") or row.get("run_tag") or "run")
        parts.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='5.5' fill='{color}'/>")
        parts.append(
            f"<text x='{x + 8:.2f}' y='{y - 8:.2f}' font-size='11' font-family='monospace'>{label}</text>"
        )
    parts.append("</svg>")
    output_path.write_text("".join(parts), encoding="utf-8")


def aggregate_bpp_performance_vs_cost(
    root_dir: str | Path,
    *,
    artifacts_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = default_config(root_dir)
    effective_artifacts_root = Path(artifacts_root or config.runtime.artifacts_dir)
    effective_output_dir = ensure_dir(Path(output_dir or (baseline_audit_root(config.runtime.root_dir) / "outputs")))
    manifests: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for spec in BASELINE_SPECS:
        manifest = load_json_if_exists(manifest_path(config.runtime.root_dir, spec.baseline_name)) or build_manifest_payload(
            spec,
            root_dir=config.runtime.root_dir,
            config=config,
        )
        status = load_json_if_exists(status_path(config.runtime.root_dir, spec.baseline_name)) or build_status_payload(
            spec,
            repo_ready=False,
            commit_sha=str(manifest.get("commit_sha", "")),
            state="missing",
        )
        manifests.append(manifest)
        statuses.append(status)

    all_rows = collect_internal_unified_results(effective_artifacts_root) + collect_external_results(config.runtime.root_dir)
    scoped_rows = [row for row in all_rows if _row_in_comparison_scope(row)]
    valid_rows = [
        row for row in scoped_rows
        if _numeric_or_none(row.get(DEFAULT_FRONTIER_OBJECTIVE_KEY)) is not None
        and _numeric_or_none(row.get(DEFAULT_FRONTIER_COST_KEY)) is not None
    ]
    frontier_by_time = _frontier(valid_rows, cost_key="wall_clock_seconds", objective_key="objective_mean")
    frontier_by_calls = _frontier(valid_rows, cost_key="llm_call_count", objective_key="objective_mean")
    strongest_baseline = None
    external_rows = [row for row in valid_rows if row.get("source_kind") == "external"]
    if external_rows:
        strongest_baseline = min(
            external_rows,
            key=lambda row: (
                float(row["objective_mean"]),
                float(row["wall_clock_seconds"]),
                str(row.get("baseline_name") or ""),
            ),
        )
    payload = {
        "generated_at": _utc_now(),
        "artifacts_root": str(effective_artifacts_root),
        "output_dir": str(effective_output_dir),
        "comparison_scope": {
            "task": COMPARISON_TASK,
            "benchmark_suite": COMPARISON_BENCHMARK_SUITE,
        },
        "result_schema_fields": list(REQUIRED_RESULT_FIELDS),
        "baseline_manifests": manifests,
        "baseline_statuses": statuses,
        "runs": scoped_rows,
        "skipped_runs": [row for row in all_rows if not _row_in_comparison_scope(row)],
        "frontiers": {
            "by_wall_clock_seconds": frontier_by_time,
            "by_llm_call_count": frontier_by_calls,
        },
        "strongest_reproduced_baseline": strongest_baseline,
        "success_criteria": {
            "max_relative_objective_regression": 0.01,
            "min_cost_drop_ratio": 0.5,
        },
    }
    write_json(effective_output_dir / "performance_vs_cost.json", payload)
    summary_lines = [
        "# BPP SOTA Reproduction Summary",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Artifacts root: `{effective_artifacts_root}`",
        f"- Comparison scope: `{COMPARISON_TASK}/{COMPARISON_BENCHMARK_SUITE}`",
        f"- Valid runs collected: `{len(valid_rows)}`",
        f"- Skipped out-of-scope runs: `{len(payload['skipped_runs'])}`",
        "",
        "## Baseline Status",
        "",
        "| Baseline | Venue | Year | State | Repo Ready | Last Error |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for manifest, status in zip(manifests, statuses, strict=True):
        summary_lines.append(
            "| "
            + " | ".join(
                (
                    str(manifest.get("baseline_name", "")),
                    str(manifest.get("venue", "")),
                    str(manifest.get("accepted_year", "")),
                    str(status.get("state", "")),
                    "yes" if status.get("repo_ready") else "no",
                    str(status.get("last_error", "")).replace("|", "/"),
                )
            )
            + " |"
        )
    summary_lines.extend(
        [
            "",
            "## Valid Runs",
            "",
            "| Method | Source | Objective | Gap | LLM Calls | Wall Clock (s) | GPU Hours | Peak VRAM (GB) | Seed |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in sorted(
        valid_rows,
        key=lambda item: (
            float(item["objective_mean"]),
            float(item["wall_clock_seconds"]),
            str(item.get("method_name") or item.get("baseline_name") or ""),
        ),
    ):
        summary_lines.append(
            "| "
            + " | ".join(
                (
                    str(row.get("method_name") or row.get("baseline_name") or row.get("run_tag") or ""),
                    str(row.get("source_kind", "")),
                    _format_number(row.get("objective_mean")),
                    _format_number(row.get("relative_gap_to_best_known")),
                    _format_number(row.get("llm_call_count"), digits=0),
                    _format_number(row.get("wall_clock_seconds")),
                    _format_number(row.get("gpu_hours")),
                    _format_number(row.get("peak_vram_gb")),
                    str(row.get("seed", "")),
                )
            )
            + " |"
        )
    if not valid_rows:
        summary_lines.extend(
            [
                "",
                "No valid performance-cost runs are available yet. Pending and failed baselines are still listed above.",
            ]
        )
    (effective_output_dir / "repro_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    _render_frontier_svg(valid_rows, effective_output_dir / "performance_cost_frontier.svg")
    return payload
