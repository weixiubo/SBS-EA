from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairwise_bpp_poc.io_utils import ensure_dir, write_json_atomic
from pairwise_bpp_poc.semantic_prior import (
    normalize_prior_budget,
    parse_prior_proposal,
    prompt_for_semantic_prior,
)


PROTOCOL_VERSION = "semantic_prior_ollama_one_call_v1"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(text)


def _get_json(url: str, *, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ollama_registry_response_not_object")
    return payload


def _post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError("ollama_chat_response_not_object")
    return result


def _api_url(chat_endpoint: str, leaf: str) -> str:
    parsed = urllib.parse.urlsplit(chat_endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid_ollama_endpoint:{chat_endpoint}")
    if parsed.path.rstrip("/") != "/api/chat":
        raise ValueError(f"ollama_endpoint_must_end_api_chat:{chat_endpoint}")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, f"/api/{leaf}", "", ""))


def _model_entry(endpoint: str, model: str, expected_digest: str, *, timeout: float) -> dict[str, Any]:
    registry = _get_json(_api_url(endpoint, "tags"), timeout=timeout)
    matches = [
        dict(item)
        for item in registry.get("models", [])
        if isinstance(item, dict) and model in {str(item.get("name", "")), str(item.get("model", ""))}
    ]
    if len(matches) != 1:
        raise ValueError(f"ollama_model_match_count:{model}:{len(matches)}")
    if str(matches[0].get("digest", "")) != expected_digest:
        raise ValueError(f"ollama_model_digest_mismatch:{matches[0].get('digest')}!={expected_digest}")
    return matches[0]


def _git(command: list[str]) -> str:
    completed = subprocess.run(
        ["git", *command], cwd=ROOT, check=False, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _seeds(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"seeds_must_be_nonempty_unique:{values}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate frozen one-call semantic-prior proposals with an Ollama model."
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domain", default="pfsp")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--feature-budget", type=int, default=8)
    parser.add_argument("--interaction-budget", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--require-clean-git", action="store_true")
    args = parser.parse_args()

    seeds = _seeds(args.seeds)
    status = _git(["status", "--porcelain", "--untracked-files=normal"])
    if args.require_clean_git and status:
        raise RuntimeError(f"generator_requires_clean_git:{status}")
    commit = _git(["rev-parse", "HEAD"])
    model_entry = _model_entry(
        args.endpoint, args.model, args.model_digest, timeout=args.timeout_seconds
    )
    prompt = prompt_for_semantic_prior(
        args.domain,
        feature_budget=args.feature_budget,
        interaction_budget=args.interaction_budget,
    )
    prompt_sha = _sha256_text(prompt)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output_dir_already_exists:{output_dir}")
    ensure_dir(output_dir / "raw")
    ensure_dir(output_dir / "responses")
    records: list[dict[str, Any]] = []

    for seed in seeds:
        request_payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "seed": seed,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "num_predict": args.max_new_tokens,
            },
        }
        started = time.perf_counter()
        response = _post_json(args.endpoint, request_payload, timeout=args.timeout_seconds)
        elapsed = time.perf_counter() - started
        message = response.get("message", {})
        if not isinstance(message, dict):
            raise ValueError("ollama_message_not_object")
        raw_text = str(message.get("content", "")).strip()
        raw_path = output_dir / "raw" / f"{args.domain}_seed{seed}_raw.txt"
        response_path = output_dir / "responses" / f"{args.domain}_seed{seed}.json"
        proposal_path = output_dir / f"{args.domain}_seed{seed}.json"
        raw_path.write_text(raw_text + "\n", encoding="utf-8")
        write_json_atomic(response_path, response)
        record: dict[str, Any] = {
            "seed": seed,
            "call_count": 1,
            "wall_clock_seconds": round(elapsed, 6),
            "raw_path": str(raw_path),
            "raw_sha256": _sha256_text(raw_text),
            "response_path": str(response_path),
            "response_sha256": _sha256_json(response),
            "proposal_path": str(proposal_path),
            "status": "invalid",
            "error": "",
            "input_token_count": int(response.get("prompt_eval_count", 0)),
            "output_token_count": int(response.get("eval_count", 0)),
        }
        try:
            prior = normalize_prior_budget(
                parse_prior_proposal(
                    raw_text, domain=args.domain, source="llm_prior"
                ),
                feature_budget=args.feature_budget,
                interaction_budget=args.interaction_budget,
            )
            if not prior.features:
                raise ValueError("proposal_has_no_valid_features")
            write_json_atomic(proposal_path, prior.to_dict())
            record.update(
                {
                    "status": "valid",
                    "proposal_sha256": _sha256_json(prior.to_dict()),
                    "feature_count": len(prior.features),
                    "interaction_count": len(prior.interactions),
                    "macro_count": len(prior.macro_expressions),
                }
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}:{exc}"
        records.append(record)
        print(
            f"seed={seed} status={record['status']} tokens={record['output_token_count']} "
            f"seconds={record['wall_clock_seconds']}",
            flush=True,
        )

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "queue_status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator_git_commit": commit,
        "generator_git_status": status,
        "model_name": args.model,
        "model_digest": args.model_digest,
        "model_registry_entry": model_entry,
        "model_registry_entry_sha256": _sha256_json(model_entry),
        "endpoint": args.endpoint,
        "domain": args.domain,
        "seeds": list(seeds),
        "feature_budget": args.feature_budget,
        "interaction_budget": args.interaction_budget,
        "prompt": prompt,
        "prompt_sha256": prompt_sha,
        "generation_parameters": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "format": "json",
            "think": False,
            "stream": False,
        },
        "expected_call_count": len(seeds),
        "finished_call_count": len(records),
        "valid_call_count": sum(row["status"] == "valid" for row in records),
        "invalid_call_count": sum(row["status"] != "valid" for row in records),
        "retry_count": 0,
        "records": records,
    }
    write_json_atomic(output_dir / "proposal_manifest.json", manifest)
    print(f"wrote={output_dir / 'proposal_manifest.json'}")
    if manifest["invalid_call_count"]:
        raise RuntimeError(f"invalid_proposal_count:{manifest['invalid_call_count']}")


if __name__ == "__main__":
    main()
