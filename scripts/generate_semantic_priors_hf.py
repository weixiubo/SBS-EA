from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
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


PROTOCOL_VERSION = "semantic_prior_hf_one_call_v1"
ARTIFACT_NAMES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(text)


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


def _artifact_manifest(model_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name in ARTIFACT_NAMES:
        path = model_path / name
        if not path.is_file():
            raise FileNotFoundError(f"missing_model_artifact:{path}")
        rows.append(
            {
                "name": name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "model_path": str(model_path),
        "files": rows,
        "manifest_sha256": _sha256_json(rows),
    }


def _load_model(model_path: Path):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()
    return model, processor, torch


def _generate(
    *,
    model: Any,
    processor: Any,
    torch: Any,
    prompt: str,
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[rendered], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    elapsed = time.perf_counter() - started
    input_tokens = int(inputs["input_ids"].shape[1])
    generated = output_ids[:, input_tokens:]
    raw_text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    metrics = {
        "input_token_count": input_tokens,
        "output_token_count": int(generated.shape[1]),
        "wall_clock_seconds": round(elapsed, 6),
        "seed": seed,
    }
    return raw_text, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate frozen one-call semantic priors with a local Hugging Face model."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--expected-artifact-manifest-sha256", default="")
    parser.add_argument(
        "--prompt-source-manifest",
        type=Path,
        default=None,
        help="Optional archived proposal manifest supplying the exact frozen prompt.",
    )
    parser.add_argument("--expected-prompt-sha256", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", default="pfsp")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--feature-budget", type=int, default=8)
    parser.add_argument("--interaction-budget", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--allow-macros", choices=("yes", "no"), default="yes")
    parser.add_argument("--require-clean-git", action="store_true")
    args = parser.parse_args()

    seeds = _seeds(args.seeds)
    status = _git(["status", "--porcelain", "--untracked-files=normal"])
    if args.require_clean_git and status:
        raise RuntimeError(f"generator_requires_clean_git:{status}")
    commit = _git(["rev-parse", "HEAD"])
    model_path = args.model_path.resolve()
    artifact_manifest = _artifact_manifest(model_path)
    if (
        args.expected_artifact_manifest_sha256
        and artifact_manifest["manifest_sha256"]
        != args.expected_artifact_manifest_sha256
    ):
        raise RuntimeError(
            "model_artifact_manifest_mismatch:"
            f"{artifact_manifest['manifest_sha256']}!="
            f"{args.expected_artifact_manifest_sha256}"
        )

    prompt_source_manifest = ""
    prompt_source_manifest_sha256 = ""
    if args.prompt_source_manifest is not None:
        prompt_source_path = args.prompt_source_manifest.resolve()
        prompt_source_manifest_sha256 = _sha256_file(prompt_source_path)
        prompt_source = json.loads(
            prompt_source_path.read_text(encoding="utf-8")
        )
        prompt = str(prompt_source.get("prompt", ""))
        if not prompt:
            raise RuntimeError("prompt_source_manifest_has_no_prompt")
        if prompt_source.get("prompt_sha256") != _sha256_text(prompt):
            raise RuntimeError("prompt_source_manifest_prompt_hash_mismatch")
        prompt_source_manifest = str(prompt_source_path)
    else:
        prompt = prompt_for_semantic_prior(
            args.domain,
            feature_budget=args.feature_budget,
            interaction_budget=args.interaction_budget,
            allow_macros=args.allow_macros == "yes",
        )
    prompt_sha256 = _sha256_text(prompt)
    if (
        args.expected_prompt_sha256
        and prompt_sha256 != args.expected_prompt_sha256
    ):
        raise RuntimeError(
            f"prompt_sha256_mismatch:{prompt_sha256}!="
            f"{args.expected_prompt_sha256}"
        )
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output_dir_already_exists:{output_dir}")
    ensure_dir(output_dir / "raw")
    model, processor, torch = _load_model(model_path)
    records: list[dict[str, Any]] = []

    for seed in seeds:
        raw_text, metrics = _generate(
            model=model,
            processor=processor,
            torch=torch,
            prompt=prompt,
            seed=seed,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        raw_path = output_dir / "raw" / f"{args.domain}_seed{seed}_raw.txt"
        proposal_path = output_dir / f"{args.domain}_seed{seed}.json"
        raw_path.write_text(raw_text + "\n", encoding="utf-8")
        record: dict[str, Any] = {
            "seed": seed,
            "call_count": 1,
            "raw_path": str(raw_path),
            "raw_sha256": _sha256_text(raw_text),
            "proposal_path": str(proposal_path),
            "status": "invalid",
            "error": "",
            **metrics,
        }
        try:
            prior = normalize_prior_budget(
                parse_prior_proposal(
                    raw_text, domain=args.domain, source="llm_prior"
                ),
                feature_budget=args.feature_budget,
                interaction_budget=args.interaction_budget,
            )
            if args.allow_macros == "no" and prior.macro_expressions:
                raise ValueError("proposal_contains_disallowed_macros")
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
            f"seed={seed} status={record['status']} "
            f"tokens={record['output_token_count']} "
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
        "model_name": args.model_name,
        "model_artifacts": artifact_manifest,
        "domain": args.domain,
        "seeds": list(seeds),
        "feature_budget": args.feature_budget,
        "allow_macros": args.allow_macros == "yes",
        "interaction_budget": args.interaction_budget,
        "prompt": prompt,
        "prompt_sha256": prompt_sha256,
        "prompt_source_manifest": prompt_source_manifest,
        "prompt_source_manifest_sha256": prompt_source_manifest_sha256,
        "generation_parameters": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "text_only": True,
        },
        "expected_call_count": len(seeds),
        "finished_call_count": len(records),
        "valid_call_count": sum(row["status"] == "valid" for row in records),
        "invalid_call_count": sum(row["status"] != "valid" for row in records),
        "retry_count": 0,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
        },
        "records": records,
    }
    write_json_atomic(output_dir / "proposal_manifest.json", manifest)
    print(f"artifact_manifest_sha256={artifact_manifest['manifest_sha256']}")
    print(f"wrote={output_dir / 'proposal_manifest.json'}")
    if manifest["invalid_call_count"]:
        raise RuntimeError(f"invalid_proposal_count:{manifest['invalid_call_count']}")


if __name__ == "__main__":
    main()
