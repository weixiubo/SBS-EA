from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def _run_command(command: list[str], *, env: dict[str, str], cwd: Path) -> dict[str, object]:
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd),
    )
    payload = {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.returncode != 0:
        raise RuntimeError(json.dumps(payload, ensure_ascii=False))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--backend", choices=("mock", "remote"), default="remote")
    parser.add_argument("--kernel", default="llm_mutation_action_v1")
    parser.add_argument("--phase", choices=("official", "calibration"), default="official")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--generator-model-path", default=None)
    parser.add_argument("--run-tag", default="bpp_bpplib_phase0")
    parser.add_argument("--dataset-kind", default="bpplib")
    parser.add_argument("--gpu-index", type=int, default=1)
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    env["POC_BPP_DATASET_KIND"] = args.dataset_kind
    env["POC_SERIAL_SINGLE_GPU"] = "1"
    env["POC_GENERATOR_GPU_INDEX"] = str(args.gpu_index)
    env["POC_JUDGE_A_GPU_INDEX"] = str(args.gpu_index)
    env["POC_JUDGE_B_GPU_INDEX"] = str(args.gpu_index)
    if args.generator_model_path:
        env["POC_GENERATOR_MODEL_PATH"] = args.generator_model_path

    results: list[dict[str, object]] = []
    for seed_text in (item.strip() for item in args.seeds.split(",") if item.strip()):
        seed = int(seed_text)
        command = [
            "python",
            "-m",
            "pairwise_bpp_poc.cli",
            "kernel-stage-b",
            "--task",
            "bpp",
            "--backend",
            args.backend,
            "--phase",
            args.phase,
            "--seed",
            str(seed),
            "--kernel",
            args.kernel,
            "--run-tag",
            f"{args.run_tag}_seed{seed}",
        ]
        if args.generator_model_path:
            command.extend(["--generator-model-path", args.generator_model_path])
        payload = _run_command(command, env=env, cwd=root_dir)
        try:
            payload["json"] = json.loads(str(payload["stdout"]))
        except Exception:
            payload["json"] = None
        results.append(payload)
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
