#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=0.34.0",
#   "bitsandbytes>=0.43.0",
#   "datasets>=2.20.0",
#   "peft>=0.12.0",
#   "torch>=2.3.0",
#   "transformers>=4.45.0",
# ]
# ///
from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-out", required=True)
    parser.add_argument("--merged-out", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:  # pragma: no cover - exercised on the remote box.
        raise RuntimeError(
            "local_editor_train.py is missing training dependencies. "
            "Run with `uv run` or install torch/transformers/datasets/peft/bitsandbytes first."
        ) from exc

    train_rows = _load_rows(Path(args.train_file))
    dev_rows = _load_rows(Path(args.dev_file))
    if not train_rows:
        raise RuntimeError("training set is empty")
    if not dev_rows:
        raise RuntimeError("dev set is empty")

    adapter_out = Path(args.adapter_out)
    merged_out = Path(args.merged_out)
    manifest_out = Path(args.manifest_out)
    adapter_out.mkdir(parents=True, exist_ok=True)
    merged_out.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    )
    model = get_peft_model(model, peft_config)

    def _render_chat(prompt: str, target_json: dict) -> tuple[str, str]:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        target = json.dumps(target_json, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return prefix, target

    def _encode_row(row: dict) -> dict:
        prefix, target = _render_chat(row["prompt"], row["target_action_json"])
        full_text = prefix + target
        full_ids = tokenizer(
            full_text,
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
        )["input_ids"]
        prefix_ids = tokenizer(
            prefix,
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
        )["input_ids"]
        labels = list(full_ids)
        prefix_len = min(len(prefix_ids), len(labels))
        for index in range(prefix_len):
            labels[index] = -100
        attention_mask = [1] * len(full_ids)
        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_dataset = Dataset.from_list([_encode_row(row) for row in train_rows])
    dev_dataset = Dataset.from_list([_encode_row(row) for row in dev_rows])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_kwargs = {
        "output_dir": str(adapter_out / "checkpoints"),
        "do_train": True,
        "do_eval": True,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "bf16": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": [],
        "remove_unused_columns": False,
        "seed": args.seed,
    }
    supported_args = set(inspect.signature(TrainingArguments.__init__).parameters)
    train_args = TrainingArguments(
        **{key: value for key, value in training_kwargs.items() if key in supported_args}
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.model.save_pretrained(adapter_out)
    tokenizer.save_pretrained(adapter_out)

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merge_base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    merged = PeftModel.from_pretrained(merge_base, str(adapter_out))
    merged = merged.merge_and_unload()
    merged.save_pretrained(merged_out, safe_serialization=True)
    tokenizer.save_pretrained(merged_out)

    manifest = {
        "base_model": args.base_model,
        "train_file": args.train_file,
        "dev_file": args.dev_file,
        "adapter_out": str(adapter_out),
        "merged_out": str(merged_out),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "epochs": args.epochs,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "train_loss": float(train_result.training_loss),
        "eval_loss": float(eval_metrics.get("eval_loss", math.nan)),
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
