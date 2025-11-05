#!/usr/bin/env python3
import argparse
import importlib
from pathlib import Path

import yaml
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from trainer.eval_metrics import eval_tool_calls


def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def download(repo_id: str, root: str) -> str:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    local = root / repo_id.replace("/", "__")
    return snapshot_download(
        repo_id=repo_id,
        local_dir=str(local),
        local_dir_use_symlinks=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    base_id = cfg["model"]["base_model_name"]               # Qwen/Qwen3-4B-Instruct-2507
    adapter_id = cfg["train"]["hub_model_id"]               # Salesteq/...-CarSales
    tok_id = cfg["data"].get("tokenizer", base_id)

    tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=False, trust_remote_code=True)

    local_root = "/workspace/models"
    base_path = download(base_id, local_root)
    adapter_path = download(adapter_id, local_root)

    data_mod = importlib.import_module(cfg["data"]["load_module"])
    (
        _train_ds,
        _eval_ds,
        tool_eval_examples,
        global_tools,
        max_ctx_from_data,
    ) = data_mod.build_datasets(cfg)

    eval_cfg = cfg.get("eval", {})
    max_new_tokens = int(eval_cfg.get("max_new_tokens_eval", 256))
    temperature = float(eval_cfg.get("temperature", 0.0))
    max_model_len = args.max_model_len or int(cfg["data"].get("max_context_length", max_ctx_from_data))

    metrics = eval_tool_calls(
        model_path=base_path,
        tokenizer=tokenizer,
        examples=tool_eval_examples,
        global_tools=global_tools,
        max_model_len=max_model_len,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        adapter_path=adapter_path,
    )

    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
