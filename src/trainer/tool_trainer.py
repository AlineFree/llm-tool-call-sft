from typing import Any, Dict, List, Optional

import torch
from transformers import Trainer

from trainer.eval_metrics import eval_tool_calls, eval_perplexity


class ToolTrainer(Trainer):
    def __init__(
        self,
        *args,
        tokenizer_for_tools,
        tool_eval_examples,
        global_tools,
        max_new_tokens_eval: int,
        temperature_eval: float,
        n_short_eval_examples: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer_for_tools = tokenizer_for_tools
        self.tool_eval_examples = tool_eval_examples or []
        self.global_tools = global_tools or []
        self.max_new_tokens_eval = max_new_tokens_eval
        self.temperature_eval = temperature_eval
        self.n_short_eval_examples = n_short_eval_examples

    def evaluate(
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset

        # everyone runs this
        ppl_stats = eval_perplexity(
            model=self.model,
            dataset=eval_ds,
            batch_size=self.args.per_device_eval_batch_size,
            device=self.model.device
            if isinstance(self.model, torch.nn.Module)
            else torch.device("cpu"),
        )

        examples = self.tool_eval_examples
        if not metric_key_prefix.startswith("full"):
            import random

            examples = random.sample(examples, min(self.n_short_eval_examples, len(examples)))

        tool_stats = eval_tool_calls(
            model=self.model,
            tokenizer=self.tokenizer_for_tools,
            examples=examples,
            global_tools=self.global_tools,
            device=self.model.device,
            max_new_tokens=self.max_new_tokens_eval,
            temperature=self.temperature_eval,
        )

        if self.is_world_process_zero():
            metrics[f"{metric_key_prefix}_ppl"] = ppl_stats["perplexity"]
            metrics[f"{metric_key_prefix}_loss_masked"] = ppl_stats["loss"]
            metrics[f"{metric_key_prefix}_num_tokens"] = ppl_stats["num_tokens"]
            for k, v in tool_stats.items():
                metrics[f"{metric_key_prefix}_{k}"] = v
            self.log(metrics)
            self.save_metrics(split=metric_key_prefix, metrics=metrics)

        # let everyone reach here
        if self.args.local_rank != -1:
            torch.distributed.barrier()

        return metrics


__all__ = [
    "ToolTrainer",
]