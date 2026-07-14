"""Offline DPO trainer for SPAD answer distillation.

This is a VERL recipe entrypoint for Stage 3 SPAD-RAG. It intentionally uses
data parallel DDP and tensor_parallel_size=1 for 1.7B actors; no model/parameter
parallelism is introduced for the small model.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer


def _init_distributed() -> tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = "gloo"
    device = torch.device("cpu")
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            device = torch.device(f"npu:{local_rank}")
            torch.npu.set_device(device)
            backend = "hccl"
    except Exception:
        pass
    if device.type == "cpu" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        backend = "nccl"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size, device


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class SpadDpoDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: str | Path, *, max_samples: int) -> None:
        rows: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                messages = item.get("messages_before_final_answer")
                chosen = str(item.get("chosen") or "").strip()
                rejected = str(item.get("rejected") or "").strip()
                if not isinstance(messages, list) or not messages or not chosen or not rejected:
                    raise ValueError(f"invalid SPAD DPO row at {path}:{line_no}")
                if "<status>" in chosen:
                    raise ValueError(f"chosen answer contains <status> at {path}:{line_no}")
                rows.append(item)
                if max_samples > 0 and len(rows) >= max_samples:
                    break
        if not rows:
            raise ValueError(f"SPAD DPO dataset is empty: {path}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        row["_row_index"] = index
        return row


def _format_prompt(tokenizer: Any, messages: list[dict[str, str]], apply_kwargs: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **apply_kwargs)


def _tokenize_pair(
    *,
    tokenizer: Any,
    messages: list[dict[str, str]],
    response: str,
    apply_kwargs: dict[str, Any],
    max_length: int,
) -> dict[str, torch.Tensor]:
    prompt_text = _format_prompt(tokenizer, messages, apply_kwargs)
    response_text = response + (tokenizer.eos_token or "")
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    response_ids = tokenizer(response_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    input_ids = torch.cat([prompt_ids, response_ids], dim=0)
    labels = input_ids.clone()
    labels[: prompt_ids.numel()] = -100
    if input_ids.numel() > max_length:
        overflow = input_ids.numel() - max_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
        if labels.ne(-100).sum().item() <= 0:
            raise ValueError("truncation removed all response tokens")
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _pad(items: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(int(item["input_ids"].numel()) for item in items)
    out: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in items:
        pad_len = max_len - int(item["input_ids"].numel())
        out["input_ids"].append(F.pad(item["input_ids"], (0, pad_len), value=pad_id))
        out["attention_mask"].append(F.pad(item["attention_mask"], (0, pad_len), value=0))
        out["labels"].append(F.pad(item["labels"], (0, pad_len), value=-100))
    return {key: torch.stack(value, dim=0) for key, value in out.items()}


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _empty_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()


def _sequence_logps(model: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[:, :-1, :]
    shifted_labels = batch["labels"][:, 1:]
    valid = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    token_logps = logits.log_softmax(dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * valid
    lengths = valid.sum(dim=-1).clamp_min(1)
    return token_logps.sum(dim=-1), -(token_logps.sum() / lengths.sum().clamp_min(1))


def _collate_builder(tokenizer: Any, apply_kwargs: dict[str, Any], max_length: int) -> Any:
    pad_id = int(tokenizer.pad_token_id)

    def collate(rows: list[dict[str, Any]]) -> dict[str, dict[str, torch.Tensor]]:
        chosen = [
            _tokenize_pair(
                tokenizer=tokenizer,
                messages=row["messages_before_final_answer"],
                response=row["chosen"],
                apply_kwargs=apply_kwargs,
                max_length=max_length,
            )
            for row in rows
        ]
        rejected = [
            _tokenize_pair(
                tokenizer=tokenizer,
                messages=row["messages_before_final_answer"],
                response=row["rejected"],
                apply_kwargs=apply_kwargs,
                max_length=max_length,
            )
            for row in rows
        ]
        return {
            "indices": torch.tensor([int(row["_row_index"]) for row in rows], dtype=torch.long),
            "chosen": _pad(chosen, pad_id),
            "rejected": _pad(rejected, pad_id),
        }

    return collate


def _precompute_reference_logps(
    *,
    model_path: str,
    dataset: SpadDpoDataset,
    tokenizer: Any,
    apply_kwargs: dict[str, Any],
    max_length: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, tuple[float, float]]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_builder(tokenizer, apply_kwargs, max_length),
    )
    ref = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True).to(device)
    ref.eval()
    out: dict[int, tuple[float, float]] = {}
    with torch.inference_mode():
        for batch in loader:
            indices = batch["indices"].tolist()
            chosen = _move(batch["chosen"], device)
            rejected = _move(batch["rejected"], device)
            ref_chosen, _ = _sequence_logps(ref, chosen)
            ref_rejected, _ = _sequence_logps(ref, rejected)
            for row_index, chosen_logp, rejected_logp in zip(indices, ref_chosen.detach().cpu().tolist(), ref_rejected.detach().cpu().tolist(), strict=True):
                out[int(row_index)] = (float(chosen_logp), float(rejected_logp))
    del ref
    _empty_device_cache(device)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size-per-gpu", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--total-training-steps", type=int, default=-1)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--pairwise-loss-weight", type=float, default=1.0)
    parser.add_argument("--chosen-sft-loss-weight", type=float, default=0.2)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    rank, local_rank, world_size, device = _init_distributed()
    dtype = torch.bfloat16 if device.type != "cpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = SpadDpoDataset(args.dataset_jsonl, max_samples=args.max_samples)
    apply_kwargs = {"enable_thinking": bool(args.enable_thinking)}
    reference_logps = _precompute_reference_logps(
        model_path=args.model_path,
        dataset=dataset,
        tokenizer=tokenizer,
        apply_kwargs=apply_kwargs,
        max_length=args.max_length,
        batch_size=max(1, args.micro_batch_size_per_gpu),
        device=device,
        dtype=dtype,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.micro_batch_size_per_gpu),
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=0,
        collate_fn=_collate_builder(
            tokenizer,
            apply_kwargs,
            args.max_length,
        ),
    )
    policy = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True).to(device)
    if hasattr(policy, "config"):
        policy.config.use_cache = False
    if hasattr(policy, "gradient_checkpointing_enable"):
        try:
            policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            policy.gradient_checkpointing_enable()
    if world_size > 1:
        ddp_kwargs: dict[str, Any] = {}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [local_rank]
        policy = DDP(policy, **ddp_kwargs)
    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    global_micro_batch = max(1, args.micro_batch_size_per_gpu * world_size)
    grad_accum = max(1, math.ceil(args.train_batch_size / global_micro_batch))
    steps_per_epoch = max(1, math.ceil(len(loader) / grad_accum))
    max_steps = args.total_training_steps if args.total_training_steps and args.total_training_steps > 0 else steps_per_epoch * max(1, args.total_epochs)
    metrics: list[dict[str, Any]] = []
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(max(1, args.total_epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)
        accum = 0
        loss_sum = 0.0
        dpo_sum = 0.0
        sft_sum = 0.0
        policy_margin_sum = 0.0
        ref_margin_sum = 0.0
        seen = 0
        for batch in loader:
            indices = batch["indices"].tolist()
            chosen = _move(batch["chosen"], device)
            rejected = _move(batch["rejected"], device)
            policy_chosen, chosen_nll = _sequence_logps(policy, chosen)
            policy_rejected, _ = _sequence_logps(policy, rejected)
            ref_values = [reference_logps[int(index)] for index in indices]
            ref_chosen = torch.tensor([item[0] for item in ref_values], device=device, dtype=policy_chosen.dtype)
            ref_rejected = torch.tensor([item[1] for item in ref_values], device=device, dtype=policy_rejected.dtype)
            logits = args.beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
            dpo_loss = -F.logsigmoid(logits).mean()
            loss = args.pairwise_loss_weight * dpo_loss + args.chosen_sft_loss_weight * chosen_nll
            (loss / grad_accum).backward()
            batch_seen = int(chosen["input_ids"].shape[0])
            accum += 1
            seen += batch_seen
            loss_sum += float(loss.detach().cpu()) * batch_seen
            dpo_sum += float(dpo_loss.detach().cpu()) * batch_seen
            sft_sum += float(chosen_nll.detach().cpu()) * batch_seen
            policy_margin_sum += float((policy_chosen - policy_rejected).mean().detach().cpu()) * batch_seen
            ref_margin_sum += float((ref_chosen - ref_rejected).mean().detach().cpu()) * batch_seen
            if accum >= grad_accum:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                denom = max(1, seen)
                local_metrics = torch.tensor(
                    [
                        loss_sum,
                        dpo_sum,
                        sft_sum,
                        policy_margin_sum,
                        ref_margin_sum,
                        float(seen),
                    ],
                    device=device,
                    dtype=torch.float32,
                )
                if world_size > 1:
                    dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)
                if rank == 0:
                    total_seen = max(1.0, float(local_metrics[5].item()))
                    metrics.append(
                        {
                            "step": step,
                            "epoch": epoch,
                            "loss": float(local_metrics[0].item()) / total_seen,
                            "dpo_loss": float(local_metrics[1].item()) / total_seen,
                            "chosen_nll": float(local_metrics[2].item()) / total_seen,
                            "policy_margin": float(local_metrics[3].item()) / total_seen,
                            "ref_margin": float(local_metrics[4].item()) / total_seen,
                            "world_size": world_size,
                            "tensor_parallel_size": 1,
                            "micro_batch_size_per_gpu": args.micro_batch_size_per_gpu,
                            "grad_accum": grad_accum,
                        }
                    )
                    print(f"SPAD VERL DPO step={step}/{max_steps} loss={metrics[-1]['loss']:.6f}", flush=True)
                accum = 0
                loss_sum = dpo_sum = sft_sum = policy_margin_sum = ref_margin_sum = 0.0
                seen = 0
                if step >= max_steps:
                    break
        if accum > 0 and step < max_steps:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
        if step >= max_steps:
            break
    _barrier()
    if rank == 0:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model = policy.module if isinstance(policy, DDP) else policy
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "backend": "verl_spad_offline_dpo",
                    "elapsed_s": time.time() - started,
                    "sample_count": len(dataset),
                    "world_size": world_size,
                    "tensor_parallel_size": 1,
                    "metrics": metrics,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    _barrier()
    _cleanup()


if __name__ == "__main__":
    main()
