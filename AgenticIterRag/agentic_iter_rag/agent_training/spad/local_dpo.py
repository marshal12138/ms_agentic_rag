"""Lightweight local DPO trainer for SPAD-RAG Stage 3 ablations."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentic_iter_rag.utils.io import iter_jsonl, write_json


def _select_device(resource_plan: dict[str, Any]) -> torch.device:
    phases = resource_plan.get("phases") if isinstance(resource_plan.get("phases"), dict) else {}
    dpo_plan = phases.get("dpo", {}) if isinstance(phases.get("dpo"), dict) else {}
    trainer = dpo_plan.get("trainer", {}) if isinstance(dpo_plan.get("trainer"), dict) else {}
    gpu_ids = trainer.get("gpu_ids") or []
    if gpu_ids:
        os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", ",".join(str(item) for item in gpu_ids))
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(item) for item in gpu_ids))
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu:0")
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _read_dataset(path: str | Path, *, max_samples: int = -1) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    if max_samples > 0:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError(f"SPAD DPO dataset is empty: {path}")
    return rows


def _format_prompt(tokenizer: Any, messages: list[dict[str, str]], apply_kwargs: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        **apply_kwargs,
    )


def _tokenize_pair(
    *,
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    response: str,
    apply_kwargs: dict[str, Any],
    max_length: int,
) -> dict[str, torch.Tensor]:
    prompt_text = _format_prompt(tokenizer, prompt_messages, apply_kwargs)
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
        if (labels != -100).sum().item() <= 0:
            raise ValueError("truncation removed all response tokens")
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _pad_batch(items: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(int(item["input_ids"].numel()) for item in items)
    out: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in items:
        pad_len = max_len - int(item["input_ids"].numel())
        out["input_ids"].append(F.pad(item["input_ids"], (0, pad_len), value=pad_id))
        out["attention_mask"].append(F.pad(item["attention_mask"], (0, pad_len), value=0))
        out["labels"].append(F.pad(item["labels"], (0, pad_len), value=-100))
    return {key: torch.stack(value, dim=0) for key, value in out.items()}


def _sequence_logps(model: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    token_logps = logits.log_softmax(dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * valid
    lengths = valid.sum(dim=-1).clamp_min(1)
    return token_logps.sum(dim=-1), -(token_logps.sum() / lengths.sum().clamp_min(1))


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_local_dpo(
    *,
    model_path: str,
    dataset_jsonl: str,
    output_dir: str | Path,
    log_dir: str | Path | None = None,
    phase_cfg: dict[str, Any],
    resource_plan: dict[str, Any],
) -> dict[str, Any]:
    """Run a small local DPO job and save a Hugging Face checkpoint."""

    started = time.time()
    device = _select_device(resource_plan)
    max_samples = int(phase_cfg.get("max_samples", -1))
    rows = _read_dataset(dataset_jsonl, max_samples=max_samples)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    apply_kwargs = dict(phase_cfg.get("apply_chat_template_kwargs") or {"enable_thinking": False})
    max_length = int(phase_cfg.get("max_length", 4096))
    batch_size = int(phase_cfg.get("train_batch_size", 64))
    micro_batch_size = max(1, int(phase_cfg.get("micro_batch_size_per_gpu", batch_size)))
    configured_total_steps = phase_cfg.get("total_training_steps")
    total_epochs = int(phase_cfg.get("total_epochs", 1))
    if configured_total_steps is None or int(configured_total_steps) <= 0:
        total_steps = max(1, math.ceil(len(rows) / batch_size)) * total_epochs
    else:
        total_steps = int(configured_total_steps)
    beta = float(phase_cfg.get("beta", 0.1))
    pairwise_weight = float(phase_cfg.get("pairwise_loss_weight", 1.0))
    sft_weight = float(phase_cfg.get("chosen_sft_loss_weight", 0.2))
    lr = float(phase_cfg.get("learning_rate", 1.0e-6))

    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.type != "cpu" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    ref = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.type != "cpu" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    ref.eval()
    for param in ref.parameters():
        param.requires_grad_(False)
    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    metrics: list[dict[str, Any]] = []
    for step in range(1, total_steps + 1):
        batch_rows = [rows[((step - 1) * batch_size + idx) % len(rows)] for idx in range(batch_size)]
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        dpo_loss_sum = 0.0
        chosen_nll_sum = 0.0
        policy_margin_sum = 0.0
        ref_margin_sum = 0.0
        seen = 0
        for start in range(0, len(batch_rows), micro_batch_size):
            micro_rows = batch_rows[start : start + micro_batch_size]
            scale = len(micro_rows) / max(1, len(batch_rows))
            chosen_items = [
                _tokenize_pair(
                    tokenizer=tokenizer,
                    prompt_messages=row["messages_before_final_answer"],
                    response=row["chosen"],
                    apply_kwargs=apply_kwargs,
                    max_length=max_length,
                )
                for row in micro_rows
            ]
            rejected_items = [
                _tokenize_pair(
                    tokenizer=tokenizer,
                    prompt_messages=row["messages_before_final_answer"],
                    response=row["rejected"],
                    apply_kwargs=apply_kwargs,
                    max_length=max_length,
                )
                for row in micro_rows
            ]
            chosen_batch = _move_batch(_pad_batch(chosen_items, tokenizer.pad_token_id), device)
            rejected_batch = _move_batch(_pad_batch(rejected_items, tokenizer.pad_token_id), device)

            policy_chosen, chosen_nll = _sequence_logps(policy, chosen_batch)
            policy_rejected, _ = _sequence_logps(policy, rejected_batch)
            with torch.no_grad():
                ref_chosen, _ = _sequence_logps(ref, chosen_batch)
                ref_rejected, _ = _sequence_logps(ref, rejected_batch)
            logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
            dpo_loss = -F.logsigmoid(logits).mean()
            loss = pairwise_weight * dpo_loss + sft_weight * chosen_nll
            (loss * scale).backward()

            weight = len(micro_rows)
            seen += weight
            loss_sum += float(loss.detach().cpu()) * weight
            dpo_loss_sum += float(dpo_loss.detach().cpu()) * weight
            chosen_nll_sum += float(chosen_nll.detach().cpu()) * weight
            policy_margin_sum += float((policy_chosen - policy_rejected).mean().detach().cpu()) * weight
            ref_margin_sum += float((ref_chosen - ref_rejected).mean().detach().cpu()) * weight

        torch.nn.utils.clip_grad_norm_(policy.parameters(), float(phase_cfg.get("clip_grad_norm", 1.0)))
        optimizer.step()
        denom = max(1, seen)
        metrics.append(
            {
                "step": step,
                "loss": loss_sum / denom,
                "dpo_loss": dpo_loss_sum / denom,
                "chosen_nll": chosen_nll_sum / denom,
                "policy_margin": policy_margin_sum / denom,
                "ref_margin": ref_margin_sum / denom,
                "micro_batch_size_per_gpu": micro_batch_size,
            }
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out)
    tokenizer.save_pretrained(out)
    metrics_dir = Path(log_dir) if log_dir is not None else out
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "spad_local_dpo_metrics.json"
    write_json(metrics_path, {"metrics": metrics, "elapsed_s": time.time() - started})
    return {
        "checkpoint": str(out),
        "metrics": metrics,
        "metrics_path": str(metrics_path),
        "elapsed_s": time.time() - started,
        "device": str(device),
        "sample_count": len(rows),
    }
