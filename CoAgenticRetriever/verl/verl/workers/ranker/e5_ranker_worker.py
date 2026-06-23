"""E5 shared-encoder ranker worker."""

from __future__ import annotations

import os
import threading
from typing import Any

import ray
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from verl import DataProto


def _cfg_get(config: Any, path: str, default=None):
    cur = config
    for part in path.split("."):
        if hasattr(cur, "get"):
            cur = cur.get(part, default)
        elif isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            return default
        if cur is default:
            return default
    return cur


def _cfg_require(config: Any, path: str):
    sentinel = object()
    value = _cfg_get(config, path, sentinel)
    if value is sentinel or value is None or value == "":
        raise KeyError(f"missing required ranker config: {path}")
    return value


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class LocalE5RankerWorker:
    """Trainable ranker with one shared E5 encoder.

    The recall retriever remains a separate frozen service. This worker only
    rescors recall top-k documents and trains the shared encoder with
    contrastive loss.
    """

    def __init__(self, config):
        self.config = config
        requested_device = str(_cfg_require(config, "ranker.device"))
        if not requested_device.startswith("cuda"):
            raise RuntimeError(
                "CoAgenticRetriever ranker requires CUDA. "
                f"Got ranker.device={requested_device!r}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CoAgenticRetriever ranker requires CUDA, but torch.cuda.is_available() is False. "
                "Run the script in a GPU-enabled environment."
            )
        self.device = torch.device(requested_device)
        self.model_path = _cfg_require(config, "ranker.model_path")
        self.rank_encoder_path = _cfg_require(config, "ranker.encoder_path")
        self.use_e5_prefix = "e5" in str(self.model_path).lower() or "e5" in str(self.rank_encoder_path).lower()
        self.temperature = float(
            _cfg_require(config, "ranker_training.loss.temperature")
        )
        self.max_grad_norm = float(
            _cfg_require(config, "ranker_training.max_grad_norm")
        )
        self.gradient_accumulation_steps = max(
            1,
            int(
                _cfg_require(config, "ranker_training.gradient_accumulation_steps")
            ),
        )
        self.rank_top_k = int(_cfg_require(config, "ranker.top_k"))
        self.tokenizer = None
        self.encoder = None
        self.optimizer = None
        self.scheduler = None
        self.step = 0

    def init_model(self, *, train_mode: bool = True):
        trust_remote_code = bool(_cfg_require(self.config, "ranker.trust_remote_code"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=trust_remote_code)
        self.encoder = AutoModel.from_pretrained(self.rank_encoder_path, trust_remote_code=trust_remote_code)
        self.encoder.to(self.device)
        self.encoder.train(mode=train_mode)

        trainable = list(self.encoder.parameters())
        lr = float(
            _cfg_require(self.config, "ranker_training.optim.lr")
        )
        weight_decay = float(
            _cfg_require(self.config, "ranker_training.optim.weight_decay")
        )
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
        total_steps = int(
            _cfg_require(self.config, "ranker_training.optim.total_steps")
        )
        warmup_steps = int(
            _cfg_require(self.config, "ranker_training.optim.warmup_steps")
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(total_steps, 1),
        )
        print(
            "[ranker-worker] initialized shared_encoder=true "
            f"path={self.model_path} rank_encoder={self.rank_encoder_path} "
            f"device={self.device} top_k={self.rank_top_k} "
            f"gradient_accumulation_steps={self.gradient_accumulation_steps}",
            flush=True,
        )

    def _format_query(self, text: str) -> str:
        return f"query: {text}" if self.use_e5_prefix else text

    def _format_doc(self, text: str) -> str:
        return f"passage: {text}" if self.use_e5_prefix else text

    @torch.no_grad()
    def encode_query(self, texts: list[str], max_length: int) -> torch.Tensor:
        if self.encoder is None:
            self.init_model()
        texts = [self._format_query(text) for text in texts]
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        outputs = self.encoder(**tokens)
        emb = mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
        return F.normalize(emb, dim=-1)

    encode_texts = encode_query

    @torch.no_grad()
    def encode_docs(self, texts: list[str], max_length: int) -> torch.Tensor:
        if self.encoder is None:
            self.init_model()
        texts = [self._format_doc(text) for text in texts]
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        outputs = self.encoder(**tokens)
        emb = mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def rank_topk(
        self,
        query: str,
        docs: list[dict[str, Any]],
        top_k: int,
        max_query_length: int,
        max_doc_length: int,
    ) -> list[dict[str, Any]]:
        if self.encoder is None:
            self.init_model()
        if top_k is None:
            raise ValueError("top_k must be explicitly provided for ranker.rank_topk")
        top_k = int(top_k)
        if not docs:
            return []
        doc_texts = [
            (str(doc.get("title") or "") + "\n" if doc.get("title") else "")
            + str(doc.get("contents") or doc.get("text") or doc.get("passage") or "")
            for doc in docs
        ]
        query_emb = self.encode_query([query], max_length=max_query_length)
        doc_emb = self.encode_docs(doc_texts, max_length=max_doc_length)
        scores = torch.matmul(query_emb, doc_emb.T).squeeze(0)
        top_scores, top_indices = torch.topk(scores, k=min(top_k, len(docs)))
        ranked = []
        for rank_position, (score, idx) in enumerate(zip(top_scores.tolist(), top_indices.tolist()), start=1):
            doc = dict(docs[idx])
            doc["recall_rank"] = int(doc.get("recall_rank") or doc.get("rank") or idx + 1)
            doc["recall_score"] = doc.get("recall_score", doc.get("retriever_score", doc.get("score")))
            doc["rank_score"] = float(score)
            doc["rank_rank"] = rank_position
            ranked.append(doc)
        return ranked

    def update_ranker_contrastive(self, data: DataProto) -> DataProto:
        if self.encoder is None:
            self.init_model()

        batch = data.batch
        query_input_ids = batch["query_input_ids"].to(self.device)
        query_attention_mask = batch["query_attention_mask"].to(self.device)
        doc_input_ids = batch["doc_input_ids"].to(self.device)
        doc_attention_mask = batch["doc_attention_mask"].to(self.device)
        labels = batch["positive_doc_index"].to(self.device)
        loss_weights = batch.get("loss_weights", None)
        if loss_weights is not None:
            loss_weights = loss_weights.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        bsz, docs_per_query, doc_len = doc_input_ids.shape
        micro_batch_count = min(self.gradient_accumulation_steps, max(1, bsz))
        micro_batch_size = max(1, (bsz + micro_batch_count - 1) // micro_batch_count)
        loss_denominator = (
            loss_weights.sum().clamp_min(1.0)
            if loss_weights is not None
            else torch.tensor(float(max(1, bsz)), device=self.device)
        )

        loss_numerator_total = 0.0
        correct_total = 0.0
        mrr_total = 0.0
        pos_score_total = 0.0
        neg_score_total = 0.0
        neg_score_count = 0
        actual_micro_batches = 0

        for start in range(0, bsz, micro_batch_size):
            end = min(start + micro_batch_size, bsz)
            actual_micro_batches += 1
            micro_query_input_ids = query_input_ids[start:end]
            micro_query_attention_mask = query_attention_mask[start:end]
            micro_doc_input_ids = doc_input_ids[start:end]
            micro_doc_attention_mask = doc_attention_mask[start:end]
            micro_labels = labels[start:end]
            micro_loss_weights = loss_weights[start:end] if loss_weights is not None else None
            micro_bsz = int(end - start)

            query_outputs = self.encoder(input_ids=micro_query_input_ids, attention_mask=micro_query_attention_mask)
            query_emb = mean_pool(query_outputs.last_hidden_state, micro_query_attention_mask)

            flat_doc_input_ids = micro_doc_input_ids.reshape(micro_bsz * docs_per_query, doc_len)
            flat_doc_attention_mask = micro_doc_attention_mask.reshape(micro_bsz * docs_per_query, doc_len)
            doc_outputs = self.encoder(input_ids=flat_doc_input_ids, attention_mask=flat_doc_attention_mask)
            doc_emb = mean_pool(doc_outputs.last_hidden_state, flat_doc_attention_mask)

            doc_emb = doc_emb.reshape(micro_bsz, docs_per_query, -1)
            query_emb = F.normalize(query_emb, dim=-1)
            doc_emb = F.normalize(doc_emb, dim=-1)
            scores = torch.einsum("bh,bkh->bk", query_emb, doc_emb)
            logits = scores / self.temperature

            per_sample_loss = F.cross_entropy(logits, micro_labels, reduction="none")
            if micro_loss_weights is not None:
                loss_numerator = (per_sample_loss * micro_loss_weights).sum()
            else:
                loss_numerator = per_sample_loss.sum()
            (loss_numerator / loss_denominator).backward()

            with torch.no_grad():
                loss_numerator_total += float(loss_numerator.detach().cpu())
                pred = logits.argmax(dim=-1)
                correct_total += float((pred == micro_labels).float().sum().detach().cpu())
                pos_scores = scores.gather(1, micro_labels[:, None]).squeeze(1)
                neg_mask = torch.ones_like(scores, dtype=torch.bool)
                neg_mask.scatter_(1, micro_labels[:, None], False)
                neg_scores = scores[neg_mask]
                pos_score_total += float(pos_scores.sum().detach().cpu())
                neg_score_total += float(neg_scores.sum().detach().cpu())
                neg_score_count += int(neg_scores.numel())

                ranks = torch.argsort(logits, dim=-1, descending=True)
                positive_ranks = (ranks == micro_labels[:, None]).nonzero(as_tuple=False)[:, 1].float() + 1.0
                mrr_total += float((1.0 / positive_ranks).sum().detach().cpu())

        grad_norm = clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1

        loss_value = loss_numerator_total / float(loss_denominator.detach().cpu())
        acc_at_1 = correct_total / max(1, bsz)
        mrr = mrr_total / max(1, bsz)
        pos_score_mean = pos_score_total / max(1, bsz)
        neg_score_mean = neg_score_total / max(1, neg_score_count)
        lr = self.scheduler.get_last_lr()[0] if self.scheduler is not None else self.optimizer.param_groups[0]["lr"]

        metrics = {
            "ranker/loss": float(loss_value),
            "ranker/acc@1": float(acc_at_1),
            "ranker/mrr": float(mrr),
            "ranker/pos_score_mean": float(pos_score_mean),
            "ranker/neg_score_mean": float(neg_score_mean),
            "ranker/score_margin": float(pos_score_mean - neg_score_mean),
            "ranker/num_queries": int(bsz),
            "ranker/num_docs_per_query": int(docs_per_query),
            "ranker/num_neg": int(bsz * (docs_per_query - 1)),
            "ranker/gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "ranker/gradient_accumulation_micro_batches": int(actual_micro_batches),
            "ranker/micro_batch_size": int(micro_batch_size),
            "ranker/lr": float(lr),
            "ranker/grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
            "ranker/local_update_step": int(self.step),
            "ranker/shared_encoder": 1,
        }
        return DataProto(batch=None, meta_info={"metrics": metrics})

    def save_checkpoint(self, path: str):
        if self.encoder is None:
            return
        os.makedirs(path, exist_ok=True)
        self.encoder.save_pretrained(os.path.join(path, "rank_encoder"))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)

    def export_encoder_state_cpu(self) -> dict[str, torch.Tensor]:
        if self.encoder is None:
            self.init_model()
        return {key: value.detach().cpu().clone() for key, value in self.encoder.state_dict().items()}

    def load_encoder_state_cpu(self, state_dict: dict[str, torch.Tensor], *, strict: bool = True) -> None:
        if self.encoder is None:
            self.init_model(train_mode=False)
        state_dict = {key: value.to(self.device, non_blocking=True) for key, value in state_dict.items()}
        self.encoder.load_state_dict(state_dict, strict=strict)
        self.encoder.eval()

LocalRankerContrastiveWorker = LocalE5RankerWorker


class E5RankerContrastiveWorker:
    """Ray actor compatible wrapper."""

    def __init__(self, config):
        self.worker = LocalE5RankerWorker(config)

    def init_model(self):
        return self.worker.init_model()

    def update_ranker_contrastive(self, data: DataProto) -> DataProto:
        return self.worker.update_ranker_contrastive(data)

    def save_checkpoint(self, path: str):
        return self.worker.save_checkpoint(path)


@ray.remote(num_gpus=0)
class SharedE5RankerActor:
    """Single shared dense-ranker inference actor.

    Ray does not assign a GPU resource to this actor because the enclosing
    training process uses manual CUDA_VISIBLE_DEVICES management. The ranker
    device is still controlled by ranker.device in the config.
    """

    def __init__(self, config):
        self.worker = LocalE5RankerWorker(config)
        self.worker.init_model(train_mode=False)
        if self.worker.encoder is not None:
            self.worker.encoder.eval()
        self._lock = threading.RLock()
        self.synced_step = 0
        print(
            "[shared-ranker-actor] initialized "
            f"device={self.worker.device} model={self.worker.model_path} encoder={self.worker.rank_encoder_path}",
            flush=True,
        )

    def rank_topk(
        self,
        query: str,
        docs: list[dict[str, Any]],
        top_k: int,
        max_query_length: int,
        max_doc_length: int,
    ) -> list[dict[str, Any]]:
        with self._lock:
            if self.worker.encoder is not None:
                self.worker.encoder.eval()
            return self.worker.rank_topk(
                query=query,
                docs=docs,
                top_k=top_k,
                max_query_length=max_query_length,
                max_doc_length=max_doc_length,
            )

    def load_encoder_state_cpu(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        step: int = 0,
        strict: bool = True,
    ) -> dict[str, int]:
        with self._lock:
            self.worker.load_encoder_state_cpu(state_dict, strict=strict)
            self.synced_step = int(step)
            return {"ranker/inference_synced_step": self.synced_step}

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "device": str(self.worker.device),
                "model_path": str(self.worker.model_path),
                "rank_encoder_path": str(self.worker.rank_encoder_path),
                "synced_step": int(self.synced_step),
            }
