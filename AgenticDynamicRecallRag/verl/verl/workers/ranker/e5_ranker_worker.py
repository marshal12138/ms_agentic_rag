"""E5 shared-encoder ranker worker."""

from __future__ import annotations

import os
from typing import Any

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


def _cfg_get_first(config: Any, paths: list[str], default=None):
    for path in paths:
        value = _cfg_get(config, path, default)
        if value is not default and value not in (None, ""):
            return value
    return default


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
        requested_device = str(
            _cfg_get_first(
                config,
                ["ranker.device", "ranker_training.device"],
                "cuda",
            )
        )
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
        self.model_path = _cfg_get_first(
            config,
            ["ranker.model_path"],
            "/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2",
        )
        self.rank_encoder_path = _cfg_get_first(
            config,
            ["ranker.encoder_path"],
            self.model_path,
        )
        self.temperature = float(
            _cfg_get_first(
                config,
                ["ranker_training.loss.temperature"],
                0.05,
            )
        )
        self.max_grad_norm = float(
            _cfg_get_first(
                config,
                ["ranker_training.max_grad_norm"],
                1.0,
            )
        )
        self.rank_top_k = int(_cfg_get_first(config, ["ranker.top_k"], 5))
        self.tokenizer = None
        self.encoder = None
        self.optimizer = None
        self.scheduler = None
        self.step = 0

    def init_model(self):
        trust_remote_code = bool(
            _cfg_get_first(
                self.config,
                ["ranker.trust_remote_code"],
                True,
            )
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=trust_remote_code)
        self.encoder = AutoModel.from_pretrained(self.rank_encoder_path, trust_remote_code=trust_remote_code)
        self.encoder.to(self.device)
        self.encoder.train()

        trainable = list(self.encoder.parameters())
        lr = float(
            _cfg_get_first(
                self.config,
                ["ranker_training.optim.lr"],
                2e-5,
            )
        )
        weight_decay = float(
            _cfg_get_first(
                self.config,
                ["ranker_training.optim.weight_decay"],
                0.01,
            )
        )
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
        total_steps = int(
            _cfg_get_first(
                self.config,
                ["ranker_training.optim.total_steps"],
                1000,
            )
        )
        warmup_steps = int(
            _cfg_get_first(
                self.config,
                ["ranker_training.optim.warmup_steps"],
                0,
            )
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(total_steps, 1),
        )
        print(
            "[ranker-worker] initialized shared_encoder=true "
            f"path={self.model_path} rank_encoder={self.rank_encoder_path} "
            f"device={self.device} top_k={self.rank_top_k}",
            flush=True,
        )

    @torch.no_grad()
    def encode_query(self, texts: list[str], max_length: int = 256) -> torch.Tensor:
        if self.encoder is None:
            self.init_model()
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
    def encode_docs(self, texts: list[str], max_length: int = 256) -> torch.Tensor:
        if self.encoder is None:
            self.init_model()
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
        top_k: int | None = None,
        max_query_length: int = 192,
        max_doc_length: int = 256,
    ) -> list[dict[str, Any]]:
        if self.encoder is None:
            self.init_model()
        top_k = self.rank_top_k if top_k is None else int(top_k)
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

        bsz, docs_per_query, doc_len = doc_input_ids.shape

        query_outputs = self.encoder(input_ids=query_input_ids, attention_mask=query_attention_mask)
        query_emb = mean_pool(query_outputs.last_hidden_state, query_attention_mask)

        flat_doc_input_ids = doc_input_ids.reshape(bsz * docs_per_query, doc_len)
        flat_doc_attention_mask = doc_attention_mask.reshape(bsz * docs_per_query, doc_len)
        doc_outputs = self.encoder(input_ids=flat_doc_input_ids, attention_mask=flat_doc_attention_mask)
        doc_emb = mean_pool(doc_outputs.last_hidden_state, flat_doc_attention_mask)

        doc_emb = doc_emb.reshape(bsz, docs_per_query, -1)
        query_emb = F.normalize(query_emb, dim=-1)
        doc_emb = F.normalize(doc_emb, dim=-1)
        scores = torch.einsum("bh,bkh->bk", query_emb, doc_emb)
        logits = scores / self.temperature

        per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
        if loss_weights is not None:
            loss = (per_sample_loss * loss_weights).sum() / loss_weights.sum().clamp_min(1.0)
        else:
            loss = per_sample_loss.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1

        pred = logits.argmax(dim=-1)
        acc_at_1 = (pred == labels).float().mean()
        pos_scores = scores.gather(1, labels[:, None]).squeeze(1)
        neg_mask = torch.ones_like(scores, dtype=torch.bool)
        neg_mask.scatter_(1, labels[:, None], False)
        neg_scores = scores[neg_mask].reshape(bsz, docs_per_query - 1)

        ranks = torch.argsort(logits, dim=-1, descending=True)
        positive_ranks = (ranks == labels[:, None]).nonzero(as_tuple=False)[:, 1].float() + 1.0
        mrr = (1.0 / positive_ranks).mean()
        lr = self.scheduler.get_last_lr()[0] if self.scheduler is not None else self.optimizer.param_groups[0]["lr"]

        metrics = {
            "ranker/loss": float(loss.detach().cpu()),
            "ranker/acc@1": float(acc_at_1.detach().cpu()),
            "ranker/mrr": float(mrr.detach().cpu()),
            "ranker/pos_score_mean": float(pos_scores.mean().detach().cpu()),
            "ranker/neg_score_mean": float(neg_scores.mean().detach().cpu()),
            "ranker/score_margin": float((pos_scores.mean() - neg_scores.mean()).detach().cpu()),
            "ranker/num_queries": int(bsz),
            "ranker/num_docs_per_query": int(docs_per_query),
            "ranker/num_neg": int(bsz * (docs_per_query - 1)),
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
