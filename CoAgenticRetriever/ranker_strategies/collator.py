"""Collator for ranker contrastive batches."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto

from .schemas import ContrastiveSample


class RankerContrastiveCollator:
    def __init__(
        self,
        tokenizer,
        max_query_length: int,
        max_doc_length: int,
        use_e5_prefix: bool,
    ):
        self.tokenizer = tokenizer
        self.max_query_length = int(max_query_length)
        self.max_doc_length = int(max_doc_length)
        if use_e5_prefix is None:
            tokenizer_name = str(getattr(tokenizer, "name_or_path", "") or "")
            use_e5_prefix = "e5" in tokenizer_name.lower()
        self.use_e5_prefix = bool(use_e5_prefix)

    def _format_query(self, text: str) -> str:
        return f"query: {text}" if self.use_e5_prefix else text

    def _format_doc(self, text: str) -> str:
        return f"passage: {text}" if self.use_e5_prefix else text

    def __call__(self, samples: list[ContrastiveSample]) -> DataProto:
        if not samples:
            raise ValueError("RankerContrastiveCollator received no samples.")

        query_texts = [self._format_query(sample.query_input) for sample in samples]
        doc_groups = [sample.documents for sample in samples]
        docs_per_query = len(doc_groups[0])
        if any(len(group) != docs_per_query for group in doc_groups):
            raise ValueError("All contrastive samples must have the same number of documents.")

        doc_texts = []
        for group in doc_groups:
            for doc in group:
                prefix = f"{doc.title}\n" if doc.title else ""
                doc_texts.append(self._format_doc(prefix + doc.text))

        query_tokens = self.tokenizer(
            query_texts,
            padding=True,
            truncation=True,
            max_length=self.max_query_length,
            return_tensors="pt",
        )
        doc_tokens = self.tokenizer(
            doc_texts,
            padding=True,
            truncation=True,
            max_length=self.max_doc_length,
            return_tensors="pt",
        )

        batch_size = len(samples)
        doc_input_ids = doc_tokens["input_ids"].reshape(batch_size, docs_per_query, -1)
        doc_attention_mask = doc_tokens["attention_mask"].reshape(batch_size, docs_per_query, -1)
        loss_weights = torch.tensor([sample.loss_weight for sample in samples], dtype=torch.float32)
        positive_doc_index = torch.tensor([sample.positive_doc_index for sample in samples], dtype=torch.long)

        batch = TensorDict(
            {
                "query_input_ids": query_tokens["input_ids"],
                "query_attention_mask": query_tokens["attention_mask"],
                "doc_input_ids": doc_input_ids,
                "doc_attention_mask": doc_attention_mask,
                "positive_doc_index": positive_doc_index,
                "loss_weights": loss_weights,
            },
            batch_size=batch_size,
        )

        non_tensor_batch: dict[str, Any] = {
            "origin_query": np.array([sample.origin_query for sample in samples], dtype=object),
            "sub_query": np.array([sample.sub_query for sample in samples], dtype=object),
            "doc_ids": np.array([[doc.doc_id for doc in sample.documents] for sample in samples], dtype=object),
            "doc_ranks": np.array([[doc.rank for doc in sample.documents] for sample in samples], dtype=object),
            "trajectory_id": np.array([sample.trajectory_id for sample in samples], dtype=object),
            "tool_call_id": np.array([sample.tool_call_id for sample in samples], dtype=object),
            "label_source": np.array([sample.label_source for sample in samples], dtype=object),
            "sample_source": np.array([sample.sample_source for sample in samples], dtype=object),
            "sample_id": np.array([sample.sample_id for sample in samples], dtype=object),
        }
        meta_info = {
            "num_docs_per_query": docs_per_query,
            "construction_examples": [serialize_sample_for_log(samples[0])],
        }
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


def serialize_sample_for_log(sample: ContrastiveSample, text_chars: int = 240) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "trajectory_id": sample.trajectory_id,
        "tool_call_id": sample.tool_call_id,
        "origin_query": sample.origin_query,
        "sub_query": sample.sub_query,
        "label_source": sample.label_source,
        "recall_top50_sample": [
            {
                "doc_id": doc.doc_id,
                "rank": doc.rank,
                "score": doc.recall_score,
                "text": doc.text[:text_chars],
            }
            for doc in sample.recall_top50_docs[:5]
        ],
        "rank_top50_sample": [
            {
                "doc_id": doc.doc_id,
                "rank": doc.rank,
                "score": doc.recall_score,
                "rank_score": doc.metadata.get("rank_score"),
                "text": doc.text[:text_chars],
            }
            for doc in sample.rank_top50_docs[:5]
        ],
        "rank_top5_sample": [
            {
                "doc_id": doc.doc_id,
                "rank": doc.rank,
                "score": doc.recall_score,
                "rank_score": doc.metadata.get("rank_score"),
                "text": doc.text[:text_chars],
            }
            for doc in sample.rank_top5_docs[:5]
        ],
        "positive": {
            "doc_id": sample.positive.doc_id,
            "rank": sample.positive.rank,
            "score": sample.positive.recall_score,
            "text": sample.positive.text[:text_chars],
        },
        "negatives": [
            {
                "doc_id": doc.doc_id,
                "rank": doc.rank,
                "score": doc.recall_score,
                "text": doc.text[:text_chars],
            }
            for doc in sample.negatives
        ],
        "positive_doc_index": sample.positive_doc_index,
        "sample_source": sample.sample_source,
        "metadata": sample.metadata,
    }
