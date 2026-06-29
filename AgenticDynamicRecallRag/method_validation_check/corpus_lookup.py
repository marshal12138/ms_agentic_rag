"""Corpus loading and answer-containment checking.

Documents are looked up by integer id (the corpus row index) via the helpers
from the readme. A document "contains the answer" if any golden-answer string
appears (case-insensitive) in its text.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import datasets

from .config import Config


def load_corpus(corpus_path: str):
    return datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)


def load_docs(corpus, doc_idxs: Iterable):
    return [corpus[int(idx)] for idx in doc_idxs]


def doc_text(doc) -> str:
    if isinstance(doc, dict):
        for key in ("contents", "text", "content", "passage"):
            val = doc.get(key)
            if val:
                return str(val)
        return " ".join(str(v) for v in doc.values())
    return str(doc)


class CorpusLookup:
    def __init__(self, config: Config):
        self.corpus = load_corpus(config.corpus_path)

    def texts_for(self, doc_idxs: List) -> Dict[object, str]:
        """Map each requested id to its document text (deduplicated fetch)."""
        unique = list(dict.fromkeys(doc_idxs))
        docs = load_docs(self.corpus, unique)
        return {idx: doc_text(doc) for idx, doc in zip(unique, docs)}


def contains_answer(text: str, golden_answers: List[str]) -> bool:
    if not text:
        return False
    lowered = text.lower()
    for ans in golden_answers:
        if ans and str(ans).lower() in lowered:
            return True
    return False
