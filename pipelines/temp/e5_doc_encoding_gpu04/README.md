# E5 wiki-18 doc encoding pressure test

This temp pipeline measures:

1. Full wiki-18 document encoding with E5-base-v2 on GPU06.
2. Loading the full FlatIP embedding index into the GPU-resident retriever on GPU07.

Default paths are local to this workspace:

- Corpus: `CoSearch_derevitives/data/retrieval/wiki-18/wiki-18.jsonl`
- Existing full index: `CoSearch_derevitives/data/retrieval/wiki-18/e5_Flat.index`
- Model: `models/retriever/e5-base-v2`

Run encoding:

```bash
bash CoSearch_derevitives/pipelines/temp/e5_doc_encoding_gpu04/00_run_encode_gpu06.sh
```

The encoding script defaults to `BATCH_SIZE=256`,
`TOKENIZERS_PARALLELISM=true`, and `RAYON_NUM_THREADS=32`. In the local sweep,
larger batches consumed much more memory but were slower for this long-text
corpus, so tokenizer parallelism is the useful speedup knob.

Quick batch-size sweep before a full run:

```bash
bash CoSearch_derevitives/pipelines/temp/e5_doc_encoding_gpu04/02_sweep_encode_batch_gpu06.sh
```

Two-stage cached-token path:

```bash
bash CoSearch_derevitives/pipelines/temp/e5_doc_encoding_gpu04/03_run_pretokenize_cpu.sh
TOKENS_META=/path/to/tokens_cpu/tokens_meta.json \
  bash CoSearch_derevitives/pipelines/temp/e5_doc_encoding_gpu04/04_run_encode_from_tokens_gpu06.sh
```

This stores fixed-length E5 `input_ids` and `attention_mask` before GPU
encoding. Full wiki-18 token cache is about 15-16 GiB at `max_length=256`.

Run retriever restart/load on GPU07. This starts an old retriever first,
measures the old-service shutdown time, then starts the retriever again with
`INDEX_FILE`.

```bash
bash CoSearch_derevitives/pipelines/temp/e5_doc_encoding_gpu04/01_run_load_retriever_gpu07.sh
```

Use `INDEX_FILE=/path/to/new/e5_Flat.index` for the newly encoded embedding
and `OLD_INDEX_FILE=/path/to/old/e5_Flat.index` for the existing service.
