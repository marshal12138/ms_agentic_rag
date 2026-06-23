# E5 doc encoding acceleration notes

Date: 2026-06-09

Current bottleneck: CPU-side JSON reading/tokenization and long-sequence
padding, not GPU memory. The original full run used `BATCH_SIZE=256` with
`TOKENIZERS_PARALLELISM=false` and reached about `1197 docs/s` while using only
about `2.7 GiB` GPU memory.

Measured on the first `131,072` wiki-18 docs on GPU06:

| Setting | Throughput | GPU peak memory | Note |
| --- | ---: | ---: | --- |
| bs=256, tokenizer parallel off | ~1197 docs/s | ~2.7 GiB | Original full-run behavior |
| bs=256, tokenizer parallel on | ~1647 docs/s | ~2.7 GiB | Best measured setting |
| bs=1024, tokenizer parallel on | ~1525 docs/s | ~7.9 GiB | Slower despite more memory |
| bs=2048, tokenizer parallel on | ~1494 docs/s | ~14.8 GiB | Slower |
| bs=4096, tokenizer parallel on | ~1479 docs/s | ~28.6 GiB | Slower |

Recommendation for single-GPU full encoding on GPU06:

```bash
GPU_ID=6 BATCH_SIZE=256 TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=32 \
  bash CoSearch_derevitives/pipelines/temp/e5_doc_encoding_gpu04/00_run_encode_gpu06.sh
```

Estimated full encode time at the measured best rate:

- `21,015,324 / 1647 = 12,759 s = 3.54 h` for encoding.
- FAISS FlatIP write time for the final ~61GiB index still needs to be added
  from the full run.

If more speed is required, the next effective option is not larger single-GPU
batch size. It is sharding the corpus across multiple idle GPUs/processes,
writing per-shard embeddings/indexes, and merging them afterward. That trades
GPU/CPU memory and disk for wall-clock time.

Another useful option is a two-stage cached-token path:

1. `03_run_pretokenize_cpu.sh`: parse JSONL, add the E5 `passage:` prefix,
   tokenize, pad/truncate to `max_length=256`, and store `input_ids` as
   `uint16` plus `attention_mask` as `uint8`.
2. `04_run_encode_from_tokens_gpu06.sh`: load the cached memmaps and run only
   the E5 model forward pass plus FAISS index construction.

Expected full token-cache size:

- `input_ids`: `21,015,324 * 256 * 2 bytes ~= 10.0 GiB`
- `attention_mask`: `21,015,324 * 256 * 1 byte ~= 5.0 GiB`
- total: about `15-16 GiB`
