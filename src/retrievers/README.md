# Retriever Services

This directory contains shared retriever service launchers and local service
implementations.

## Unified Launcher

Use `start_dense_retriever_server.sh` as the entry point.

Default mode is CPU:

```bash
bash src/retrievers/start_dense_retriever_server.sh
```

GPU mode:

```bash
bash src/retrievers/start_dense_retriever_server.sh \
  --mode gpu \
  --gpu-id 5 \
  --doc-dtype float16 \
  --port 8050
```

Options:

- `--mode cpu|gpu`: serving mode. Default: `cpu`.
- `--gpu-id GPU_ID`: GPU id used by GPU mode. Default: `5`.
- `--doc-dtype float16|float32`: GPU doc embedding dtype. Default: `float16`.
- `--query-batch-size N`: internal GPU query batch size. Default: `32`.
- `--port PORT`: service port. Default: `8010`.

Environment variables are also supported: `MODE`, `GPU_ID`, `DOC_DTYPE`,
`QUERY_BATCH_SIZE`, `PORT`, `PY`, `INDEX_FILE`, `CORPUS_FILE`,
`RETRIEVER_MODEL`.

## CPU Mode

CPU mode launches the Search-R1 native retrieval server:

```text
CoSearch/Search-R1/search_r1/search/retrieval_server.py
```

It uses the original FAISS `IndexFlatIP` search path and keeps the document
embedding index on CPU.

## GPU Mode

GPU mode launches:

```text
src/retrievers/gpu_dense_retriever_server.py
```

This server loads the full `e5_Flat.index` document embeddings into GPU memory
as a torch tensor and runs retrieval as:

```python
scores = query_emb @ doc_embeddings.T
top_scores, top_idxs = torch.topk(scores, k=topk)
```

The query encoder is also placed on GPU. E5 query processing follows the
Search-R1 behavior: `query: ` prefix, mean pooling, and normalized query
embeddings.

The GPU implementation exists because the current Python environment exposes an
incomplete FAISS GPU API: `faiss.index_cpu_to_all_gpus` exists, but GPU resource
classes are missing, so the Search-R1 FAISS GPU path cannot be used reliably.

## Resource Notes

Measured on GPU 5, NVIDIA H20, wiki-18 full index:

- Full doc count: `21,015,324`
- Embedding dim: `768`
- `float16` GPU service: roughly `42GB` observed `nvidia-smi` memory after
  benchmark.
- `float32` GPU service: roughly `66GB` observed `nvidia-smi` peak memory after
  batch=1 benchmark.
- Startup time: roughly `40-45s` to ready state on the current machine.

Run GPU services outside the sandbox or with elevated execution. In sandboxed
commands, `nvidia-smi` may list the GPU while PyTorch CUDA or local socket access
still fails.

## Alignment Caveats

The GPU service uses the original FAISS flat index vectors, but it does not call
FAISS `index.search()` during serving.

- `float32` is closest to the original Search-R1 `IndexFlatIP` behavior.
- `float16` is faster and uses less memory, but can slightly change scores and
  top-k ordering when candidate scores are very close.
- The implementation assumes the FAISS flat index row order matches the corpus
  row order, which is the same assumption used by the Search-R1 server.

For strict reproduction, compare CPU FAISS top-k results against GPU torch
top-k results on a random query sample before reporting final metrics.
