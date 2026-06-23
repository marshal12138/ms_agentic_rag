# GPU Retriever Benchmark

This folder contains the GPU 5 retriever launch and QPS benchmark artifacts.
The launcher uses `src/retrievers/gpu_dense_retriever_server.py`, which keeps both
the E5 query encoder and wiki-18 document embeddings on GPU.

Default setup:

- GPU: `5`
- Retriever port: `8050`
- Device: `cuda`
- Doc embedding index: torch tensor on GPU, loaded from `e5_Flat.index`
- Doc embedding dtype: `float16`
- Query source: `data/co_search/local_flashrag/co_search_ablation.train.parquet`
- Output folders: `logs/`, `queries/`, `results/`, `run/`

Typical run:

```bash
bash 03_run_qps_sweep.sh
```

The sweep starts the retriever if needed, prepares query JSONL, runs benchmark
cases, and writes a markdown summary under `results/`.

GPU access note: run service startup and benchmark commands outside the sandbox
or with elevated execution. In sandboxed commands, `nvidia-smi` may still list
the GPU while PyTorch CUDA and local socket access fail.
