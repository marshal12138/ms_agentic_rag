# CoAgenticRetriever: Joint Training of Retrieval Reasoning and Document Ranking

CoAgenticRetriever jointly trains a **multi-step retrieval policy** and a **generative document ranker** via GRPO for tool-augmented retrieval. The policy issues sub-queries through the retrieval tool; the ranker reorders candidate documents from a fixed dense retriever before the policy observes them. Both parts are optimized end-to-end from answer correctness.

Two technical contributions make this work:
- **Semantic grouping**: clusters sub-queries by token-level F1 similarity to form valid GRPO groups for the ranker, improving sampling efficiency without additional rollouts.
- **Composite tool score**: combines a ranking quality signal (Hit@k) with trajectory-level answer correctness to give the ranker both immediate and long-term learning signals.

## Step 1: Set Up Environments
### Training environment (`search-llm`)

```bash
bash conda_setup/setup_conda_env.sh
conda activate search-llm
```

See [conda_setup/README.md](conda_setup/README.md) for optional flags (CUDA version, force-recreate, skip flash-attn).

### Retriever environment (`retriever`)

We use e5-base as the retriever. The retrieval server setup follows [Search-R1](https://github.com/PeterGriffinJin/Search-R1), which is already cloned at `Search-R1/`. You can create the conda environment with:

```bash
conda create -n retriever python=3.10
conda activate retriever

# Install torch with conda (needed for faiss-gpu)
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets

# faiss-gpu for efficient retrieval
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# FastAPI server
pip install uvicorn fastapi
```

---

## Step 2: Download Retrieval Index and Corpus

The retriever uses a Wikipedia passage index (e5-base-v2 embeddings). Download from Search-R1:

```bash
save_path=/your/data/path

cd Search-R1
python scripts/download.py --save_path $save_path

# Merge split index files
cat $save_path/part_* > $save_path/e5_Flat.index

# Decompress corpus
gzip -d $save_path/wiki-18.jsonl.gz
```

---

## Step 3: Launch the Retrieval Server

The retrieval server must be running before training starts. Current training
and evaluation tasks auto-start it through
`../scripts/coagenticRetriever_local/00_start_dense_retriever_server.sh` when
`AUTO_START_RECALL_SERVICE=1`.

To launch it manually, set the retrieval assets and model path:

```bash
INDEX_FILE="/path/to/e5_Flat.index"
CORPUS_FILE="/path/to/wiki-18.jsonl"
RETRIEVER_MODEL="/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2"
RECALL_TOP_K=50
PORT=8030
bash ../scripts/coagenticRetriever_local/00_start_dense_retriever_server.sh
```

Use `RETRIEVAL_SERVICE_URL="http://<host>:8030/retrieve"` if training runs in a
separate process.

---

## Step 4: Prepare Training and Evaluation Data

Before launching training, place the train/eval parquet files at the paths expected by the training script, or override the paths with `TRAIN_DATA` and `VAL_DATA`:

```bash
data/coAgenticRetriever/albation_1/train.parquet
data/coAgenticRetriever/albation_1/val.parquet
```

For custom paths:

```bash
EXP_NAME=my_coagentic_retriever_run \
bash ../scripts/coagenticRetriever_v2/01_train_launcher.sh --main_run_config=coAgenticRetriever_main
```

---

## Step 5: Launch CoAgenticRetriever Training

Pass the retriever URL as an environment variable when using an already running
retriever:

```bash
RETRIEVAL_SERVICE_URL="http://<retriever-hostname>:8030/retrieve" \
EXP_NAME=my_coagentic_retriever_run \
bash ../scripts/coagenticRetriever_v2/01_train_launcher.sh --main_run_config=coAgenticRetriever_main
```

The v2 launcher generates a per-run tool config from the final Hydra config, so
recall/ranker cutoffs stay in the canonical config chain instead of the static
tool template.

---

## Citation

If you use the original CoSearch work that this code evolved from, please cite:

- https://arxiv.org/abs/2604.17555

```bibtex
@article{zeng2026cosearch,
	title={CoSearch: Joint Training of Reasoning and Document Ranking via Reinforcement Learning for Agentic Search},
	author={Hansi Zeng, Liam Collins, Bhuvesh Kumar, Neil Shah, Hamed Zamani},
	journal={arXiv preprint arXiv:2604.17555},
	year={2026},
	doi={10.48550/arXiv.2604.17555}
}
```

---

## Authorship

This repository contains sample code developed as part of a collaboration between Snap Inc. and the University of Massachusetts Amherst. Rights to the sample code remain with the original author(s) and are licensed under the terms described in the LICENSE file.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). See the [LICENSE](LICENSE) file for details.
