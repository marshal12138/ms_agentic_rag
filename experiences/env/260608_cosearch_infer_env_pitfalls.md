# 260608 CoSearch Infer 环境踩坑记录

## 适用范围

本记录适用于在 `CoAgenticRtriver` 中启动 dense retriever、retrieval proxy、Ray、vLLM、VERL infer / train 这类依赖 GPU、本地端口和长期后台进程的任务。

## 1. 服务类 / GPU 类命令必须一开始就提权

症状：

- 在 sandbox 内启动 retriever / proxy / Ray / vLLM 后，流程可以跑很久，但到端口访问、GPU 服务或本地 HTTP 联通阶段才暴露权限问题。
- 之后不得不重新用 `require_escalated` 跑同一条长命令，造成无谓等待。

原因：

- 这类命令不是普通文件读写，而是启动本地服务、绑定端口、访问 GPU、创建 Ray 进程和 vLLM engine。
- sandbox 下即使某些步骤看起来能启动，也不能作为可靠运行环境。

正确处理：

- 只读检查、`DRY_RUN=1`、`grep` / `find` / `sed`、`compileall`、已存在 JSONL 的字段检查，可以不提权。
- 以下命令必须一开始就用 `sandbox_permissions=require_escalated`：
  - `00_start_dense_retriever_server.sh`
  - `01_train_*.sh`
  - `02_infer_*.sh`
  - `run_prepare_chunk_ranking_examples.sh`
  - 任何启动 Ray / vLLM / 本地 HTTP server / GPU 长任务 / 模型下载的命令

反思：

- 这不是需要通过失败来验证的事项。只要命令会启动本地服务、绑定端口、访问 GPU 或启动 Ray/vLLM，就应在首次执行前要求提权。
- 不允许先在 sandbox 中跑长流程再观察是否失败；这会把权限问题拖到 10-20 分钟后才暴露。

## 1.1 retriever 验证通过后不要反复重启

症状：

- retriever / proxy 已经验证联通，但后续为了排查 agent、checkpoint、metrics 等问题，每次都重新执行完整 wrapper。
- 每次重启 retriever 都重新等待加载 index/corpus 和端口 ready，浪费时间。

原因：

- retriever 是相对独立的服务依赖；一旦验证 `http://127.0.0.1:${PROXY_PORT}/retrieve` 可用，后续错误通常在 Ray/vLLM/VERL/checkpoint/reward/metrics。
- 把 retriever 生命周期和 agent infer 生命周期绑死，会导致排查 agent 问题时重复支付 retriever 启动成本。

正确处理：

- 首次启动并验证 retriever/proxy 后，保持服务常驻。
- 后续只重跑 agent infer / VERL 部分。
- 脚本应支持类似开关：
  - `REUSE_RETRIEVER=1`
  - `KEEP_RETRIEVER_ALIVE=1`
  - `SKIP_RETRIEVER_START=1`
- 如果没有这些开关，先补脚本，不要手工重复启动完整 wrapper。

## 2. 当前 H20 显存不是空闲状态

症状：

- vLLM 初始化报显存不足，例如 free memory 约 30GB，但 `gpu_memory_utilization=0.45` 需要 40GB 以上。
- 主 agent 初始化后再启动 reranker server，会再次申请显存并失败。

原因：

- 机器上已有其它 vLLM worker group 长驻，占用每卡约 64GB。
- H20 总显存大，但不能按空卡估算。

正确处理：

- 正式跑前先看 `nvidia-smi`。
- 在已有大进程占用时，优先使用保守参数：
  - `GPU_MEMORY_UTILIZATION=0.30`
  - `AGENT_WORKERS=4`
  - `MAX_NUM_SEQS=4`
  - `VAL_BATCH_SIZE=4`
  - `TRAIN_BATCH_SIZE=4`
  - `ACTOR_BATCH_SIZE=4`
- 如果仍失败，按顺序降低：
  - `GPU_MEMORY_UTILIZATION=0.25` 或 `0.22`
  - `MAX_NUM_SEQS=3`、`2`
  - `AGENT_WORKERS=3`、`2`、`1`
- batch size 要和当前 FSDP world size / GPU 数保持可整除，否则会触发 batch divisibility 校验失败。

## 2.1 禁用 reranker 后要重新评估 main agent 可用 GPU

症状：

- 原脚本按双 agent 资源拆分，默认前 4 张卡给 main agent，后 4 张卡给 reranker。
- 当任务明确“不使用 reranker”时，如果仍固定 `GPU_IDS=0,1,2,3`，就没有充分利用 8 卡资源。

正确处理：

- 禁用 reranker 后，首先重新评估是否可把全部可用 GPU 给 main agent。
- 但不能盲目把 4 卡改 8 卡，必须先检查 checkpoint FSDP world size。
- 如果 checkpoint 目录只有：

```text
model_world_size_4_rank_0.pt
...
model_world_size_4_rank_3.pt
```

则当前 resume world size 是 4。直接用 8 卡会去找 `model_world_size_8_rank_*.pt`，不能直接加载。

决策规则：

1. 先检查 checkpoint 分片 world size。
2. 如果 checkpoint world size 与目标 GPU 数一致，直接使用目标 GPU 数。
3. 如果不一致，先做 checkpoint merge / reshard / 转换，再切换 GPU 数。
4. 如果只是为了快速产出数据集，优先使用 checkpoint 原始 world size，避免引入转换风险。

## 3. 本项目 VERL 副本可能缺少从源项目迁移的模块和配置

症状：

- `ModuleNotFoundError: No module named 'verl.models'`
- `ModuleNotFoundError: No module named 'verl.models.transformers'`
- Hydra 报找不到 `config/data/legacy_data`

原因：

- `CoAgenticRtriver/CoAgenticRtriver/verl/verl/` 是本项目本地 VERL 副本，可能没有完整迁移 `models/`。
- `config/data/legacy_data.yaml` 也可能缺失。

正确处理：

- 确认存在：

```text
CoAgenticRtriver/CoAgenticRtriver/verl/verl/models/
CoAgenticRtriver/CoAgenticRtriver/config/data/legacy_data.yaml
```

- 如缺失，从源项目补齐：

```text
CoSearch_derevitives/CoSearch/verl/verl/models/
CoSearch_derevitives/CoSearch/config/data/legacy_data.yaml
```

- 补齐后至少做 import / compile smoke test，再启动长任务。

## 4. 运行环境固定优先使用 ms_cosearch_official

本次 VERL / CoSearch infer 跑通使用的是：

```text
/data04/envs/ms/ms_cosearch_official/bin/python
```

不要在长任务里临时切换到未知 Python 环境。环境不一致会把问题伪装成代码问题。

## 5. 本机没有 `rg` 时直接用 `grep` / `find`

本机环境可能没有 `rg`。遇到搜索任务时不要反复试错，直接使用：

```bash
grep -R ...
find ... -type f ...
```

## 复用检查清单

启动任何 CoSearch 长任务前先确认：

1. 是否需要本地端口、GPU、Ray、vLLM；如果需要，直接提权。
2. `nvidia-smi` 是否显示已有大进程；如果有，先用保守显存参数。
3. retriever/proxy 是否已验证可用；如果可用，后续不要反复重启。
4. 禁用 reranker 后是否重新评估 main agent GPU 数。
5. checkpoint 的 world size 是否匹配目标 GPU 数。
6. 本项目 VERL 副本是否有 `verl.models` 和 `config/data/legacy_data.yaml`。
7. Python 环境是否为 `/data04/envs/ms/ms_cosearch_official/bin/python`。
8. checkpoint 的加载内容是否匹配当前 infer / train 场景。
