# 2026-06-22 CoAgenticRetriever CoSearch-Aligned Eval 工作记录

## 背景

在比较 CoSearch 与 CoAgenticRetriever 的 no-ranker 评估结果时，发现二者虽然看起来使用相同 agent 模型、相同评估数据、且都不启用 reranker，但实际推理行为并不一致。

主要差异包括：

- CoAgenticRetriever 原 eval 会向 Qwen chat template 注入 tool schema，CoSearch 源流程不注入。
- CoAgenticRetriever 原 eval 的 turn/context budget 与 CoSearch 源流程不同。
- vLLM serving 参数也存在差异，例如 `max_model_len` 与 tensor parallel 设置。

这些差异会改变实际送入 Qwen3-4B 的 prompt，从而改变首轮 search query、retrieval top-k 文档和最终答案。因此，不能直接把原 CoAgenticRetriever eval 与 CoSearch eval 视为同等设置。

## 本次新增内容

新增了一条 CoSearch-aligned 的 CoAgenticRetriever 推理链路，用于把 CoAgenticRetriever eval 的 prompt/context 行为尽量对齐 CoSearch 源流程。

新增/使用的入口包括：

- `scripts/coagenticRetriever_local/evaluate_coagentic_vllm_cosearch_aligned.py`
- `scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only_cosearch_aligned.sh`
- `tasks/eval_tasks/coAgenticRetriever/eval_CAR_async_label_dpskv4f_v0622_cosearch_aligned.sh`

其中 task 脚本包含两段对照：

- `async_label_dpskv4f_v0616_full_cosearch_aligned`
- `async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned`

第一段启用 dense E5 ranker，第二段使用 no-ranker 模式，便于做 full vs no-ranker 对照。

## 当前关键默认行为

`02_infer_qwen3_4b_ablation_val_only_cosearch_aligned.sh` 当前默认：

- `ENABLE_THINKING=false`
- `INJECT_TOOL_SCHEMA=false`
- `MAX_ASSISTANT_TURNS=6`
- `MAX_USER_TURNS=6`
- `MAX_PROMPT_LENGTH=11264`
- `MAX_RESPONSE_LENGTH=1024`
- `MAX_MODEL_LEN=12288`
- `MAX_TOOL_RESPONSE_LENGTH=4096`

这些默认值使 aligned eval 更接近当前 CoSearch 推理行为：

- 不向 Qwen chat template 传入 `tools=...`。
- 允许 6 轮 assistant/user 交互。
- 使用更大的 prompt budget 和较短 response budget。

需要注意的是，当前 task 脚本 `eval_CAR_async_label_dpskv4f_v0622_cosearch_aligned.sh` 本身没有显式传入这些 `MAX_*` 和 `INJECT_TOOL_SCHEMA` 参数，而是依赖 aligned launcher 的默认值。

## 当前端口与资源设置

`eval_CAR_async_label_dpskv4f_v0622_cosearch_aligned.sh` 当前默认：

- `AGENT_GPU_IDS=0,1`
- `AGENT_TP_SIZE=2`
- `RANK_GPU_ID=2`
- `RECALL_GPU_ID=3`
- `PROXY_PORT=8030`

注意：虽然之前讨论过使用 `8035` 来避免撞已有 `8030` retriever，但当前脚本实际默认仍是 `8030`。如果希望避免复用或碰撞已有 retriever 服务，应在运行时显式传入：

```bash
PROXY_PORT=8035 bash tasks/eval_tasks/coAgenticRetriever/eval_CAR_async_label_dpskv4f_v0622_cosearch_aligned.sh
```

或者后续将 task 默认端口改成 `8035`。

## 和原 CoAgenticRetriever Eval 的区别

原始 eval 链路：

- `scripts/coagenticRetriever_local/evaluate_coagentic_vllm.py`
- `scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh`

它的默认行为更接近 CoAgenticRetriever 原实现：

- 默认注入 Qwen tool schema。
- 默认 turn/context budget 较小，例如 `MAX_ASSISTANT_TURNS=2`、`MAX_USER_TURNS=2`、`MAX_MODEL_LEN=8192`。

aligned eval 链路的核心目的不是替代原始 eval，而是提供一个更接近 CoSearch 源流程的对照入口，用来排除 prompt/template/budget 差异对评估结果的干扰。

## 后续待处理

后续如果要让训练与 aligned eval 完全一致，还需要处理训练侧：

- 训练 budget 应进入 YAML 配置，作为默认训练参数管理。
- `train_CAR_async_ranker_training_ds_flash_mix_signal_fix.sh` 应只保留实验编排和显式开关。
- Qwen tool schema 注入应作为显式开关放在训练 task 脚本中，默认 `false`。
- CoAgenticRetriever 训练 agent loop 需要支持关闭 `tools=...` 注入，同时保留工具实际调用能力。

