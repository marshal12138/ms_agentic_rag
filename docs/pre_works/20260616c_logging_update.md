## user:
目前日志中的/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/train_logs/coAgenticRetriever/260616-161547-CAR_async_ranker_training_ds_flash_v1/rollout_data/main_agent/1.jsonl只存了agent 轨迹，没有observation的完整内容，是否可以打印出完整内容？如果已经实现了，那么通过哪个参数可以控制？它应该是类似于eval脚本中的KEEP_TRACE=full/partial配置字段；如果没有实现，给出你的实现计划
## assistant:
目前没有实现类似 eval `KEEP_TRACE=full/partial` 的训练侧开关。

现状是：

- `rollout_data/main_agent/1.jsonl` 只 dump 了 `input/output/gts/score/step` 和 reward extra 字段。
- observation/tool response 在 `output` 里是 agent 实际看到的文本，已经受 `MAX_TOOL_RESPONSE_LENGTH=2048` 截断，所以出现 `...(truncated)`。
- 完整 top50 文档其实在训练时已存在于 `tool_call_details`：
  - `recall_top50_docs`
  - `rank_top50_docs`
  - `rank_top5_docs`
- 但 `_log_rollout_data()` 没把 `batch.non_tensor_batch["tool_call_details"]` 写入 jsonl，所以当前日志文件无法恢复完整 observation/top50。

当前已有相关控制参数：

- `MAX_TOOL_RESPONSE_LENGTH`：控制 agent 实际看到的 tool response 最大字符数，不是日志 full/partial。
- `DUMP_ROLLOUT_EVERY_STEP_NUM`：每隔多少 step dump。
- `DUMP_ROLLOUT_NUM_EVERYTIME`：每次 dump 多少条。
- `MAX_ROLLOUT_DUMP_NUM`：最多保留多少 dump 文件。

实现计划：

1. 新增训练侧参数，例如 `ROLLOUT_TRACE_MODE=partial|full`，默认 `partial` 保持现有行为。
2. 在 `01_train...sh` 和 `00_run_agentic_iter_rag_verl.sh` 透传为 Hydra override，例如 `trainer.rollout_trace_mode="${ROLLOUT_TRACE_MODE}"`。
3. 在 `CoAgenticRankerContrastiveRayTrainer._log_rollout_data()` 或 dump 前处理：
   - `partial`：保持现状。
   - `full`：额外写出 `tool_call_details`，包括 recall top50、rank top50、rank top5。
4. 为避免 `main_agent/1.jsonl` 过大，可以同时支持 `ROLLOUT_TRACE_MODE=separate_full`：
   - `main_agent/1.jsonl` 保持轻量。
   - 完整 observation 写到 `rollout_data/main_agent_full/1.jsonl` 或 `tool_call_details/1.jsonl`。
5. 加 dry-run 输出该参数，并用 1-step 小训练验证 jsonl 中确实含 `tool_call_details[0].rank_top50_docs` 长度 50。

## user:
好的，立刻实现这个新的日志功能

## 我会直接实现训练侧 ROLLOUT_TRACE_MODE=partial|full。实现会分两层：agent loop 保存未被 MAX_TOOL_RESPONSE_LENGTH 截断前的 tool response；trainer dump 在 full 时把 tool_call_details 写入 rollout jsonl，默认 partial 保持现有文件大小和行为。




我先补齐入口到 Hydra 的参数链路，再改 trainer dump。这个开关默认 partial，所以旧训练日志格式不变；只有显式 ROLLOUT_TRACE_MODE=full 才会把完整 details 写进 rollout jsonl。




Steered conversation

明白，默认改成 full。这样后续训练默认会写完整 tool_call_details / observation 详情；需要轻量日志时再显式设 ROLLOUT_TRACE_MODE=partial。我继续实现。




我会先加参数校验，避免拼写错误静默退回；支持 full 和 partial 两个值，默认 full。







补丁里的 dry-run 上下文和当前文件略有出入，我先定位精确位置再分块改，避免误伤脚本。

## assistant:
已实现训练侧 rollout 完整日志开关。

控制参数：

- `ROLLOUT_TRACE_MODE=full|partial`
- 默认值：`full`
- 透传 Hydra 字段：`trainer.rollout_trace_mode`

行为：

- `full`：`rollout_data/main_agent/<step>.jsonl` 每条样本额外包含：
  - `rollout_trace_mode`
  - `tool_call_details`
  - `messages`
  - `initial_query`
  - `answers`
  - `request_id`
- `partial`：保持轻量日志，只保留原有轨迹、reward extra 和 `rollout_trace_mode`。

`tool_call_details` 每个 tool call 里新增：

- `full_observation`：截断前的完整 tool response
- `agent_observation`：受 `MAX_TOOL_RESPONSE_LENGTH` 截断后、agent 实际看到的 response
- `observation_truncated`
- `full_observation_chars`
- `agent_observation_chars`
- `max_tool_response_length`
- `tool_response_truncate_side`
- 原已有完整文档字段继续保留：
  - `recall_top50_docs`
  - `rank_top50_docs`
  - `rank_top5_docs`

改动文件：

- `CoAgenticRetriever/verl/verl/experimental/agent_loop/coagentic_retriever_agent_loop.py`
- `CoAgenticRetriever/verl/verl/trainer/ppo/coagentic_retriever_ray_trainer.py`
- `CoAgenticRetriever/config/coagentic_retriever_trainer.yaml`
- `scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh`
- `scripts/coagenticRetriever_local/assets/00_run_agentic_iter_rag_verl.sh`

验证结果：

- `bash -n scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh` 通过。
- `bash -n scripts/coagenticRetriever_local/assets/00_run_agentic_iter_rag_verl.sh` 通过。
- `python -m py_compile ...` 通过。
- `/data04/envs/ms/ms_cosearch_official/bin/python -m py_compile ...` 通过。
- dry-run 通过，输出包含 `rollout trace mode: full`。
- 非法值验证通过：`ROLLOUT_TRACE_MODE=bad` 会退出并提示 `use full or partial`。
- 轻量 dump 模拟通过：
  - `full` jsonl 含 `tool_call_details[0].rank_top50_docs` 和 `full_observation`。
  - `partial` jsonl 不含 `tool_call_details`。
