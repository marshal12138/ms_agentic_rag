## user:
训练和推理(/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/eval_tasks/eval_CAR_async_label_dpskv4f_v0616.sh)过程的真实轨迹格式是一样的吗？其prompt和response以及template是否会有差别？
## assistant:
不完全一样。当前这份 eval 脚本和训练脚本的真实交互结构相似，但 prompt/template 配置存在明显差异。

主要差异：

- **thinking 模式不同**
  - 训练脚本强制：`++data.apply_chat_template_kwargs.enable_thinking=False`
  - eval 脚本默认：`ENABLE_THINKING=true`
  - 结果是训练 prompt 会带 no-think stub：`<think>\n\n</think>`，但 eval 会让模型正常输出 `<think>...</think>`。

- **prompt 内容不同**
  - 当前训练样例要求：
    `1) <reason> ... </reason>`
  - 当前 eval 样例要求：
    `1) <think> ... </think>`
  - 所以 response 也会不同：训练输出多是 `<reason>...`，eval 输出多是 `<think>...`。

- **tool template 不同**
  - 训练走 VERL agent loop，并调用 tokenizer 的 `apply_chat_template(..., tools=tool_schemas, ...)`，prompt 里有系统级 `# Tools` 和 `<tools>...</tools>` schema。
  - eval evaluator 当前没有传 `tools=...` 给 `apply_chat_template`，只依赖 user prompt 里的文字说明工具格式。
  - 我抽样确认：训练 prompt 有 `<tools>`，eval prompt 没有。

- **生成参数不同**
  - 训练：`temperature=1.0`，`response_length=4096`，`prompt_length=4096`，`max_tool_response_length=2048`，默认 2 assistant/user turns。
  - eval：`temperature=0.0`，`max_response_length=1024`，`max_prompt_length=11264`，`max_tool_response_length=4096`，默认 6 assistant/user turns。

- **轨迹结构相似但不是同一格式**
  - 二者都是：assistant 输出 tool_call -> tool_response -> assistant 输出 answer。
  - 但训练 reward 接收的是 role-labelled transcript，类似：
    `assistant tool_call -> user <tool_response> -> assistant answer`
  - eval 主要保存 metrics/trace，并直接从最后 `<answer>` 算 EM/F1，不走训练 reward 函数。

结论：当前训练和 eval **不能认为是严格同分布推理**。如果要对齐，至少要把 eval 脚本改成：

```bash
ENABLE_THINKING=false
```

并使用和训练相同的 eval parquet/prompt 模板，最好也让 evaluator 传入同样的 tool schema chat template。否则 eval 会更像 “think 模式 + 另一个 prompt 格式” 下的评估。
## user:
我改动了默认的eval data path和thinking设置，现在再确认一下prompt 内容；将eval的生成参数和训练对齐；然后再继续比较一下train-eval的异同
## assistant:
已完成。主要改动在 [02_infer_qwen3_4b_ablation_val_only.sh](/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh:86)：

- `MAX_MODEL_LEN=8192`
- `MAX_ASSISTANT_TURNS=2`
- `MAX_USER_TURNS=2`
- `MAX_PROMPT_LENGTH=4096`
- `MAX_RESPONSE_LENGTH=4096`
- `MAX_TOOL_RESPONSE_LENGTH=2048`
- `TEMPERATURE=1.0`
- `TOP_P=1.0` 原本已一致

我也确认了当前默认 eval parquet：`data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet`。prompt 内容现在是 no-think 的 `<reason>` 版本：要求每轮输出 `<reason>...</reason>`，然后输出 `<tool_call>` 或 `<answer>`；首轮必须 search；数据 prompt 本身不含 `<think>`。但 Qwen3 在 `enable_thinking=false` 渲染时会自动在 assistant 起始处加入空的 `<think>\n\n</think>` stub，这是 chat template 行为，不代表数据 prompt 是 think 模式。

当前 train-eval 主要一致点：

- train/eval 都使用 `co_search_ablation.*.parquet` 的 `<reason>` prompt 风格。
- eval 默认 `ENABLE_THINKING=false`，和训练侧 `++data.apply_chat_template_kwargs.enable_thinking=False` 对齐。
- top-N/top-M 默认仍是 `50/5`。
- 生成长度、turn 数、temperature 已对齐训练默认。

仍然不一致的关键点：

- 训练 agent loop 调用 `apply_chat_template(..., tools=tool_schemas, enable_thinking=False)`，会额外注入 Qwen `<tools>` system schema；eval 当前只传 `enable_thinking`，没有传 `tools`，所以严格说渲染后的 prompt 还不是完全一致。
- eval 走 `/v1/completions` 手写多轮逻辑；训练走 VERL agent loop。
- eval 输出用 stop `<|im_end|>`，日志里通常不含 stop token；训练日志可能保留。
- eval 评估是 EM/F1/status；训练还经过 reward/format/min-one-search 约束。
- `eval_CAR_async_label_dpskv4f_v0616.sh` 里前两个 baseline block 仍显式写了 `.parquet.bak`，不会使用 base 脚本的新默认路径；后两个 async block 才会吃默认路径。

已做校验：`bash -n` 通过，`DRY_RUN=1` 通过，未启动 GPU 服务。

## user:
1. 要保证渲染后的 prompt也是一致的，eval和train对齐； 2. eval 必须走 /v1/completions 手写多轮逻辑，但是逻辑要和train对齐；3. eval 输出用 stop <|im_end|>，这点是否和train一致？如果不一致，也要以train为准，进行对齐；4.

## assistant：
我会按你的 1-3 直接对齐代码：先确认训练侧的模板、tool schema、停止条件和多轮拼接方式，然后改 eval 的手写 /v1/completions 流程。第 4 点是空的，我先不阻塞，按已有三点完成。




Read a file and searched code


训练侧没有在 CoAgenticRetriever agent loop 里显式配置 stop 字符串；它把整段 output.token_ids 加入轨迹，然后靠解析 <tool_call>/<answer> 和 turn/response length 终止。eval 现在显式 stop=["<|im_end|>"]，这不完全一致，我会把它改成可配置且默认不传 stop，以训练为准。




我会优先复用训练侧同一套 tool 初始化逻辑来拿 tool schema，而不是在 eval 里手写一份 schema；这样后续工具描述或参数变动时不会再次分叉。接下来改 evaluator 增加 --tool-config-path，eval shell 生成和训练一致的 tool config，并在 prompt/tool-response 渲染时传入 tools=tool_schemas。




Read 3 files and searched code


我已经确认训练路径的关键行为：首轮和工具返回后的下一轮都通过同一个 apply_chat_template(..., tools=tool_schemas, add_generation_prompt=True, enable_thinking=False) 渲染；生成后先追加原始输出，再在发现有效 tool call 时截断到第一个 </tool_call>。我现在按这个状态机修 eval，而不是只修首轮 prompt。




Read 2 files and searched code


我会避免为了拿 tool schema 而初始化 ranker/GPU。训练配置中的 tool schema 与 ranker_enabled 无关，所以 eval 侧会从同一份 tool config 读取 class/config，再用无副作用路径生成同一份 OpenAI tool schema，并在渲染 prompt 时传入 tokenizer。




Read 2 files and searched code


我会把 eval 的对齐范围限定在轨迹生成链路本身：prompt/tool template、tool response 拼接、tool_call 截断和 stop 配置。eval 的评分方式和服务部署参数不会在这一步强行混进来。




Read a file and searched code


我发现 eval 自己的 tool response 格式函数和训练工具用的是同名但不同位置的实现，需要确认内容是否等价；如果不等价，也要让 eval 使用训练工具的格式。




Read 3 files


多轮逻辑还有一个细节：训练在达到 max_assistant_turns/max_user_turns/response_length 时直接终止，不再额外解析最后一轮答案；eval 之前会在最后一轮继续抽取 answer。为了和训练更接近，我会按训练顺序判断：先追加输出，再检查长度/turn limit，再解析 tool/answer。




Read 2 files


我现在开始改代码：evaluate_coagentic_vllm.py 会新增 tool schema 加载和 train-style chat template 拼接，02_infer...sh 会生成并传入 eval 用的 tool config。改完会做静态语法和一个 tokenizer 级别的一致性检查。




Read 2 files


我会做一个小的实现选择：tool schema 的来源优先读 eval 生成的 tool config；如果 config 内没有显式 tool_schema，对 CoAgenticRetrieverTool 用和训练类相同的默认 schema，但实例化时临时关闭 ranker，避免 eval 只为 schema 加载 ranker/GPU。这样 prompt 对齐，同时不引入额外服务依赖。




Read 2 files and searched code


我会把 STOP_SEQUENCES 默认设为空，表示 eval 不再显式传 stop=["<|im_end|>"]。如果以后要恢复旧行为，可以外部传 STOP_SEQUENCES='<|im_end|>'，但默认以训练的无显式 stop 为准。




Edited a file


第一段改动已经加入：eval 现在具备 tool schema 读取、模板传 tools、以及可配置 stop 的能力。下一步我会替换 evaluate_one 的多轮状态机，让 prompt 拼接顺序和训练 agent loop 一致。




Edited a file


我会修正一个实现细节：训练侧 response_length 统计包括工具返回 token，eval 也应该维护同样的累计长度，而不是只看 assistant 文本长度。现在补上这个计数，同时保留 eval 对最终 <answer> 的提取，避免二轮回答被误判为空。




Edited a file


我还会补一个对齐项：eval shell 以前没有生成训练侧那种 .coagentic_tool_config.yaml，所以 evaluator 没有统一的 schema 来源。现在会由 eval shell 在 runtime log 目录下生成同格式 tool config，并传给 evaluator。




刚才补 argparse 的 patch 命中了旧上下文，我先读一下文件底部的实际参数定义，再按当前位置补进去。




Read a file and searched code


我继续补 argparse 和 EvalArgs 字段。然后会检查是否所有新增字段都有 shell 传入和 run_config 落盘。




Read a file


我注意到 evaluator 的 argparse 默认 --enable-thinking 还是 True；虽然 shell 会传 --no-enable-thinking，但默认值本身与 base eval 脚本不一致。我会顺手改成 False，避免直接调用 evaluator 时踩坑。




Edited a file


现在改 shell：加入 eval tool config 的生成、env/report 记录、传参，以及 STOP_SEQUENCES 的默认空值。这里会仿照训练脚本生成同结构 YAML。




Edited a file


Worked for 10m 21s




我需要处理 no-ranker 模式的 tool config：训练 no-ranker/禁用 ranker 时 tool schema 仍一致，但配置里的 ranker_enabled 不能保持 true。现在 shell 需要根据 RUN_MODE=no-ranker 自动把 eval tool config 的 ranker_enabled 写 false。

## assistant continued:

已补完未完成项，并做了对齐校验：

- `02_infer_qwen3_4b_ablation_val_only.sh`
  - 增加 `RUN_MODE=full` 默认值，避免直接运行时 `set -u` 下未定义。
  - `RUN_MODE=no-ranker` 时强制 `COAGENTIC_RANKER_ENABLED=false`，生成的 eval tool config 会写 `ranker_enabled: false`。
  - `RANKER_TOP_K/RANK_TOP_K` 默认值改为跟训练一致的 `TOP_M`，默认生成 `top_k: 5`，不再默认 50。
  - `STOP_SEQUENCES` 默认仍为空，eval 不再显式传 `stop=["<|im_end|>"]`，以训练侧无显式 stop 的行为为准。

- `evaluate_coagentic_vllm.py`
  - tool response 追加后只把去掉 system prompt 前缀的 token 计入 `response_token_count`，和训练侧 `response_mask` 计数一致。
  - 合法 tool call 后按训练侧 `_truncate_after_first_tool_call` 的 token 级方式截到第一个 `</tool_call>`，并同步扣减被截断 token。
  - 继续通过 `/v1/completions` 手写多轮逻辑，但 prompt/tool schema/tool response 拼接走和训练相同的 `apply_chat_template(..., tools=..., add_generation_prompt=True, enable_thinking=False)`。

验证结果：

- `bash -n scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh` 通过。
- `/data04/envs/ms/ms_cosearch_official/bin/python -m py_compile scripts/coagenticRetriever_local/evaluate_coagentic_vllm.py` 通过。
- `DRY_RUN=1 RUN_MODE=no-ranker` 生成 tool config：`ranker_enabled: false`、`top_k: 5`。
- `DRY_RUN=1 RUN_MODE=full` 生成 tool config：`ranker_enabled: true`、`top_k: 5`。
- tokenizer 级对齐检查通过：eval 首轮 prompt、system prompt、tool response 模板与训练侧直接调用 tokenizer 的结果逐 token 相等；渲染后含 `<tools>`，且 `enable_thinking=false` 时包含 Qwen no-think stub。

## user:

抛弃动态策略，使用静态 `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever/config/coagentic_retriever_tool_config.yaml`，且任何冲突参数，均以静态配置为准；立即修改并实际验证改动成功；修改完之后不要保留任何动态策略的痕迹。

## assistant:

已完成静态 tool config 单一来源改造：

- 训练入口链路：
  - `scripts/coagenticRetriever_local/assets/00_run_agentic_iter_rag_verl.sh`
  - `scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh`
  - `tasks/train_tasks/train_CAR_async_labeling_ds_flash.sh`
- eval 入口：
  - `scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh`
- legacy 训练脚本：
  - `CoAgenticRetriever/scripts/train_coagentic_retriever_grpo.sh`

现在不再生成 `${LOG_DIR}/${EXP_NAME}.coagentic_tool_config.yaml` 或 `${RUN_NAME}.coagentic_tool_config.yaml`。所有入口都固定使用：

```bash
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever/config/coagentic_retriever_tool_config.yaml
```

并从该静态 YAML 反向覆盖脚本中的冲突参数，包括：

- `RETRIEVAL_SERVICE_URL`
- `TOP_N` / `RECALL_TOP_K`
- `TOP_M`
- `RANK_TOP_K` / `RANKER_TOP_K`
- `RANKER_BASE_MODEL_PATH` / `RANKER_MODEL`
- `RANKER_ENCODER_PATH`
- `RANKER_CONFIG_DEVICE` / `RANKER_DEVICE`
- `RANKER_MAX_QUERY_LENGTH`
- `RANKER_MAX_DOC_LENGTH`
- retry 参数

静态 YAML 补充了 `max_concurrent_per_worker: 4`，使并发也由静态配置显式控制。

验证：

- `bash -n` 通过：
  - `scripts/coagenticRetriever_local/assets/00_run_agentic_iter_rag_verl.sh`
  - `scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh`
  - `scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh`
  - `CoAgenticRetriever/scripts/train_coagentic_retriever_grpo.sh`
- `python -m py_compile scripts/coagenticRetriever_local/evaluate_coagentic_vllm.py` 通过。
- 训练 dry-run：
  - `TOOL_CONFIG=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever/config/coagentic_retriever_tool_config.yaml`
  - `RANK_TOP_K=50`
  - `TOP_N=50`
  - `TOP_M=5`
  - `RANKER_CONFIG_DEVICE=cuda:4`
  - dry-run 目录下未生成任何 `*coagentic_tool_config.yaml`。
- eval dry-run：
  - `TOOL_CONFIG=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever/config/coagentic_retriever_tool_config.yaml`
  - `RANKER_TOP_K=50`
  - `RANKER_MAX_QUERY_LENGTH=256`
  - `RANKER_MAX_DOC_LENGTH=512`
  - dry-run 目录下未生成任何 `*coagentic_tool_config.yaml`。
- 搜索当前入口和 CoAgenticRetriever scripts：
  - 无 `cat > "${TOOL_CONFIG}"`
  - 无 `write_tool_config`
  - 无 `Generated tool config`
  - 无 `/tmp/coagentic_retriever_tool_config`
  - 无新的 `*.coagentic_tool_config.yaml` 动态输出逻辑。

## 2026-06-16 16:06 补充：eval 继续按 train 轨迹生成逻辑对齐

在静态 tool config 单一来源改造之后，继续检查 eval 与训练 agent loop 的真实分叉，补齐以下对齐项：

- eval 继续走 `/v1/completions` 手写多轮逻辑，但生成预算改为训练同款：
  - 首轮 prompt 渲染后只按 `MAX_PROMPT_LENGTH=4096` 做一次 left truncation，对齐训练 `RLHFDataset` 的 `data.truncation=left`。
  - 首轮之后不再每轮按 `MAX_PROMPT_LENGTH` 重新截断 prompt；后续 prompt 直接沿用 token 流追加 assistant/tool response，对齐 `CoAgenticRetrieverAgentLoop`。
  - 每次 `/v1/completions` 的 `max_tokens` 改为 `MAX_MODEL_LEN - len(prompt_ids)`，对齐训练 `vllm_async_server.py` 的 `max_tokens = config.max_model_len - len(prompt_ids)`，不再固定使用 `MAX_RESPONSE_LENGTH`。
  - 如果模型单轮输出超过剩余 `MAX_RESPONSE_LENGTH`，eval 只用训练最终可见的截断 token 提取 answer/status，避免把训练会被截掉的尾部答案算进去。

- eval tool response 文本改为优先复用训练实现：
  - `verl.tools.utils.search.format_tool_response`
  - 若直接调用 evaluator 且未设置 `PYTHONPATH`，才回退到本地等价实现。

- eval tool call 解析继续按 SearchR1 parser 语义：
  - JSON 非法或 tool name 非 `search` 时跳过。
  - `arguments` 非 dict 时按训练 parser 置空 dict，而不是直接丢弃。
  - 空 query 也会进入工具执行分支，并返回训练工具同款 `Error: No query provided` tool response，而不是提前判为无 valid tool。

- shell/evaluator 补充 `MAX_MODEL_LEN`：
  - `02_infer_qwen3_4b_ablation_val_only.sh` 将 `MAX_MODEL_LEN=8192` 传入 evaluator。
  - env/report/dry-run 输出均记录 `MAX_MODEL_LEN`。

验证：

- `bash -n scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh` 通过。
- `/data04/envs/ms/ms_cosearch_official/bin/python -m py_compile scripts/coagenticRetriever_local/evaluate_coagentic_vllm.py` 通过。
- `DRY_RUN=1 STRATEGY_NAME=align_check RUN_MODE=full bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh` 通过，解析出的静态配置为：
  - `TOOL_CONFIG=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever/config/coagentic_retriever_tool_config.yaml`
  - `TOP_N=50`
  - `TOP_M=5`
  - `RANKER_TOP_K=50`
  - `RANKER_MAX_QUERY_LENGTH=256`
  - `RANKER_MAX_DOC_LENGTH=512`
  - `MAX_MODEL_LEN=8192`
  - `MAX_PROMPT_LENGTH=4096`
  - `MAX_RESPONSE_LENGTH=4096`
  - `MAX_TOOL_RESPONSE_LENGTH=2048`
  - `STOP_SEQUENCES=`，即默认不传 stop，对齐训练无显式 stop。
- tokenizer 级一致性检查通过：
  - eval 首轮 prompt 与训练直接调用 `tokenizer.apply_chat_template(..., tools=tool_schemas, add_generation_prompt=True, enable_thinking=False)` 逐 token 相等。
  - system prompt 逐 token 相等。
  - tool response 模板和去 system prompt 后的 delta 逐 token 相等。
  - 渲染后含 `<tools>`，且含 Qwen no-think stub `<think>\n\n</think>`。
  - 当前抽样首轮 prompt token 长度为 396，因此训练式首轮 `max_tokens` 为 `8192 - 396 = 7796`，不是固定 4096。

## 2026-06-16 16:24 补充：静态配置真实 one-step 训练验证

训练 one-step 前发现并修正了一个静态配置切换后暴露出的预检参数错误：

- 静态 YAML 中 `ranker.top_k=50` 后，训练脚本曾把 `RANK_TOP_K` 传给 retrieval preflight 的 `--top-m`。
- reward helper 只支持 `top_m <= 5`，因此错误的 `--top-m 50` 会触发 `AssertionError: Only support up to 5 retrieved documents`。
- 已改为 preflight 使用 `TOP_M`，即 `--top-n 50 --top-m 5`；`ranker.top_k` 仍由静态 YAML 保持为 50。

真实训练验证命令：

```bash
TOTAL_STEPS=1 bash tasks/train_tasks/train_CAR_async_labeling_ds_flash.sh
```

本次 run：

- run dir: `log/train_logs/coAgenticRetriever/260616-161547-CAR_async_labeling_ds_flash_v1`
- checkpoint: `checkpoints/qwen3_4b_probe/coAgenticRetriever/260616-161547-CAR_async_labeling_ds_flash_v1/global_step_1`

训练入口实际加载的静态 tool config：

- `TOOL_CONFIG=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever/config/coagentic_retriever_tool_config.yaml`
- `default_top_n=50`
- `default_top_m=5`
- `ranker.top_k=50`
- `ranker.max_query_length=256`
- `ranker.max_doc_length=512`
- `ranker.device=cuda:4`

retrieval preflight 结果：

- `top_n=50`
- `top_m=5`
- `num_recall_docs=50`
- `num_ranked_docs=50`

真实 rollout/训练结果：

- rollout 完成：`trajectories=512/512`
- actor update 完成：`micro_batch=32/32`
- step 指标中 `ranker/trace_enriched_tool_calls=510`
- step 指标中 `ranker/trace_ranked_docs=25500`，即 `510 * 50`，确认 ranker 排序后的 top50 序列被保留进入轨迹信号，不是只保留 top5。
- `async_labeling/candidate_tool_calls=510`
- `async_labeling/selected_tool_calls=10`
- `async_labeling/invalid_requests=0`
- `training/global_step=1`

收尾验证：

- 训练脚本返回码 0。
- checkpoint conversion 完成。
- actor HF safetensors validation 通过：
  - `checkpoints/qwen3_4b_probe/coAgenticRetriever/260616-161547-CAR_async_labeling_ds_flash_v1/global_step_1/hf_safetensors/actor`
- 训练结束后 `nvidia-smi` 显示 `No running processes found`，无残留 GPU 进程。

## 2026-06-16 16:35 补充：preflight fail-fast 修正

发现原训练/eval 入口的 recall preflight 策略有设计问题：

- 旧逻辑把完整工具语义验证 `00_check_coagentic_tool_retrieval.py` 同时当作 readiness probe 使用。
- 语义验证失败、参数错误、reward 断言失败、服务未启动，都会被折叠成同一个非零返回。
- 等待循环只看到非零返回，就继续重试，导致 `TOP_M/RANK_TOP_K` 这类确定性配置错误被表现成“卡住/等待服务 ready”。

已修正为两阶段：

- `check_recall_http_ready`：只做轻量 `/retrieve` HTTP ready 检查，失败时可以等待服务启动。
- `run_recall_preflight`：服务 ready 后只跑一次语义验证；失败立即打印错误并 `exit 2`，不再重试 readiness。

同时新增本地参数校验：

- `RECALL_TOP_K` 必须为正整数。
- `TOP_M` 必须为正整数。
- `TOP_M <= RECALL_TOP_K`。
- 当前 reward preflight 仍受 `answer_match_reward` 限制，`TOP_M > 5` 会在启动任何 GPU 服务前直接报错：
  - `TOP_M is agent-visible docs. Do not pass RANK_TOP_K/ranker.top_k here.`

同步修改范围：

- `scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh`
- `scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh`
- `scripts/coagenticRetriever_local/assets/00_check_coagentic_tool_retrieval.py`

额外清理策略：

- 训练入口的 `cleanup_background_tasks` trap 前移到启动 judge/retriever 等后台服务之前。
- 如果 LLM judge 已启动，而 recall semantic preflight 后续失败，也会触发 cleanup，不再遗留 GPU 服务。

验证：

- `bash -n scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh` 通过。
- `bash -n scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh` 通过。
- `/data04/envs/ms/ms_cosearch_official/bin/python -m py_compile scripts/coagenticRetriever_local/assets/00_check_coagentic_tool_retrieval.py` 通过。
- 直接错误用例：

```bash
/data04/envs/ms/ms_cosearch_official/bin/python scripts/coagenticRetriever_local/assets/00_check_coagentic_tool_retrieval.py \
  --project-root CoAgenticRetriever \
  --url http://127.0.0.1:8030/retrieve \
  --query 'who got the first nobel prize in physics?' \
  --top-n 50 \
  --top-m 50 \
  --expect-contains ''
```

输出为：

```text
ERROR: --top-m exceeds current reward preflight limit of 5 visible documents; use agent-visible TOP_M here, not ranker.top_k/RANK_TOP_K.
```

并立即返回非零状态，不访问 retrieval 服务。

- `DRY_RUN=1 TOTAL_STEPS=1 bash tasks/train_tasks/train_CAR_async_labeling_ds_flash.sh` 通过。
- `DRY_RUN=1 bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh` 通过。

## 2026-06-16 16:39 补充：RUN_MODE full / no-ranker 保留

前面的静态 tool config 改造里，eval shell 曾把 `RUN_MODE=no-ranker` 反向改写成 `full`，原因是把静态 YAML 中的 `ranker_enabled: true` 当成了运行模式真值。

已修正为：

- `RUN_MODE` 继续由调用方显式控制，`full` 与 `no-ranker` 都保留。
- 静态 tool config 仍只负责提供统一的参数来源：
  - `TOP_N`
  - `TOP_M`
  - `RANKER_MODEL`
  - `RANKER_ENCODER_PATH`
  - `RANKER_DEVICE`
  - `RANKER_MAX_QUERY_LENGTH`
  - `RANKER_MAX_DOC_LENGTH`
- `no-ranker` 仅在 eval shell 内部把 `COAGENTIC_RANKER_ENABLED=false`，并让 evaluator 按 `--run-mode no-ranker` 走禁用 ranker 的路径。

验证：

- `DRY_RUN=1 RUN_MODE=full bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh` 输出 `RUN_MODE=full`。
- `DRY_RUN=1 RUN_MODE=no-ranker bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh` 输出 `RUN_MODE=no-ranker`。

## 2026-06-16 16:58 补充：真实 GPU eval 对齐验证

验证对象：

- 最近完整训练日志：`log/train_logs/coAgenticRetriever/260616-161547-CAR_async_labeling_ds_flash_v1`
- 更晚的 `260616-163504-CAR_async_labeling_ds_flash_v1` 只有 env 文件，无 rollout/metrics，未作为对照。
- 验证时 `260616-161547` 对应的 `global_step_1` checkpoint 目录已不在当前 checkpoint tree 中，因此 eval probe 使用训练 env 记录的初始模型：
  - agent model: `/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B`
  - ranker/retriever model: `/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2`
  - 这与该次训练 step=1 rollout 生成阶段的起始模型/未异步更新 ranker 状态一致；训练 metrics 中 `ranker/async_updates=0`。

真实 eval 命令参数：

- `RUN_MODE=full`
- `MAX_EVAL_NUM=1`
- `EVAL_BATCH_SIZE=1`
- `MAX_NUM_SEQS=1`
- `KEEP_TRACE=full`
- `AGENT_GPU_IDS=0`
- `RANK_GPU_ID=1`
- `RECALL_GPU_ID=2`

输出目录：

- trace dir: `log/eval_res/coAgenticRetriever/260616-1652-codex_real_eval_align_probe`
- report: `reports/eval/coAgenticRetriever/260616-1652-codex_real_eval_align_probe.report.md`
- llm io: `log/eval_res/coAgenticRetriever/260616-1652-codex_real_eval_align_probe/runtime_logs/codex_real_eval_align_probe.llm_io.jsonl`

本次真实启动先暴露了一个服务端 bug：

- `src/retrievers/gpu_dense_retriever_server.py` 在 `/retrieve` 收到 `return_scores=false` 时，`batch_search(..., return_score=False)` 只返回 `results`，但 endpoint 仍按 `(results, scores)` 解包，导致 HTTP 500。
- 已修正 endpoint：`return_scores=true` 时解包 `(results, scores)`；`return_scores=false` 时只取 `results`。
- 同时增强 eval shell readiness：如果 `/retrieve` 返回 HTTP 5xx，立即打印 fatal readiness error 和日志尾部并 `exit 2`，不再作为“未 ready”继续等待。

修复后检查：

- `bash -n scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh` 通过。
- `/data04/envs/ms/ms_cosearch_official/bin/python -m py_compile src/retrievers/gpu_dense_retriever_server.py scripts/coagenticRetriever_local/evaluate_coagentic_vllm.py scripts/coagenticRetriever_local/assets/00_check_coagentic_tool_retrieval.py` 通过。

真实 eval 结果：

- recall semantic preflight 通过：`top_n=50 top_m=5`
- eval 成功退出，`summary.failure_count=0`
- `run_config.json`：
  - `run_mode=full`
  - `tool_config_path=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever/config/coagentic_retriever_tool_config.yaml`
  - `top_n=50`
  - `top_m=5`
  - `ranker_top_k=50`
  - `max_model_len=8192`
  - `max_prompt_length=4096`
  - `stop_sequences=None`
- `summary.json` / `traces.jsonl`：
  - `num_recall_docs=50`
  - `num_ranked_docs=50`
  - `num_agent_visible_docs=5`
  - `ranker_enabled=true`
  - `tool_calls=1`
  - `status=answered`
  - `retrieved_top50_chunks[0]` 长度为 50
  - `ranked_top5_chunks[0]` 长度为 5

prompt / 轨迹对齐检查：

- eval 首轮 `prompt_token_count=396`。
- train `llm_io` 首条 `prompt_token_count=396`。
- eval 首轮 `sampling_params.max_tokens=7796`，即 `8192 - 396`。
- eval 首轮 `sampling_params.stop=null`，不再显式传 `<|im_end|>` stop。
- eval 第二轮 `prompt_token_count=950`，`sampling_params.max_tokens=7242`，即 `8192 - 950`。
- eval 首轮 prompt 与 train 首轮 prompt 的问题文本不同；将 `Question: ...` 归一化后：
  - normalized prompt 完全相等。
  - `Question:` 前的 tool schema/system/user 模板前缀完全相等。
  - `<|im_start|>assistant\n<think>\n\n</think>\n\n` 后缀完全相等。
- eval 第二轮 prompt 含：
  - `<tool_response>...</tool_response>`
  - `<|im_start|>assistant`
  - `<think>\n\n</think>\n\n`
- train `rollout_data/main_agent/1.jsonl` 首条输出结构为：
  - assistant `<tool_call>`
  - user `<tool_response>`
  - assistant `<think>\n\n</think>`
  - final `<answer>`
- eval `llm_io` 两轮结构为：
  - record 0：输出 `<tool_call>`，无 `<answer>`
  - record 1：prompt 含 `<tool_response>`，输出 `<answer>`，无第二次 `<tool_call>`

训练侧对照证据：

- train metrics：
  - `ranker/enabled=1`
  - `ranker/trace_enriched_tool_calls=510`
  - `ranker/trace_ranked_docs=25500`
  - `25500 / 510 = 50.0`
- 因此训练侧仍是 ranker 对 recall top50 排序并保留 top50 轨迹信号；agent 可见 top5。
- eval 真实 trace 与该语义一致：ranker 排序 50，agent response 只注入 5。

清理状态：

- eval 退出后 `nvidia-smi` 显示 `No running processes found`。
- `ss -ltnp sport = :8030` 无监听。
- `ss -ltnp sport = :8040` 无监听。

结论：

- full-mode eval 的 prompt 构造、tool schema、no-think stub、stop 策略、`max_tokens` 策略、tool response 拼接、ranker top50/top5 语义已经按 train 对齐，并已通过 1 条真实 GPU eval 轨迹验证。
