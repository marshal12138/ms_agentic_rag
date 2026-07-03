# AIR Reranker GRPO Trainer 模块详细设计

更新日期：2026-07-03

## 1. 目标

这篇文档讲 AIR 专用 LLM reranker GRPO trainer 怎么组织代码。

核心目标：

- 复用 VERL 的训练能力。
- 但 AIR 自己控制 branch dataset、continuation rollout、reward 和 manifest。
- 不直接照搬旧 `search_r1_reranker_reward_agent_loop_worker` 的数据假设。

第一版 trainer 只训练 reranker，search agent 冻结。

## 2. 非目标

第一版 trainer 不做：

- 不训练 search agent。
- 不做 agent/reranker 交替训练。
- 不支持 all-steps reranker 训练。
- 不实现新的分布式训练框架。
- 不在 shell 里维护业务配置。

## 3. 包结构

建议新增：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/
  __init__.py
  branch_dataset.py
  parser.py
  reward.py
  continuation_rollout.py
  trainer_entry.py
  service_bundle.py
  schema.py
```

职责：

- `schema.py`：定义 branch sample、parse result、reward result、continuation result 的轻量 dataclass。
- `branch_dataset.py`：从增强轨迹构造 branch dataset。
- `parser.py`：解析 reranker 输出。
- `reward.py`：计算 reranker reward。
- `continuation_rollout.py`：执行 frozen agent continuation。
- `trainer_entry.py`：训练入口，读取 AIR final config，启动训练。
- `service_bundle.py`：训练完成后生成 service bundle。

代码注释要求：

- 每个模块文件顶部写中文 docstring，说明模块职责。
- 复杂函数前写中文注释，说明输入输出和失败策略。
- 不要写“给变量赋值”这种低价值注释。

## 4. 训练入口

`main_train_llm_reranker.py` 从 placeholder 改成真实入口。

职责：

1. 解析 `--config` 和 `--manifest`。
2. 读取 AIR final config。
3. 校验 reranker_training 配置。
4. 如果需要，调用 branch dataset builder。
5. 启动 AIR reranker GRPO trainer。
6. 写 stage manifest。
7. 训练完成后把 reranker model path 写回 final config。

内部可以委托：

```text
agentic_iter_rag.reranker_training.trainer_entry.run_from_config()
```

## 5. Trainer Entry 设计

`trainer_entry.py` 核心接口：

```text
run_from_config(config_path: Path, stage_manifest: Path, dry_run: bool = False) -> dict
```

执行流程：

1. 加载 final config。
2. 解析 `reranker_training`。
3. 确定 branch dataset manifest。
4. 确定 reranker output dir。
5. 生成 VERL 训练配置或 dotlist。
6. 启动训练进程。
7. 收集 checkpoint。
8. 写 manifest。

dry-run 时：

- 不启动训练。
- 写出将要使用的 dataset、base model、output dir、resource plan、VERL 命令摘要。

代码注释要求：

- 配置读取处写中文注释，说明所有业务参数来自 final config。
- dry-run 分支写中文注释，说明不会启动训练服务。
- 写 manifest 前写中文注释，说明下游 service bundle 依赖哪些输出。

## 6. VERL 对接边界

AIR 不重写底层优化器。

复用 VERL：

- actor rollout/ref worker。
- logprob 计算。
- GRPO advantage。
- checkpoint 保存。

AIR 自己负责：

- branch dataset schema。
- reranker 输出 parser。
- continuation rollout。
- reward extra info。
- stage manifest。
- service bundle。

需要新增或适配一个 AIR reranker reward manager。它可以参考现有 `RerankerRewardManager`，但 reward 来源不同：

- 不是静态 qrels。
- 是 continuation answer reward。

## 7. UID Grouping

默认 UID：

```text
trajectory_id:step_index
```

同一个 UID 下：

- prompt 完全相同。
- reranker 采样多个 response。
- 每个 response 独立 continuation。
- 得到多个 reward。

trainer 要把 UID 写入 batch extra info，供 GRPO 分组。

如果某个 group reward 全一样：

- 第一版沿用 VERL 现有过滤或 fallback 策略。
- 过滤统计写入 metrics。

## 8. Batch 构造

branch dataset 每条样本进入 VERL 后：

- `prompt` 是 reranker prompt。
- `response` 由 reranker rollout 生成。
- `reward` 由 AIR reward runner 计算。
- `extra_info` 保留 continuation 需要的上下文。

需要注意：

- `messages_before_tool_response` 可能很大，要确认 DataProto non_tensor_batch 能承载。
- 如果太大，后续可以改成 `context_ref` 引用外部 JSONL，但第一版先直接内嵌，保证实现简单。

## 9. Checkpoint 和 Manifest

训练 stage manifest 推荐包含：

```json
{
  "stage": "train_llm_reranker",
  "status": "completed",
  "base_model": ".../Qwen3-4B",
  "branch_dataset_manifest": "...",
  "output_dir": "...",
  "reranker_model": "...",
  "trainer_method": "grpo",
  "reward_strategy": "answer_reward",
  "global_steps": 100,
  "service_bundle_ready": false
}
```

如果训练失败：

- `status=failed`
- 写入 error type 和 message。
- 不写 fake reranker_model。

## 10. 训练入口 Shell

新增 shell 入口时要保留 AIR 注释风格。

必须写清楚：

- 这是 reranker 训练入口。
- 它只选择配置组。
- 业务参数写在 YAML。
- CLI dotlist 只用于临时覆盖。
- 不允许 shell-only 业务配置。

## 11. Runner 集成

`run_pipeline.py` 增加：

- `build_reranker_branch_dataset`
- `train_llm_reranker`
- `build_service_bundle`

`train_llm_reranker` 不再写 placeholder manifest，而是调用 `main_train_llm_reranker.py` 或 `trainer_entry.run_from_config`。

dry-run 仍然只写 manifest。

## 12. 错误处理

直接失败：

- branch dataset manifest 缺失。
- base model 路径不存在。
- frozen agent model 缺失。
- reward function 加载失败。
- VERL 训练进程非零退出。
- 没找到训练后 reranker checkpoint。

不应该吞掉这些错误，因为下游 service bundle 必须依赖真实 checkpoint。

## 13. 实现计划

顺序：

1. 新增 `reranker_training` 包和 schema。
2. 实现 branch dataset builder。
3. 实现 parser 和 reward。
4. 实现 continuation runner。
5. 实现 trainer entry dry-run。
6. 接入真实 VERL 训练。
7. 接入 pipeline runner。
8. 写训练 stage manifest。

中文注释要求：

- 新增 Python 模块必须有中文模块说明。
- 训练入口和 runner stage 分发处必须写中文注释。
- VERL 对接边界必须注释清楚，避免后续误以为 AIR 重写了训练框架。
- shell 入口必须解释配置来源和禁止 shell-only 业务配置的原因。

## 14. 测试计划

### 14.1 dry-run 测试

运行训练 task `--dry-run`。

期望：

- stage manifest 写出 base model、dataset manifest、output dir。
- 不启动 VERL。

### 14.2 branch dataset 缺失测试

不提供 branch dataset manifest。

期望：

- 如果构造 stage 没执行，train stage 失败。
- 错误信息指向 `reranker_training.input.branch_dataset_manifest`。

### 14.3 reward mock 训练测试

用 2 条 fake branch sample。

mock continuation 和 reward。

期望：

- 能构造 reranker batch。
- reward 写到最后有效 token。

### 14.4 checkpoint 测试

训练完成后：

- reranker checkpoint 路径存在。
- manifest 中 `reranker_model` 指向真实路径。

### 14.5 注释验收

人工检查：

- 新增 Python 模块中文注释充足。
- shell 入口中文注释充足。
- 配置读取和 manifest 写入处有中文说明。
