# 当前 reward：`spad_em_teacher_backoff` 梳理和使用

日期：2026-07-11；北京时间09点

## 1. 结论

当前 SPAD Stage1 组级 reward 正式命名为：

```text
spad_em_teacher_backoff
```

名称含义是：先对同一 UID 的整组 rollout 计算 actor 完整答案 EM；只有整组 EM 全零时，
才调用 teacher 判断各条轨迹的检索证据状态，并以较小权重提供回退奖励。

稳定训练入口为：

```text
compute_spad_em_teacher_backoff_batch
```

0710 reward `spad_teacher_f1_0710` 仍在独立模块中，二者的代码入口和运行契约互不混用。

## 2. 奖励公式

设同一问题 UID 有 `n=8` 条 rollout，第 `i` 条 actor 完整答案对任一 gold alias 的 exact match 为
`em_i`。默认 teacher 部分奖励系数 `alpha=0.1`。

```text
如果组内存在任意 em_i = 1：
    reward_i = em_i
    整组不调用 teacher

如果整组 em_i 全为 0：
    teacher_status_i = 1，条件为：
        该条 rollout 有检索证据；
        teacher XML 格式合法；
        status 为 supported_answer 或 ambiguous_evidence。
    其他情况 teacher_status_i = 0。

    reward_i = alpha * teacher_status_i
```

因此默认奖励取值为 `0`、`0.1` 或 `1.0`。teacher 判断的是“现有证据是否足以支持回答”，
不再生成答案后与 gold 计算 F1。

以下项目只记录审计值，不参与当前公式：搜索次数、第二次及以后搜索成本、重复 query、
teacher answer F1。输出字段 `teacher_f1` 为兼容 VERL 既有指标而保留，在本 reward 下实际写入的是
最终 reward，不应解释为 teacher 答案 F1。

## 3. 完整答案与多 Gold 语义

actor 必须输出闭合的 `<answer>...</answer>`；缺少闭标签或答案为空时 `em_i=0`。当前 SPAD
Stage1 的 EM 是“预测与任一 gold alias 完全匹配即为 1”。新数据中的 gold 列表直接参与该判定，
不与旧 Search-R1 或旧 SPAD 数据做比较。

teacher status 固定为三个枚举值：

| status | teacher status reward |
| --- | ---: |
| `supported_answer` | 1 |
| `ambiguous_evidence` | 1 |
| `insufficient_evidence` | 0 |
| XML/请求错误或无检索证据 | 0 |

`ambiguous_evidence` 获得回退奖励是当前实现的明确语义：它表示检索轨迹包含可回答信息，但支持了
多个不兼容答案；该奖励鼓励检索有效性，不把它当作 gold 正确性奖励。

## 4. 代码与独立入口

当前 reward 模块：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward.py
```

该模块声明：

```python
REWARD_VERSION = "spad_em_teacher_backoff"
```

专用入口 `compute_spad_em_teacher_backoff_batch` 会校验 `reward_cfg.type`。如果误把其他 reward
类型传入该入口，会立即报错，不会静默使用错误公式。

历史通用入口 `compute_spad_search_policy_reward_batch` 继续保留，供已有运行脚本兼容；新生成的
训练计划一律指向专用入口。2026-07-11 已启动的 512 SPAD run 仍可能在其冻结的
`verl_command_plan.json` 中记录通用入口，但其 `reward_cfg.type=spad_em_teacher_backoff`，公式与
专用入口完全相同。

0710 reward 的独立模块和入口分别为：

```text
search_policy_teacher_reward_0710.py
compute_spad_teacher_f1_0710_details
```

## 5. UID 流式调度

当前 reward 必须使用 VERL `BatchRewardManager`，因为是否调用 teacher 取决于同一 UID 的全部
8 条 rollout。训练路由看到 `spad_em_teacher_backoff` 后会强制选择 `batch` manager，避免配置
遗漏时错误退回逐条 `naive` manager。

默认开启 `stream_group_reward=true`：同一 UID 的 8 条 rollout 完成后立即在所属 rollout worker
提交该组 reward，teacher 请求与其他 UID 的生成重叠，不等待整个训练 batch rollout 全部结束。
`stream_group_max_inflight=1` 限制每个 worker 同时处理的 UID 组数。

调度不改变奖励公式。相关审计字段包括：

- `stream_group_ready_offset_s`：UID 组完成生成的时间偏移。
- `stream_group_reward_wall_s`：该组 reward/teacher 墙钟耗时。
- `stream_group_finished_offset_s`：该组 reward 完成的时间偏移。
- `stream_group_worker_pid`：执行组 reward 的 worker PID。

## 6. 与 0710 reward 的边界

| 配置 | `spad_em_teacher_backoff` | `spad_teacher_f1_0710` |
| --- | --- | --- |
| actor 停止位置 | `</answer>` | `<answer>` |
| actor 答案 | 完整答案 | 不生成答案正文 |
| 判定粒度 | 同 UID 8 条成组 | 每条 rollout 独立 |
| 主奖励 | actor answer EM | teacher answer 对 gold 的 F1 |
| teacher 条件 | 仅 EM 全零组 | 合法且有证据的 rollout |
| teacher 作用 | 证据状态回退，默认最高 0.1 | 生成短答案并作为主奖励 |
| reward manager | `batch` | `naive` |
| UID 流式组奖励 | 默认开 | 关 |
| Python 入口 | `compute_spad_em_teacher_backoff_batch` | `compute_spad_teacher_f1_0710_details` |

## 7. 使用方法

正式配置只需选择名称：

```yaml
agent_training:
  reward:
    type: spad_em_teacher_backoff
    spad_em_teacher_backoff:
      partial_reward: 0.1
```

`spad_rag_base.yaml`、正式 SPAD overlay 和 512 scale overlay 当前都已选择该名称。启动时无需再
手工指定 Python 函数。正式运行前建议先加 `--dry-run`，并在 `verl_command_plan.json` 检查：

```text
reward_model.reward_manager=batch
+reward_model.use_reward_loop=False
+reward_model.stream_group_reward=True
custom_reward_function.name=compute_spad_em_teacher_backoff_batch
+actor_rollout_ref.rollout.stop=['</tool_call>','</answer>']
+custom_reward_function.reward_kwargs.reward_cfg.type=spad_em_teacher_backoff
```

切换到0710方案时，只把 `agent_training.reward.type` 设为 `spad_teacher_f1_0710`，或使用已冻结的
0710 overlay。不要手工交叉组合函数入口、manager、stop token 和 stream 开关。

## 8. 关键审计字段

每条 rollout 至少关注：

| 字段 | 含义 |
| --- | --- |
| `reward_type` | 必须为 `spad_em_teacher_backoff` |
| `actor_answer` | 从闭合 answer 标签解析出的 actor 答案 |
| `actor_answer_parse_status` | `parsed` 或具体解析失败原因 |
| `em_reward` | actor 答案 EM，0 或 1 |
| `group_uid` / `group_size` | 组标识与组大小，正式配置应为 8 |
| `group_all_em_zero` | 是否触发 teacher 回退分支 |
| `partial_reward_applied` | 是否采用 teacher 部分奖励公式 |
| `teacher_called` | 该 rollout 是否实际请求 teacher |
| `teacher_evidence_status` | teacher 的证据状态 |
| `teacher_status_reward` | teacher 状态二值奖励 |
| `score` | 最终送入 GRPO 的 reward |

## 9. 验证范围

测试覆盖以下契约：

- 正式名称自动选择专用 batch 入口、闭合 answer stop 和流式组奖励。
- 专用入口拒绝其他 reward type。
- 有正 EM 的 UID 组整组跳过 teacher，奖励分别为 actor EM。
- EM 全零组才调用 teacher，并按 `partial_reward=0.1` 回退。
- UID 组大小不是 8 时立即失败。
- 0710 名称仍选择独立模块、独立入口和 opening-answer stop。

训练运行时完整环境测试结果：`64/64` 通过。
