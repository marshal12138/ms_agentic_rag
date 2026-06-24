# CoSearch GRPO 训练入口配置体系说明

本文梳理以下训练入口涉及的参数、参数文件、加载顺序和边界：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoSearch/scripts/train_co_search_grpo.sh
```

该入口属于 `CoSearch_derevitives/CoSearch` 目录下的原始 CoSearch/verl 训练体系，不是 `scripts/cosearch_local/*` 那套本地封装脚本。本文只讨论这条入口实际触达的配置链路。

## 1. 总体结论

训练启动链路是：

```text
scripts/train_co_search_grpo.sh
  -> python main_co_search_ppo.py key=value key=value ...
    -> Hydra 加载 config/co_search_trainer.yaml
      -> co_search_trainer.yaml 通过 defaults 组合组件 YAML
      -> 命令行 Hydra override 覆盖默认配置
      -> 运行时额外读取 agent loop config 和 tool config
```

核心根配置不是 shell 脚本，而是：

```text
CoSearch_derevitives/CoSearch/config/co_search_trainer.yaml
```

shell 脚本负责准备运行环境、生成若干动态参数，然后以 Hydra dotlist 的形式覆盖根配置。

实际配置来源可以分为 5 层，优先级从低到高大致如下：

1. 组件 YAML 默认值：`config/actor/*`、`config/rollout/*`、`config/model/*` 等。
2. Hydra 根配置：`config/co_search_trainer.yaml` 中的 `_self_` 字段。
3. 脚本传给 `python main_co_search_ppo.py` 的 `key=value` / `+key=value` 覆盖项。
4. 运行时被路径引用的外挂配置：`co_search_agent_loop_config.yaml` 和 `/tmp/co_search_tool_config_*.yaml`。
5. 数据样本中的 `non_tensor_batch`、`tools_kwargs`、`reward_model`、`extra_info` 等运行时字段。

其中第 4 层不是 Hydra defaults 的一部分，而是在 Python 运行中由 agent loop / tool registry 手动读取。

## 2. 关键文件清单

| 文件或参数源 | 类型 | 谁读取 | 作用边界 |
|---|---|---|---|
| `scripts/train_co_search_grpo.sh` | shell 入口 | 用户 / Slurm / shell | 设置环境变量、资源、模型数据路径、超参，并拼 Hydra overrides |
| `main_co_search_ppo.py` | Python Hydra main | Python 进程 | 定义 Hydra 根配置 `config/co_search_trainer.yaml`，初始化 Ray、worker、dataset、trainer |
| `config/co_search_trainer.yaml` | Hydra 根 YAML | Hydra | CoSearch 训练配置树的根，组合 actor/data/ref/rollout/model/critic/reward 等组件 |
| `config/actor/dp_actor.yaml` | Hydra 组件 YAML | Hydra defaults | 主 agent actor 的 FSDP/PPO/optimizer 默认配置 |
| `config/actor/reranker_dp_actor.yaml` | Hydra 组件 YAML | Hydra defaults | reranker actor 的 FSDP/PPO/optimizer 默认配置 |
| `config/actor/actor.yaml` | Hydra 组件 YAML | `dp_actor.yaml` defaults | actor 抽象默认值，如 PPO batch、KL、optimizer、checkpoint |
| `config/data/legacy_data.yaml` | Hydra 组件 YAML | Hydra defaults | parquet 数据、prompt/response 长度、dataset 行为 |
| `config/model/hf_model.yaml` | Hydra 组件 YAML | Hydra defaults | 主 agent HuggingFace 模型、tokenizer、LoRA、gradient checkpointing 等 |
| `config/model/reranker_hf_model.yaml` | Hydra 组件 YAML | Hydra defaults | reranker HuggingFace 模型，结构与主 model 类似 |
| `config/rollout/rollout.yaml` | Hydra 组件 YAML | Hydra defaults | 主 agent rollout/vLLM/multi-turn/agent-loop 配置 |
| `config/rollout/reranker_rollout.yaml` | Hydra 组件 YAML | Hydra defaults | reranker rollout 配置 |
| `config/ref/dp_ref.yaml` | Hydra 组件 YAML | Hydra defaults | 主 agent reference policy 配置 |
| `config/ref/reranker_dp_ref.yaml` | Hydra 组件 YAML | Hydra defaults | reranker reference policy 配置 |
| `config/critic/dp_critic.yaml` | Hydra 组件 YAML | Hydra defaults | critic 配置；本入口中通过 `critic.enable=False` 关闭 |
| `config/reward_model/dp_reward_model.yaml` | Hydra 组件 YAML | Hydra defaults | reward model / reward manager 默认配置 |
| `config/algorithm/rollout_correction.yaml` | Hydra 组件 YAML | Hydra defaults | rollout correction 默认配置 |
| `config/co_search_agent_loop_config.yaml` | 运行时 YAML | `AgentLoopManager` | 注册 `co_search_agent` 到 `CoSearchAgentLoop` 类 |
| `/tmp/co_search_tool_config_${SLURM_JOB_ID:-local}.yaml` | 运行时 YAML，脚本动态生成 | `CoSearchAgentLoop` / tool registry | 定义 `CoSearchTool` 实例和检索服务参数 |
| `verl/verl/experimental/agent_loop/uid_group_functions.py` | Python 函数文件 | `load_custom_function` | reranker GRPO UID 分组函数 |
| `verl/verl/experimental/agent_loop/score_assign_functions.py` | Python 函数文件 | `load_custom_function` | reranker 分数归因函数 |
| `verl/verl/utils/reward_score/search_qa_f1_with_format_penalty.py` | Python 函数文件 | reward loop | 主 agent multiturn reward 计算函数 |

容易混淆但本入口没有直接使用的文件：

```text
config/search_r1_tools_config.yaml
config/search_r1_agent_loop_config.yaml
config/ppo_trainer.yaml
config/ppo_megatron_trainer.yaml
config/_generated_ppo_trainer.yaml
config/_generated_ppo_megatron_trainer.yaml
Search-R1/*
```

这些文件可能用于其它训练或示例入口，但不是 `scripts/train_co_search_grpo.sh` 的主配置来源。

## 3. 配置加载顺序

### 3.1 shell 层

入口脚本先设置 Slurm、conda、Python path、vLLM backend、网络接口、HF cache、模型路径、数据路径、训练超参和函数路径。

关键片段：

```bash
PROJECT_ROOT=/work/hzeng_umass_edu/ir-research/CoSearch
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/verl:${PYTHONPATH:-}"

CHECKPOINT_PATH="Qwen/Qwen2.5-7B-Instruct"
RERANKER_CHECKPOINT_PATH="Qwen/Qwen2.5-7B-Instruct"

TRAIN_DATA="['${PROJECT_ROOT}/data/co_search/nq_40_multihop_60_51K/cot/co_search_rl_51k.train.parquet']"
VAL_DATA="['${PROJECT_ROOT}/data/co_search/nq_40_multihop_60_51K/cot/co_search_26k.sample_eval.parquet']"
```

这部分不是 Hydra 配置文件，但它决定了后续 Hydra override 的值。

### 3.2 Hydra 根配置

`main_co_search_ppo.py` 中定义：

```python
@hydra.main(config_path="config", config_name="co_search_trainer", version_base=None)
def main(config):
    run_dual_agent_ppo(config)
```

因此 Hydra 首先读取：

```text
CoSearch_derevitives/CoSearch/config/co_search_trainer.yaml
```

### 3.3 defaults 展开

`co_search_trainer.yaml` 的 `defaults` 负责把组件 YAML 挂载到配置树不同 namespace：

```yaml
defaults:
  - actor@actor_rollout_ref.actor: dp_actor
  - actor@reranker_actor_rollout_ref.actor: reranker_dp_actor
  - data@data: legacy_data
  - ref@actor_rollout_ref.ref: dp_ref
  - ref@reranker_actor_rollout_ref.ref: reranker_dp_ref
  - rollout@actor_rollout_ref.rollout: rollout
  - rollout@reranker_actor_rollout_ref.rollout: reranker_rollout
  - model@actor_rollout_ref.model: hf_model
  - model@reranker_actor_rollout_ref.model: reranker_hf_model
  - critic@critic: dp_critic
  - reward_model@reward_model: dp_reward_model
  - algorithm@algorithm.rollout_correction: rollout_correction
  - _self_
```

这里的 `_self_` 表示 `co_search_trainer.yaml` 自己的字段会覆盖前面 defaults 引入的同名字段。

### 3.4 命令行 override

shell 脚本最后执行：

```bash
python main_co_search_ppo.py \
  algorithm.adv_estimator=grpo \
  data.train_files=${TRAIN_DATA} \
  ...
```

这些命令行参数优先级高于 YAML 默认值。

Hydra 语义：

- `key=value`：覆盖已有 key。
- `+key=value`：新增 key 或新增 dict 内部字段。

脚本里 `+custom_reward_function.reward_kwargs.format_penalty=-0.2` 使用 `+`，因为 `reward_kwargs` 默认是空 dict，里面没有 `format_penalty`。

### 3.5 运行时外挂配置

以下配置不是 Hydra defaults，而是在代码运行过程中通过路径读取：

```text
actor_rollout_ref.rollout.agent.agent_loop_config_path
actor_rollout_ref.rollout.multi_turn.tool_config_path
reranker_uid_group_function.path
reranker_score_assign_function.path
custom_reward_function.path
```

这类配置的边界是：Hydra 只保存路径和函数名，真正 import / instantiate 发生在业务代码中。

## 4. shell 入口参数分组

### 4.1 运行环境参数

脚本顶部 Slurm 参数：

```bash
#SBATCH --job-name=cosearch-grpo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=7-00:00:00
```

这些只影响 Slurm 调度，不进入 Hydra 配置树。

conda 和 Python path：

```bash
. /work/hzeng_umass_edu/miniconda3/etc/profile.d/conda.sh
conda activate search-llm
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/verl:${PYTHONPATH:-}"
```

vLLM / NCCL / GLOO / HF cache：

```bash
export VLLM_DISABLE_FLASHINFER=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-en0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-en0}"
export HF_HOME="/gypsum/work1/zamani/hzeng/.cache/huggingface/"
```

这些影响底层 runtime，不是 Hydra 参数。

### 4.2 模型和数据参数

```bash
CHECKPOINT_PATH="Qwen/Qwen2.5-7B-Instruct"
RERANKER_CHECKPOINT_PATH="Qwen/Qwen2.5-7B-Instruct"
TRAIN_DATA="[...]"
VAL_DATA="[...]"
PROJECT_NAME="co_search"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/co_search"
```

对应 Hydra override：

```bash
data.train_files=${TRAIN_DATA}
data.val_files=${VAL_DATA}
actor_rollout_ref.model.path=${CHECKPOINT_PATH}
reranker_actor_rollout_ref.model.path=${RERANKER_CHECKPOINT_PATH}
trainer.project_name="${PROJECT_NAME}"
trainer.default_local_dir="${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXP_NAME}"
```

### 4.3 动态 tool config

脚本通过 heredoc 生成：

```bash
TOOL_CONFIG="/tmp/co_search_tool_config_${SLURM_JOB_ID:-local}.yaml"
```

内容大致是：

```yaml
tools:
  - class_name: verl.tools.co_search_tool.CoSearchTool
    config:
      type: native
      retrieval_service_url: "http://localhost:8000/retrieve"
      timeout: 30
      max_retries: 3
      retry_delay: 1.0
      retry_backoff: 2.0
      default_top_n: 50
      default_top_m: 5
      hit_cutoffs: [1, 3, 5]
      tool_score_metric: "hit"
      trivial_answers: ["yes", "no", "true", "false"]
      format_penalty: -0.2
```

它通过以下 override 进入配置树：

```bash
actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG}
```

然后 `CoSearchAgentLoop` 运行时读取这个路径并初始化 tool。

### 4.4 硬件和 Ray 参数

```bash
NNODES="${SLURM_NNODES:-2}"
N_GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
TP_SIZE=1
```

对应 override：

```bash
trainer.nnodes=${NNODES}
trainer.n_gpus_per_node=${N_GPUS_PER_NODE}
actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}
reranker_actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}
```

Python 中 `DualAgentTaskRunner.init_resource_pool_mgr` 会按 `trainer.nnodes` 和 `trainer.n_gpus_per_node` 做资源拆分：

- 单节点：同一节点 GPU 前半给 main agent，后半给 reranker。
- 多节点：前半节点给 main agent，后半节点给 reranker。

### 4.5 训练超参

脚本定义：

```bash
N_ROLLOUTS=8
TEMPERATURE=1.0
TOTAL_EPOCHS=1
TRAIN_BATCH_SIZE=512
ACTOR_LR=1e-6
ACTOR_BATCH_SIZE=128
ACTOR_MICRO_BATCH_SIZE_PER_GPU=1
LOG_PROB_MCRI_BATCH_SIZE_PER_GPU=2
ACTOR_LR_WARMUP_STEPS_RATIO=0.04
KL_LOSS_COEF=0.001
SAVE_FREQ=10
TEST_FREQ=20
```

这些会分别覆盖 main agent 和 reranker 的 actor / rollout / ref 配置。

典型映射：

```bash
actor_rollout_ref.rollout.n=${N_ROLLOUTS}
actor_rollout_ref.rollout.temperature=${TEMPERATURE}
actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_BATCH_SIZE}
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_MICRO_BATCH_SIZE_PER_GPU}
actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}

reranker_actor_rollout_ref.rollout.n=${N_ROLLOUTS}
reranker_actor_rollout_ref.rollout.temperature=${TEMPERATURE}
reranker_actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
reranker_actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_BATCH_SIZE}
reranker_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_MICRO_BATCH_SIZE_PER_GPU}
reranker_actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
```

### 4.6 reward / UID grouping / score assignment 参数

脚本定义：

```bash
REWARD_FN_PATH="${PROJECT_ROOT}/verl/verl/utils/reward_score/search_qa_f1_with_format_penalty.py"
TRAIN_REWARD_FN="search_qa_f1_penalty_compute_score"
FORMAT_PENALTY=-0.2

UID_GROUP_FN_PATH="${PROJECT_ROOT}/verl/verl/experimental/agent_loop/uid_group_functions.py"
UID_GROUP_FN_NAME="group_by_muid_ans_in_doc_subq_rougeL1"
UID_GROUP_THRESHOLD=0.8

SCORE_ASSIGN_FN_PATH="${PROJECT_ROOT}/verl/verl/experimental/agent_loop/score_assign_functions.py"
SCORE_ASSIGN_FN_NAME="sum_tool_agent_score_with_cond_threshold"
AGENT_THRESHOLD=0.8
COND_THRESHOLD=0.8
```

对应 override：

```bash
custom_reward_function.path="${REWARD_FN_PATH}"
custom_reward_function.name="${TRAIN_REWARD_FN}"
+custom_reward_function.reward_kwargs.format_penalty=${FORMAT_PENALTY}

reranker_uid_group_function.path="${UID_GROUP_FN_PATH}"
reranker_uid_group_function.name="${UID_GROUP_FN_NAME}"
+reranker_uid_group_function.uid_group_kwargs.threshold=${UID_GROUP_THRESHOLD}

reranker_score_assign_function.path="${SCORE_ASSIGN_FN_PATH}"
reranker_score_assign_function.name="${SCORE_ASSIGN_FN_NAME}"
+reranker_score_assign_function.score_assign_kwargs.agent_threshold=${AGENT_THRESHOLD}
+reranker_score_assign_function.score_assign_kwargs.cond_threshold=${COND_THRESHOLD}
```

这三组参数的共同特征是：Hydra 只保存路径、函数名和 kwargs，实际函数加载由 Python 代码完成。

## 5. Hydra 根配置的模块边界

### 5.1 `actor_rollout_ref`

主 agent 的训练、推理和 reference policy 组合。

子树：

```text
actor_rollout_ref.model
actor_rollout_ref.actor
actor_rollout_ref.rollout
actor_rollout_ref.ref
```

其中：

- `model` 控制 HF 模型加载、tokenizer、LoRA、gradient checkpointing。
- `actor` 控制训练更新：PPO/GRPO batch、KL loss、optimizer、FSDP。
- `rollout` 控制生成：vLLM、采样、TP、multi-turn、agent loop。
- `ref` 控制 reference policy：用于 KL loss 或 KL reward。

本入口中：

```bash
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.actor.use_kl_loss=True
```

因此主 agent 是 async vLLM rollout + multi-turn tool 调用 + actor KL loss。

### 5.2 `reranker_actor_rollout_ref`

reranker agent 的训练、推理和 reference policy 组合。

子树结构与 `actor_rollout_ref` 类似：

```text
reranker_actor_rollout_ref.model
reranker_actor_rollout_ref.actor
reranker_actor_rollout_ref.rollout
reranker_actor_rollout_ref.ref
reranker_actor_rollout_ref.trainable
```

默认 `co_search_trainer.yaml` 中：

```yaml
reranker_actor_rollout_ref:
  trainable: false
```

但脚本覆盖为：

```bash
reranker_actor_rollout_ref.trainable=True
```

因此这条入口训练 reranker，而不是只把 reranker 当固定 inference model。

### 5.3 `data`

来自 `config/data/legacy_data.yaml`。

负责：

- `train_files` / `val_files`
- `prompt_key`
- `reward_fn_key`
- `max_prompt_length`
- `max_response_length`
- `train_batch_size`
- `truncation`
- dataset class override
- tokenizer chat template kwargs

本入口覆盖：

```bash
data.max_prompt_length=20480
data.max_response_length=4096
data.train_batch_size=512
data.truncation='error'
```

注意：`rollout.prompt_length` 和 `rollout.response_length` 默认会通过 `${oc.select:data.max_prompt_length,512}` / `${oc.select:data.max_response_length,512}` 继承 data 长度，但脚本又显式覆盖了 rollout 的 prompt/response length，所以最终以脚本值为准。

### 5.4 `algorithm`

来自 `co_search_trainer.yaml`。

默认：

```yaml
algorithm:
  adv_estimator: gae
  use_kl_in_reward: False
  kl_penalty: kl
  kl_ctrl:
    type: fixed
    kl_coef: 0.001
```

脚本覆盖：

```bash
algorithm.adv_estimator=grpo
algorithm.use_kl_in_reward=False
```

因此该入口使用 GRPO advantage estimator，但不把 KL 放进 reward，而是通过 actor 的 `use_kl_loss=True` 走 policy loss 上的 KL loss。

### 5.5 `critic`

来自 `config/critic/dp_critic.yaml`。

脚本覆盖：

```bash
critic.enable=False
```

因此这条 GRPO 训练不启用 critic。

### 5.6 `reward_model`

来自 `config/reward_model/dp_reward_model.yaml`。

默认：

```yaml
enable: False
use_reward_loop: False
reward_manager: naive
```

脚本覆盖：

```bash
reward_model.enable=False
reward_model.reward_manager=multiturn
reward_model.use_reward_loop=True
```

这个组合含义比较特殊：

- 不启用单独的 reward model 权重服务。
- 启用 reward loop。
- 使用 `multiturn` reward manager。
- reward 由 `custom_reward_function` 指定的函数基于 multi-turn trajectory / tool extra fields 计算。

`main_co_search_ppo.py` 中如果 `reward_model.use_reward_loop=True`，则普通 `reward_fn` 和 `val_reward_fn` 会置为 `None`，后续 reward 在 agent loop 内通过 `RewardManagerWorker` 异步计算。

### 5.7 `custom_reward_function`

根配置中默认：

```yaml
custom_reward_function:
  path: null
  name: compute_score
  reward_kwargs: {}
```

脚本指定：

```bash
custom_reward_function.path=.../search_qa_f1_with_format_penalty.py
custom_reward_function.name=search_qa_f1_penalty_compute_score
+custom_reward_function.reward_kwargs.format_penalty=-0.2
```

边界：

- 这里只定义主 agent 最终 answer/reasoning 的 reward 计算函数。
- reranker 每次 search/rerank 的 tool score 不在这里定义，而是在 `CoSearchTool` 和 `reranker_score_assign_function` 中定义。

### 5.8 `reranker_uid_group_function`

根配置默认：

```yaml
reranker_uid_group_function:
  path: null
  name: group_by_muid_ans_in_doc
  uid_group_kwargs: {}
```

脚本指定：

```bash
reranker_uid_group_function.path=.../uid_group_functions.py
reranker_uid_group_function.name=group_by_muid_ans_in_doc_subq_rougeL1
+reranker_uid_group_function.uid_group_kwargs.threshold=0.8
```

边界：

- 只影响 reranker 的 GRPO group id。
- 不影响主 agent 的 dataset uid。
- 用来把同一 main uid 下的 reranker outputs 根据 `answer_in_docs` 和 sub-query 相似度进一步分组。

### 5.9 `reranker_score_assign_function`

根配置默认：

```yaml
reranker_score_assign_function:
  path: null
  name: max_tool_agent_score
  score_assign_kwargs: {}
```

脚本指定：

```bash
reranker_score_assign_function.path=.../score_assign_functions.py
reranker_score_assign_function.name=sum_tool_agent_score_with_cond_threshold
+reranker_score_assign_function.score_assign_kwargs.agent_threshold=0.8
+reranker_score_assign_function.score_assign_kwargs.cond_threshold=0.8
```

边界：

- 只影响 reranker 的训练分数。
- 它把 reranker tool score、主 agent continuation score、`answer_in_docs` 等信号合成为 reranker 的最终 score。
- 不直接改变主 agent 的 final reward。

### 5.10 `trainer`

根配置中定义训练过程控制：

```yaml
trainer:
  total_epochs: 30
  project_name: verl_examples
  experiment_name: gsm8k
  nnodes: 1
  n_gpus_per_node: 8
  save_freq: -1
  val_before_train: True
  test_freq: -1
  reranker_main_train_n_ratio: 2
  reranker_sampling_val_start_step: 1000
  reranker_filter_no_answer_in_docs: false
```

脚本覆盖：

```bash
trainer.nnodes=${NNODES}
trainer.n_gpus_per_node=${N_GPUS_PER_NODE}
trainer.total_epochs=1
trainer.experiment_name=${EXP_NAME}
trainer.val_before_train=False
trainer.project_name=co_search
trainer.save_freq=10
trainer.test_freq=20
trainer.reranker_sampling_val_start_step=10000
trainer.reranker_filter_no_answer_in_docs=false
```

边界：

- trainer 控制全局训练循环、checkpoint、validation、日志、资源拆分。
- agent/reranker 具体 batch、LR、KL 不在 trainer 中，而在各自 actor 子树中。

## 6. 运行时外挂配置详解

### 6.1 `co_search_agent_loop_config.yaml`

路径：

```text
CoSearch_derevitives/CoSearch/config/co_search_agent_loop_config.yaml
```

内容：

```yaml
- name: co_search_agent
  _target_: verl.experimental.agent_loop.co_search_agent_loop.CoSearchAgentLoop
  track_messages: false
```

它的作用是把字符串 `co_search_agent` 映射到 Python 类：

```text
verl.experimental.agent_loop.co_search_agent_loop.CoSearchAgentLoop
```

脚本通过以下 override 启用它：

```bash
actor_rollout_ref.rollout.agent.default_agent_loop=co_search_agent
actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENT_LOOP_CONFIG}
```

代码消费位置：

```text
verl/verl/experimental/agent_loop/agent_loop.py
```

其中 agent loop manager 会读取 `agent_loop_config_path`，把 YAML 中的 `name` 注册到 agent loop registry。

### 6.2 `/tmp/co_search_tool_config_*.yaml`

该文件由 shell 脚本临时生成，不在仓库中。

启用路径：

```bash
actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG}
```

消费位置：

```text
verl/verl/experimental/agent_loop/co_search_agent_loop.py
```

`CoSearchAgentLoop.init_class` 会调用：

```python
tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
```

tool registry 根据 YAML 中的：

```yaml
class_name: verl.tools.co_search_tool.CoSearchTool
```

动态实例化 `CoSearchTool`。

### 6.3 `CoSearchTool` 配置边界

`CoSearchTool` 负责：

1. 接收主 agent 产生的 search query。
2. 调 dense retrieval API，拿 top-N docs。
3. 如果 `use_reranker=True`，调用 reranker agent 对 top-N docs rerank。
4. 返回 top-M docs 给主 agent。
5. 计算 tool-level 指标，如 `average_hit_at_ks`、`ndcg_at_m`、`tool_score`、`answer_in_docs`。

主要配置项：

| 配置项 | 当前脚本值 | 作用 |
|---|---:|---|
| `retrieval_service_url` | `${RETRIEVAL_SERVICE_URL:-http://localhost:8000/retrieve}` | dense retriever HTTP API |
| `timeout` | `30` | 检索请求超时 |
| `max_retries` | `3` | 检索失败重试次数 |
| `retry_delay` | `1.0` | 初始重试等待 |
| `retry_backoff` | `2.0` | 指数退避倍率 |
| `default_top_n` | `50` | dense retrieval 取回文档数 |
| `default_top_m` | `5` | rerank 后返回给 agent 的文档数 |
| `tool_score_metric` | `hit` | tool score 使用 hit 还是 ndcg |
| `trivial_answers` | `yes/no/true/false` | 简单答案不做 answer-in-docs 正例 |
| `format_penalty` | `-0.2` | reranker 输出格式错误时的惩罚 |

当前动态 YAML 没有显式写 `use_reranker`。`CoSearchTool` 默认：

```python
self.use_reranker = config.get("use_reranker", True)
```

因此默认会启用 reranker。若要做 retrieval-only，需要在 tool YAML 中显式加入：

```yaml
use_reranker: false
```

### 6.4 函数路径配置

`reranker_uid_group_function` 和 `reranker_score_assign_function` 都通过 `load_custom_function` 加载：

```text
verl/verl/experimental/agent_loop/function_loaders.py
```

加载步骤：

1. 从 config 读取 `path` 和 `name`。
2. 用 `importlib.util.spec_from_file_location` 动态 import Python 文件。
3. 从 module 中取出对应函数。
4. 如果有 kwargs，用 `functools.partial` 包装。

因此这类配置的边界是函数级插件，不是 YAML-only 配置。

## 7. 主要 override 分组表

### 7.1 算法和数据

| Hydra key | 脚本值 | 含义 |
|---|---|---|
| `algorithm.use_kl_in_reward` | `False` | 不把 KL 放入 reward |
| `algorithm.adv_estimator` | `grpo` | 使用 GRPO advantage estimator |
| `data.train_files` | `${TRAIN_DATA}` | 训练 parquet |
| `data.val_files` | `${VAL_DATA}` | 验证 parquet |
| `data.train_batch_size` | `512` | 每次训练 batch |
| `data.max_prompt_length` | `20480` | prompt 最大长度 |
| `data.max_response_length` | `4096` | response 最大长度 |
| `data.truncation` | `error` | 超长时报错 |

### 7.2 主 agent model / rollout

| Hydra key | 脚本值 | 含义 |
|---|---|---|
| `actor_rollout_ref.model.path` | `Qwen/Qwen2.5-7B-Instruct` | 主 agent 初始模型 |
| `actor_rollout_ref.model.use_remove_padding` | `True` | remove padding 优化 |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | `True` | 训练省显存 |
| `actor_rollout_ref.rollout.name` | `vllm` | rollout engine |
| `actor_rollout_ref.rollout.mode` | `async` | async rollout |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `1` | vLLM TP |
| `actor_rollout_ref.rollout.n` | `8` | 每个 prompt rollout 数 |
| `actor_rollout_ref.rollout.temperature` | `1.0` | 训练采样温度 |
| `actor_rollout_ref.rollout.max_model_len` | `24576` | vLLM 最大上下文 |
| `actor_rollout_ref.rollout.prompt_length` | `20480` | rollout prompt 长度 |
| `actor_rollout_ref.rollout.response_length` | `4096` | rollout response 长度 |

### 7.3 主 agent multi-turn / tool / agent loop

| Hydra key | 脚本值 | 含义 |
|---|---|---|
| `actor_rollout_ref.rollout.multi_turn.enable` | `True` | 启用 multi-turn |
| `actor_rollout_ref.rollout.multi_turn.max_user_turns` | `6` | user/tool observation 最大轮数 |
| `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` | `6` | assistant 最大轮数 |
| `actor_rollout_ref.rollout.multi_turn.max_parallel_calls` | `1` | 每轮最多一个 tool call |
| `actor_rollout_ref.rollout.multi_turn.max_tool_response_length` | `4096` | tool response token 上限 |
| `actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side` | `left` | tool response 截断方向 |
| `actor_rollout_ref.rollout.multi_turn.format` | `search_r1` | tool parser 格式 |
| `actor_rollout_ref.rollout.multi_turn.tool_config_path` | `/tmp/co_search_tool_config_*.yaml` | tool YAML 路径 |
| `actor_rollout_ref.rollout.agent.num_workers` | `8` | agent loop worker 数 |
| `actor_rollout_ref.rollout.agent.default_agent_loop` | `co_search_agent` | 默认 agent loop |
| `actor_rollout_ref.rollout.agent.agent_loop_config_path` | `config/co_search_agent_loop_config.yaml` | agent loop registry YAML |

### 7.4 主 agent actor / ref

| Hydra key | 脚本值 | 含义 |
|---|---|---|
| `actor_rollout_ref.actor.kl_loss_coef` | `0.001` | actor KL loss 系数 |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `128` | PPO/GRPO mini batch |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | `1` | 单 GPU micro batch |
| `actor_rollout_ref.actor.optim.lr` | `1e-6` | actor 学习率 |
| `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio` | `0.04` | warmup ratio |
| `actor_rollout_ref.actor.use_kl_loss` | `True` | 启用 actor KL loss |
| `actor_rollout_ref.actor.kl_loss_type` | `low_var_kl` | KL estimator |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size` | `2` | sequence parallel size |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | `False` | 关闭参数 offload |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | `False` | 关闭 optimizer offload |
| `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | `2` | ref logprob micro batch |
| `actor_rollout_ref.ref.fsdp_config.param_offload` | `False` | ref 不 offload |

### 7.5 reward / trainer

| Hydra key | 脚本值 | 含义 |
|---|---|---|
| `critic.enable` | `False` | 不使用 critic |
| `reward_model.enable` | `False` | 不启用独立 RM 模型 |
| `reward_model.reward_manager` | `multiturn` | 使用 multiturn reward loop |
| `reward_model.use_reward_loop` | `True` | agent loop 内异步算 reward |
| `custom_reward_function.path` | `search_qa_f1_with_format_penalty.py` | 自定义 reward 文件 |
| `custom_reward_function.name` | `search_qa_f1_penalty_compute_score` | 自定义 reward 函数 |
| `custom_reward_function.reward_kwargs.format_penalty` | `-0.2` | reward 格式惩罚 |
| `trainer.total_epochs` | `1` | 总 epoch |
| `trainer.val_before_train` | `False` | 训练前不 validation |
| `trainer.logger` | `['console','wandb']` | 日志 backend |
| `trainer.save_freq` | `10` | checkpoint 间隔 |
| `trainer.test_freq` | `20` | validation 间隔 |
| `trainer.rollout_data_dir` | checkpoint 下 rollout_data | rollout dump |
| `trainer.validation_data_dir` | checkpoint 下 validation_data | validation dump |

### 7.6 reranker model / rollout / actor

reranker 的覆盖项基本与主 agent 对称，差异是 namespace 换成：

```text
reranker_actor_rollout_ref.*
```

额外关键项：

```bash
reranker_actor_rollout_ref.trainable=True
```

这使 reranker 进入可训练模式。trainer 会创建 `Role.RerankerActorRollout` 和 reranker ref policy，而不是只创建 inference-only `Role.RerankerRollout`。

### 7.7 reranker grouping / scoring

| Hydra key | 脚本值 | 含义 |
|---|---|---|
| `reranker_uid_group_function.path` | `uid_group_functions.py` | UID grouping 函数文件 |
| `reranker_uid_group_function.name` | `group_by_muid_ans_in_doc_subq_rougeL1` | 按 main uid、answer_in_docs、sub-query ROUGE 聚类 |
| `reranker_uid_group_function.uid_group_kwargs.threshold` | `0.8` | sub-query 聚类阈值 |
| `reranker_score_assign_function.path` | `score_assign_functions.py` | score assignment 函数文件 |
| `reranker_score_assign_function.name` | `sum_tool_agent_score_with_cond_threshold` | 条件合并 tool score 和 agent score |
| `reranker_score_assign_function.score_assign_kwargs.agent_threshold` | `0.8` | agent score 二值化阈值 |
| `reranker_score_assign_function.score_assign_kwargs.cond_threshold` | `0.8` | answer_in_docs=True 时的 tool score 门槛 |
| `trainer.reranker_sampling_val_start_step` | `10000` | 达到该 step 后 reranker 训练中使用 validation sampling params |
| `trainer.reranker_filter_no_answer_in_docs` | `false` | 期望过滤 no-answer-in-docs reranker 输出 |

## 8. 代码消费点

| 配置 key | 消费位置 | 行为 |
|---|---|---|
| `config_path="config", config_name="co_search_trainer"` | `main_co_search_ppo.py` | 选择 Hydra 根配置 |
| `ray_kwargs.ray_init` | `main_co_search_ppo.py` | 传给 `ray.init`，并与默认 PPO Ray runtime env merge |
| `actor_rollout_ref.model.path` | `main_co_search_ppo.py` | copy/download 主模型并创建 tokenizer/processor |
| `reranker_actor_rollout_ref.model.path` | `main_co_search_ppo.py` | copy/download reranker 模型并创建 tokenizer |
| `reward_model.use_reward_loop` | `main_co_search_ppo.py` | 决定普通 reward_fn 是否置空 |
| `data.train_files` / `data.val_files` | `main_co_search_ppo.py` | 创建 train/val dataset |
| `trainer.nnodes` / `trainer.n_gpus_per_node` | `DualAgentTaskRunner.init_resource_pool_mgr` | 拆分 main agent / reranker GPU pool |
| `reranker_actor_rollout_ref.trainable` | `main_co_search_ppo.py` / trainer | 决定 reranker 是 trainable 还是 inference-only |
| `actor_rollout_ref.rollout.agent.agent_loop_config_path` | `agent_loop.py` | 加载 agent loop registry YAML |
| `actor_rollout_ref.rollout.multi_turn.tool_config_path` | `co_search_agent_loop.py` | 加载 tool YAML |
| `actor_rollout_ref.rollout.multi_turn.format` | `co_search_agent_loop.py` | 选择 tool parser，例如 `search_r1` |
| `custom_reward_function.*` | `RewardManagerWorker` / `get_custom_reward_fn` | 动态加载 reward 函数 |
| `reranker_uid_group_function.*` | `CoSearchAgentLoopWorker` | 动态加载 UID grouping 函数 |
| `reranker_score_assign_function.*` | `CoSearchAgentLoopWorker` | 动态加载 score assignment 函数 |
| `trainer.reranker_sampling_val_start_step` | `CoSearchAgentLoopWorker.generate_sequences` | 决定 reranker sampling params 用 train 还是 val |

## 9. 当前入口的实际训练语义

按脚本当前参数，最终训练语义可以概括为：

1. 主 agent 和 reranker 都从 `Qwen/Qwen2.5-7B-Instruct` 初始化。
2. 使用 GRPO，不启用 critic。
3. 不使用 KL-in-reward，但主 agent 和 reranker actor 都启用 `use_kl_loss=True`。
4. 主 agent 用 async vLLM rollout，multi-turn search_r1 格式。
5. 主 agent 每次 search tool call 先调 dense retriever，再调 reranker agent rerank。
6. dense retriever URL 从 `RETRIEVAL_SERVICE_URL` 环境变量注入，默认 `http://localhost:8000/retrieve`。
7. reranker 是 trainable，不是固定模型。
8. reward model 权重服务关闭，但 `reward_model.use_reward_loop=True`，所以通过 multiturn reward loop 调自定义 reward 函数。
9. reranker 的 GRPO grouping 由 `group_by_muid_ans_in_doc_subq_rougeL1(threshold=0.8)` 控制。
10. reranker 的分数由 `sum_tool_agent_score_with_cond_threshold(agent_threshold=0.8, cond_threshold=0.8)` 控制。

## 10. 已发现的配置风险和不一致

### 10.1 `reranker_filter_no_answer_in_docs` 字段名可能不生效

脚本设置：

```bash
trainer.reranker_filter_no_answer_in_docs=${FILTER_NO_ANSWER_IN_DOCS}
```

`co_search_trainer.yaml` 中也定义：

```yaml
trainer:
  reranker_filter_no_answer_in_docs: false
```

但 `CoSearchAgentLoopWorker` 中读取的是：

```python
self.filter_no_answer_in_docs = config.trainer.get("filter_no_answer_in_docs", False)
```

也就是说代码读取 `trainer.filter_no_answer_in_docs`，而脚本和 YAML 写的是 `trainer.reranker_filter_no_answer_in_docs`。

影响：

- 当前脚本里的 `FILTER_NO_ANSWER_IN_DOCS` 很可能不会生效。
- 即使设置为 `true`，worker 仍会得到默认 `False`。

修正方向：

1. 要么脚本改传：

```bash
+trainer.filter_no_answer_in_docs=${FILTER_NO_ANSWER_IN_DOCS}
```

2. 要么 Python 代码改读：

```python
config.trainer.get("reranker_filter_no_answer_in_docs", False)
```

建议统一使用带 `reranker_` 前缀的字段名，因为根 YAML 已经这么定义，语义也更清楚。

### 10.2 Slurm 资源声明和脚本默认值不完全一致

Slurm 头部：

```bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
```

但脚本默认：

```bash
NNODES="${SLURM_NNODES:-2}"
N_GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
```

在 Slurm 正常注入变量时，通常会使用 Slurm 实际值。但如果手动在非 Slurm 环境直接运行脚本，默认会变成 `2 nodes x 8 GPUs`，并进入 multi-node Ray 分支，调用 `scontrol` / `srun`。这和脚本头部的 `1 node x 4 GPUs` 不一致。

建议：

- 若该脚本只允许 Slurm 运行，可以保留，但文档中注明。
- 若希望本地/单机可运行，应把默认值改成：

```bash
NNODES="${SLURM_NNODES:-1}"
N_GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
```

### 10.3 `VAL_REWARD_FN` 定义但没有使用

脚本定义：

```bash
TRAIN_REWARD_FN="search_qa_f1_penalty_compute_score"
VAL_REWARD_FN="search_qa_f1_penalty_compute_score"
```

但 override 只有：

```bash
custom_reward_function.name="${TRAIN_REWARD_FN}"
```

没有单独传 `VAL_REWARD_FN`。当前 train/val 使用同一个 custom reward function，因此功能上没有问题，但 `VAL_REWARD_FN` 变量是冗余的，可能误导读者以为训练和验证 reward 函数可以分别配置。

### 10.4 `hit_cutoffs` 在 tool YAML 中可能只是冗余配置

动态 tool YAML 写了：

```yaml
hit_cutoffs: [1, 3, 5]
```

但 `CoSearchTool.__init__` 没有把 `hit_cutoffs` 缓存为成员变量。`execute` 里读取的是：

```python
hit_cutoffs = create_kwargs.get("hit_cutoffs", [1,3,5])
```

因此当前 YAML 中的 `hit_cutoffs` 是否生效取决于 tool registry/BaseTool 是否把 config 传入了 `create_kwargs`。从 `CoSearchTool` 本身看，它至少有内置默认 `[1,3,5]`，所以当前行为与脚本期望一致；但如果未来想改成 `[1,5,10]`，需要确认这一路是否真正消费。

更稳妥的实现是让 `CoSearchTool.__init__` 显式读取：

```python
self.hit_cutoffs = config.get("hit_cutoffs", [1, 3, 5])
```

然后 `execute` 中使用：

```python
hit_cutoffs = create_kwargs.get("hit_cutoffs", self.hit_cutoffs)
```

### 10.5 `config/search_r1_tools_config.yaml` 容易被误认为生效

仓库中存在：

```text
config/search_r1_tools_config.yaml
```

但此入口实际使用的是脚本动态生成的：

```text
/tmp/co_search_tool_config_${SLURM_JOB_ID:-local}.yaml
```

原因是 retrieval URL 需要由环境变量 `RETRIEVAL_SERVICE_URL` 注入，而静态 YAML 无法直接引用 shell 变量。

因此排查 tool 参数时，应优先看 Slurm 输出里打印的：

```text
Generated tool config: /tmp/co_search_tool_config_...yaml
```

而不是仓库内的 `search_r1_tools_config.yaml`。

### 10.6 `reward_model.enable=False` 和 `reward_model.use_reward_loop=True` 不矛盾

这组配置看起来容易误解：

```bash
reward_model.enable=False
reward_model.reward_manager=multiturn
reward_model.use_reward_loop=True
```

含义是：

- 不启动独立 reward model 权重。
- 仍然启用 reward loop。
- reward loop 使用 rule/custom reward 函数，而不是 RM 模型。

所以不能简单把 `reward_model.enable=False` 理解成“不计算 reward”。

## 11. 修改配置时的建议边界

### 11.1 改模型或数据

优先改 shell 变量：

```bash
CHECKPOINT_PATH=...
RERANKER_CHECKPOINT_PATH=...
TRAIN_DATA=...
VAL_DATA=...
```

或在命令行追加 Hydra override。

### 11.2 改检索服务

优先通过环境变量：

```bash
RETRIEVAL_SERVICE_URL=http://host:port/retrieve \
bash scripts/train_co_search_grpo.sh
```

不要直接改 `config/search_r1_tools_config.yaml`，因为此入口不使用它。

### 11.3 改 tool 行为

需要修改脚本生成的 tool YAML 内容，或改成引用一个静态 co_search tool YAML。

典型可改项：

```yaml
default_top_n: 50
default_top_m: 5
tool_score_metric: "hit"   # or "ndcg"
format_penalty: -0.2
use_reranker: true
```

### 11.4 改 agent loop 类

改：

```text
config/co_search_agent_loop_config.yaml
```

以及脚本中的：

```bash
actor_rollout_ref.rollout.agent.default_agent_loop=co_search_agent
```

如果只改 YAML 中的 `name`，但不改 `default_agent_loop`，数据样本未显式提供 `agent_name` 时会找不到对应 agent loop。

### 11.5 改 reranker grouping 或 scoring

优先改脚本变量：

```bash
UID_GROUP_FN_NAME=...
UID_GROUP_THRESHOLD=...
SCORE_ASSIGN_FN_NAME=...
AGENT_THRESHOLD=...
COND_THRESHOLD=...
```

如果新增函数，需要放在对应 Python 文件中，或把 `*_FN_PATH` 指向新文件。

### 11.6 改 GRPO/PPO 基础训练超参

主 agent 和 reranker 当前使用同一组 shell 变量。若希望二者不同，应拆分变量，例如：

```bash
MAIN_ACTOR_LR=...
RERANKER_ACTOR_LR=...
MAIN_ACTOR_BATCH_SIZE=...
RERANKER_ACTOR_BATCH_SIZE=...
```

然后分别覆盖：

```bash
actor_rollout_ref.actor.optim.lr=${MAIN_ACTOR_LR}
reranker_actor_rollout_ref.actor.optim.lr=${RERANKER_ACTOR_LR}
```

### 11.7 改 reward

主 agent final reward：

```bash
custom_reward_function.path=...
custom_reward_function.name=...
+custom_reward_function.reward_kwargs.xxx=...
```

reranker tool/reward score：

```bash
tool_score_metric=...
format_penalty=...
reranker_score_assign_function.name=...
```

这两者不要混淆：`custom_reward_function` 主要服务主 agent final answer reward；reranker 的每步分数还依赖 tool metrics 和 score assignment function。

## 12. 如何查看最终 Hydra 配置

如果只想看 Hydra merge 后的配置，可以在 `CoSearch` 目录下用 Hydra 的 `--cfg job`：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoSearch
python main_co_search_ppo.py --cfg job
```

若想带上脚本中的所有 override，需要把脚本最后 `python main_co_search_ppo.py \ ...` 那串参数复制出来，并在末尾加：

```bash
--cfg job
```

注意：

- `--cfg job` 只打印 Hydra 配置，不应启动训练主体。
- 动态 tool config 仍需要先存在；也就是脚本生成 `/tmp/co_search_tool_config_*.yaml` 后再打印最准确。

## 13. 一句话边界图

```text
Slurm/shell
  控制资源、环境变量、路径变量、实验名、临时 tool YAML

Hydra root: config/co_search_trainer.yaml
  控制完整训练配置树，并通过 defaults 组合组件默认值

Hydra component YAML
  控制 actor/data/model/rollout/ref/critic/reward_model 等默认结构

Hydra command line overrides
  控制本次实验真正不同于默认值的地方

Agent loop config
  控制 agent loop 名称到 Python 类的映射

Tool config
  控制 search tool、retrieval API、top-n/top-m、tool score 细节

Python function config
  控制 reward、reranker grouping、reranker score assignment 的插件函数

Dataset/runtime fields
  控制每条样本的 question、answers、uid、tools_kwargs、reward_model ground truth
```

这条入口的核心特点是：Hydra 管“大训练配置树”，shell 管“实验实例化和运行环境”，agent/tool/function config 管“CoSearch 多轮搜索和 reranker 训练的可插拔逻辑”。
