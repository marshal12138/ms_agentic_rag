# 当前 Config 调用链

本文只说明当前 `main_coagentic_retriever.py` 训练入口实际用到的 config 链路。

需要先区分两类 config：

- Hydra defaults 自动组合进最终配置的 YAML。
- 运行时通过某个字段路径再加载的 YAML。

这两类都“被使用”，但调用方式不同。

## 1. Python 入口

当前训练入口：

```text
CoAgenticRetriever/main_coagentic_retriever.py
```

Hydra 入口声明：

```python
@hydra.main(
    config_path="config",
    config_name="coagentic_retriever_trainer",
    version_base=None,
)
```

含义：

```text
main_coagentic_retriever.py
  -> CoAgenticRetriever/config/coagentic_retriever_trainer.yaml
```

`coagentic_retriever_trainer.yaml` 是这个入口的唯一 Hydra 主配置。

## 2. 主 Config 的 Defaults

`coagentic_retriever_trainer.yaml` 通过 `defaults` 自动组合这些配置：

```text
coagentic_retriever_trainer.yaml
├── actor@actor_rollout_ref.actor: dp_actor
├── data@data: legacy_data
├── ref@actor_rollout_ref.ref: dp_ref
├── rollout@actor_rollout_ref.rollout: rollout
├── model@actor_rollout_ref.model: hf_model
├── critic@critic: dp_critic
├── reward_model@reward_model: dp_reward_model
├── algorithm@algorithm.rollout_correction: rollout_correction
├── experimental/ranker_base@_global_: ranker_contrastive
├── experimental/async_ranker_training_base@_global_: async_ranker_training
└── _self_
```

展开成实际文件：

```text
CoAgenticRetriever/config/coagentic_retriever_trainer.yaml
├── CoAgenticRetriever/config/actor/dp_actor.yaml
├── CoAgenticRetriever/config/data/legacy_data.yaml
├── CoAgenticRetriever/config/ref/dp_ref.yaml
├── CoAgenticRetriever/config/rollout/rollout.yaml
├── CoAgenticRetriever/config/model/hf_model.yaml
├── CoAgenticRetriever/config/critic/dp_critic.yaml
├── CoAgenticRetriever/config/reward_model/dp_reward_model.yaml
├── CoAgenticRetriever/config/algorithm/rollout_correction.yaml
├── CoAgenticRetriever/config/experimental/ranker_base/ranker_contrastive.yaml
└── CoAgenticRetriever/config/experimental/async_ranker_training_base/async_ranker_training.yaml
```

## 3. 嵌套 Defaults

上面有几个配置本身还会继续组合基础配置。

### 3.1 Actor

主配置选择：

```text
actor@actor_rollout_ref.actor: dp_actor
```

`actor/dp_actor.yaml` 内部继续组合：

```text
actor/dp_actor.yaml
├── ../optim@optim: fsdp
├── ../engine@fsdp_config: fsdp
├── actor
└── _self_
```

展开成实际文件：

```text
CoAgenticRetriever/config/actor/dp_actor.yaml
├── CoAgenticRetriever/config/optim/fsdp.yaml
├── CoAgenticRetriever/config/engine/fsdp.yaml
└── CoAgenticRetriever/config/actor/actor.yaml
```

最终挂载位置：

```text
actor_rollout_ref.actor
```

### 3.2 Reference Model

主配置选择：

```text
ref@actor_rollout_ref.ref: dp_ref
```

`ref/dp_ref.yaml` 内部继续组合：

```text
ref/dp_ref.yaml
├── ref
├── ../engine@fsdp_config: fsdp
└── _self_
```

展开成实际文件：

```text
CoAgenticRetriever/config/ref/dp_ref.yaml
├── CoAgenticRetriever/config/ref/ref.yaml
└── CoAgenticRetriever/config/engine/fsdp.yaml
```

最终挂载位置：

```text
actor_rollout_ref.ref
```

### 3.3 Critic

主配置选择：

```text
critic@critic: dp_critic
```

`critic/dp_critic.yaml` 内部继续组合：

```text
critic/dp_critic.yaml
├── ../optim@optim: fsdp
├── ../engine@model.fsdp_config: fsdp
├── critic
└── _self_
```

展开成实际文件：

```text
CoAgenticRetriever/config/critic/dp_critic.yaml
├── CoAgenticRetriever/config/optim/fsdp.yaml
├── CoAgenticRetriever/config/engine/fsdp.yaml
└── CoAgenticRetriever/config/critic/critic.yaml
```

最终挂载位置：

```text
critic
```

注意：即使当前训练通过 runtime 配置关闭 critic worker，`critic` 配置仍会被 Hydra 组合进最终 config。

### 3.4 Reward Model

主配置选择：

```text
reward_model@reward_model: dp_reward_model
```

`reward_model/dp_reward_model.yaml` 内部继续组合：

```text
reward_model/dp_reward_model.yaml
├── reward_model
└── _self_
```

展开成实际文件：

```text
CoAgenticRetriever/config/reward_model/dp_reward_model.yaml
└── CoAgenticRetriever/config/reward_model/reward_model.yaml
```

最终挂载位置：

```text
reward_model
```

注意：即使当前训练通过 `reward_model.enable=false` 关闭 reward model，这份配置仍会被 Hydra 组合进最终 config。

## 4. 直接叶子 Config

下面这些配置由 `coagentic_retriever_trainer.yaml` 直接选择，本身没有继续组合其它 YAML。

```text
data@data: legacy_data
  -> CoAgenticRetriever/config/data/legacy_data.yaml
  -> 最终挂载位置：data

rollout@actor_rollout_ref.rollout: rollout
  -> CoAgenticRetriever/config/rollout/rollout.yaml
  -> 最终挂载位置：actor_rollout_ref.rollout

model@actor_rollout_ref.model: hf_model
  -> CoAgenticRetriever/config/model/hf_model.yaml
  -> 最终挂载位置：actor_rollout_ref.model

algorithm@algorithm.rollout_correction: rollout_correction
  -> CoAgenticRetriever/config/algorithm/rollout_correction.yaml
  -> 最终挂载位置：algorithm.rollout_correction

experimental/ranker_base@_global_: ranker_contrastive
  -> CoAgenticRetriever/config/experimental/ranker_base/ranker_contrastive.yaml
  -> 最终挂载位置：trainer / recall_retriever / ranker / ranker_training

experimental/async_ranker_training_base@_global_: async_ranker_training
  -> CoAgenticRetriever/config/experimental/async_ranker_training_base/async_ranker_training.yaml
  -> 最终挂载位置：ranker_training.signal_source / ranker_training.shared_inference_ranker / ranker_training.async_ranker_training
```

## 5. 运行时引用的 Config

下面两个 YAML 不在 `coagentic_retriever_trainer.yaml` 的 `defaults` 里，所以不是 Hydra 自动组合进来的。

它们通过最终 config 里的路径字段被运行时代码加载：

```text
actor_rollout_ref.rollout.multi_turn.tool_config_path
actor_rollout_ref.rollout.agent.agent_loop_config_path
```

当前使用的文件：

```text
CoAgenticRetriever/config/coagentic_retriever_tool_config.yaml
CoAgenticRetriever/config/coagentic_retriever_agent_loop_config.yaml
```

调用关系：

```text
coagentic_retriever_trainer.yaml
  -> rollout@actor_rollout_ref.rollout: rollout/rollout.yaml
  -> launcher / Hydra override 设置：
       actor_rollout_ref.rollout.multi_turn.tool_config_path
       actor_rollout_ref.rollout.agent.agent_loop_config_path
  -> 运行时加载：
       coagentic_retriever_tool_config.yaml
       coagentic_retriever_agent_loop_config.yaml
```

关键区别：

```text
Hydra defaults 自动组合：
  actor/dp_actor.yaml
  data/legacy_data.yaml
  ref/dp_ref.yaml
  rollout/rollout.yaml
  model/hf_model.yaml
  critic/dp_critic.yaml
  reward_model/dp_reward_model.yaml
  algorithm/rollout_correction.yaml
  experimental/ranker_base/ranker_contrastive.yaml
  experimental/async_ranker_training_base/async_ranker_training.yaml

字段路径指向后运行时加载：
  coagentic_retriever_tool_config.yaml
  coagentic_retriever_agent_loop_config.yaml
```

## 6. 一页总览

```text
main_coagentic_retriever.py
└── config/coagentic_retriever_trainer.yaml
    ├── actor@actor_rollout_ref.actor: actor/dp_actor.yaml
    │   ├── optim/fsdp.yaml
    │   ├── engine/fsdp.yaml
    │   └── actor/actor.yaml
    ├── data@data: data/legacy_data.yaml
    ├── ref@actor_rollout_ref.ref: ref/dp_ref.yaml
    │   ├── ref/ref.yaml
    │   └── engine/fsdp.yaml
    ├── rollout@actor_rollout_ref.rollout: rollout/rollout.yaml
    ├── model@actor_rollout_ref.model: model/hf_model.yaml
    ├── critic@critic: critic/dp_critic.yaml
    │   ├── optim/fsdp.yaml
    │   ├── engine/fsdp.yaml
    │   └── critic/critic.yaml
    ├── reward_model@reward_model: reward_model/dp_reward_model.yaml
    │   └── reward_model/reward_model.yaml
    ├── algorithm@algorithm.rollout_correction: algorithm/rollout_correction.yaml
    ├── experimental/ranker_base/ranker_contrastive.yaml
    ├── experimental/async_ranker_training_base/async_ranker_training.yaml
    └── _self_

运行时路径引用，不是 Hydra defaults：
├── coagentic_retriever_tool_config.yaml
└── coagentic_retriever_agent_loop_config.yaml
```

## 7. 当前链路外的文件

`CoAgenticRetriever/config` 里还有其它 YAML，例如 PPO、SFT、Megatron、generation、evaluation 相关配置。

这些文件不属于当前 `main_coagentic_retriever.py -> coagentic_retriever_trainer.yaml` 链路。只有切换 Python 入口，或显式通过 Hydra override 选择它们时，它们才会参与当前任务。
