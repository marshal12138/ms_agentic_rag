# `docs/` 索引

这个文件是 `docs/` 的常驻入口。作用只有两个：

1. 说明 `docs/` 下各目录和关键文档分别记录什么。
2. 规定这些文档在什么情况下需要同步更新。

## 目录与文档

| 路径 | 作用 |
| --- | --- |
| `docs/readme.md` | 本索引页。用于快速了解 `docs/` 结构和更新规则。 |
| `docs/framework.md` | 项目级框架入口。说明项目能力、核心目录职责和标准使用过程，不作为 `docs/` 索引页。 |
| `docs/framework_AgenticIterRag.md` | AgenticIterRag 相关框架文档占位页；当前基本未维护。 |
| `docs/framework_CoAgenticRetriever.md` | `CoAgenticRetriever/` 目录结构、核心实现入口、策略配置点说明。 |
| `docs/src.md` | `src/` 目录说明，重点记录共享日志、报告、checkpoint 等基础设施。 |
| `docs/Experiments/` | 实验入口和训练脚本使用说明。 |
| `docs/Experiments/how_to_train.md` | 训练脚本使用方法，重点记录 `EXP_NAME`、`RUN_NAME`、`GROUP_NAME`、日志与 checkpoint 路径。 |
| `docs/FQA/` | 常见问题、机制解释、指标含义、反复被问到的结论。 |
| `docs/FQA/250604.md` | CoSearch FAQ，记录 reward、训练机制、指标解释等问答内容。 |
| `docs/planning/` | 方案设计和改造计划。这里记录“准备做什么”，不代表当前代码已经实现。 |
| `docs/planning/260606_reranker_bce_step_plan.md` | reranker BCE step 方案草案。 |
| `docs/planning/260606_retriever_contrastive_step_plan.md` | 早期 retriever contrastive 方案。 |
| `docs/planning/260610_retriever_contrastive_step_plan_fix_ver2.md` | 当前 CoAgenticRetriever retriever 训练修正版方案。 |
| `docs/pre_works/` | 阶段性工作总结、复现记录、历史实验交接材料。 |
| `docs/pre_works/20260602_cosearch_复现前期工作总结.md` | CoSearch 早期复现与环境准备总结。 |
| `docs/pre_works/20260603_qwen3_4b配置实验工作总结.md` | Qwen3-4B 配置与实验记录总结。 |
| `docs/pre_works/20260610_coagentic_retriever核心框架工作总结.md` | CoAgenticRetriever 核心框架实现与验证记录。 |
| `docs/train_and_eval/` | 正式训练/评估说明。适合放稳定入口，而不是临时方案。 |
| `docs/train_and_eval/cosearch_eval_basic.md` | CoSearch 基础评估说明，记录 `11_evaluate_cosearch_base.sh` 的用法和默认行为。 |
| `docs/train_and_eval/coAgenticRetriever_eval_basic.md` | CoAgenticRetriever 基础评估说明，记录 CAR 评估入口、路径和默认行为。 |

## `docs/framework.md` 的作用和更新原则

`docs/framework.md` 是项目级总览，不是 `docs/` 目录索引。它应回答：

1. 这个项目当前具备哪些能力。
2. 项目有哪些核心目录，每个目录承担什么职责。
3. 标准训练、推理、评估和结果沉淀流程应该怎样串起来。

更新 `docs/framework.md` 时遵循以下原则：

- 只写项目结构、职责边界和标准流程，不写单次实验流水账。
- 目录职责发生变化、新增长期维护的一线模块或公共模块时，需要同步更新。
- 新增临时脚本、一次性 pipeline、单次评估结果时，不直接扩写 `framework.md`；应先沉淀到对应专题文档、`reports/` 或 `experiences/`。
- 具体命令、参数、日志路径、指标细节应放到 `docs/Experiments/`、`docs/train_and_eval/`、`docs/src.md` 或对应专题文档。
- `docs/` 目录下文档索引和更新规则由本页维护，不放到 `docs/framework.md`。

## 使用顺序

建议按下面顺序查阅：

1. 想了解 `docs/` 里该看哪一类文档：先看 `docs/readme.md`。
2. 想了解整个项目能力、目录职责和标准使用过程：看 `docs/framework.md`。
3. 想看训练入口怎么用：看 `docs/Experiments/how_to_train.md`。
4. 想看共享日志、checkpoint、报告系统：看 `docs/src.md`。
5. 想看 CoAgenticRetriever 核心实现结构：看 `docs/framework_CoAgenticRetriever.md`。
6. 想看历史背景、踩坑记录：看 `docs/pre_works/`。
7. 想看未来方案或未落地设计：看 `docs/planning/`。

## 更新规则

- 新增 `docs/` 下的目录或关键文档时，必须同步更新本页。
- 某份文档的职责发生变化时，必须同步更新本页对应描述。
- 新增稳定训练/评估入口时，优先更新 `docs/Experiments/how_to_train.md` 或 `docs/train_and_eval/`，然后再更新本页索引。
- 新增方案设计时，放入 `docs/planning/`，并在本页补一行用途说明。
- 新增阶段性总结或交接材料时，放入 `docs/pre_works/`，并在本页补一行用途说明。
- FAQ 类内容只放入 `docs/FQA/`，不要混入 `planning` 或 `pre_works`。
- 本页保持简洁，不记录参数细节、命令细节、实现细节；这些内容应写回对应专题文档。
