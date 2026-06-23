1. 报告说明：
1.1 用llm as judge作为ranker，agent用base版本
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/reports/eval/coAgenticRetriever/260622-1551-async_label_dpskv4f_v0616_llm_judge_rank_wt_ori_agt_llm.report.md

1.2 用llm as judge作为ranker，agent用CAR训练后版本：
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/reports/eval/coAgenticRetriever/260622-1415-async_label_dpskv4f_v0616_llm_judge_rank.report.md

1.3 agent+dense ranker训练后，full 测试：
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/reports/eval/coAgenticRetriever/260622-0920-async_label_dpskv4f_v0616_full.report.md

1.4 agent+dense ranker训练后，no-ranker 测试：
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/reports/eval/coAgenticRetriever/260622-0922-async_label_dpskv4f_v0616_full_no_ranker.report.md

1.5 base模型-full:
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/reports/eval/coAgenticRetriever/260622-1636-coageticretriever_origin_full.report.md

1.6 base模型-no-ranker:
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/reports/eval/coAgenticRetriever/260622-1639-coageticretriever_origin_no_ranker.report.md


## 2. Effect Metrics 汇总

| 编号 | 测试设置 | Run mode | Ranker | N | Micro EM | Micro F1 | Macro EM | Macro F1 |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1.1 | base agent + LLM-as-judge ranker | full | LLM judge | 350 | 0.2914 | 0.3791 | 0.2914 | 0.3791 |
| 1.2 | CAR 训练后 agent + LLM-as-judge ranker | full | LLM judge | 350 | 0.3686 | 0.4518 | 0.3686 | 0.4518 |
| 1.3 | CAR 训练后 agent + 训练后 dense ranker | full | dense ranker | 350 | 0.3086 | 0.3928 | 0.3086 | 0.3928 |
| 1.4 | CAR 训练后 agent + recall top-5 | no-ranker | disabled | 350 | 0.3171 | 0.3974 | 0.3171 | 0.3974 |
| 1.5 | base agent + base dense ranker | full | dense_e5 | 350 | 0.2657 | 0.3482 | 0.2657 | 0.3482 |
| 1.6 | base agent + recall top-5 | no-ranker | disabled | 350 | 0.2571 | 0.3422 | 0.2571 | 0.3422 |

## 3. Effect Metrics By Dataset 汇总

| 数据集 | 1.1 EM | 1.1 F1 | 1.2 EM | 1.2 F1 | 1.3 EM | 1.3 F1 | 1.4 EM | 1.4 F1 | 1.5 EM | 1.5 F1 | 1.6 EM | 1.6 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 0.1800 | 0.2148 | 0.3000 | 0.3403 | 0.2800 | 0.3261 | 0.2800 | 0.3152 | 0.1800 | 0.2444 | 0.2000 | 0.2607 |
| bamboogle | 0.2200 | 0.3309 | 0.3000 | 0.4273 | 0.1000 | 0.1950 | 0.2000 | 0.2900 | 0.1800 | 0.2426 | 0.1800 | 0.2427 |
| hotpotqa | 0.2400 | 0.3532 | 0.3800 | 0.4720 | 0.3000 | 0.3640 | 0.2600 | 0.3651 | 0.2200 | 0.3010 | 0.2400 | 0.3224 |
| musique | 0.1000 | 0.1778 | 0.0800 | 0.1398 | 0.1000 | 0.2094 | 0.1400 | 0.1822 | 0.1200 | 0.2043 | 0.1200 | 0.2066 |
| nq | 0.3600 | 0.4627 | 0.4400 | 0.5593 | 0.4600 | 0.5650 | 0.3800 | 0.4833 | 0.3800 | 0.4557 | 0.3000 | 0.3934 |
| popqa | 0.3800 | 0.4564 | 0.3800 | 0.4642 | 0.3800 | 0.4642 | 0.3400 | 0.4394 | 0.2800 | 0.3502 | 0.2600 | 0.3302 |
| triviaqa | 0.5600 | 0.6578 | 0.7000 | 0.7593 | 0.5400 | 0.6260 | 0.6200 | 0.7067 | 0.5000 | 0.6394 | 0.5000 | 0.6394 |
