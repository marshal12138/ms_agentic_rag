# AIR End-Point Dense Rank Score Analysis

- run_id: `260708-161708-dense_rank_endpoint_score_analysis`
- generated_at: `2026-07-08T16:17:32`
- source_manifest: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/llm_reranker_branch_train_set/260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_end_point_top50_top5_short_reason/manifest.json`
- source_dataset_jsonl: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/llm_reranker_branch_train_set/260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_end_point_top50_top5_short_reason/dataset.jsonl`
- prompt_template_version: `cosearch_rerank_topm_v1_short_reason_fixed_example`
- candidate_top_n: `50`
- visible_top_m: `5`
- json_summary: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/pipelines/formal/agenticIterRag/dense_rank_analysis/reports/260708-161708-dense_rank_endpoint_score_analysis.summary.json`
- row_metrics_csv: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/pipelines/formal/agenticIterRag/dense_rank_analysis/reports/260708-161708-dense_rank_endpoint_score_analysis.row_metrics.csv`

## Analysis Objective

Compare dense E5 recall score properties for end-point search queries between:

- Group A: `top50_hit_top5_miss`, where top50 contains answer evidence but original dense top5 does not.
- Group B: `top5_hit`, where original dense top5 already contains answer evidence.
- Group C: `top50_miss`, where top50 contains no answer evidence.

The query object is the final search query of each AIR trajectory (`step_policy=end_point`).

## Group Counts

| group | count | ratio |
| --- | ---: | ---: |
| A top50_hit_top5_miss | 642 | 0.1259 |
| B top5_hit | 3250 | 0.6373 |
| C top50_miss | 1208 | 0.2369 |
| total | 5100 | 1.0000 |

## Group A: top50_hit_top5_miss

| metric | n | mean | std | min | p25 | p50 | p75 | p90 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top1_score | 642 | 0.856853 | 0.025882 | 0.800781 | 0.836914 | 0.856445 | 0.876465 | 0.891602 | 0.899902 | 0.933594 |
| top5_mean_score | 642 | 0.846183 | 0.023799 | 0.789551 | 0.828809 | 0.846191 | 0.864160 | 0.878027 | 0.885059 | 0.915625 |
| top5_min_score | 642 | 0.838552 | 0.023251 | 0.777344 | 0.821289 | 0.837891 | 0.856934 | 0.868652 | 0.876953 | 0.903809 |
| top50_mean_score | 642 | 0.823402 | 0.020609 | 0.772920 | 0.807441 | 0.821143 | 0.838584 | 0.851543 | 0.857998 | 0.885225 |
| top50_score_range | 642 | 0.043605 | 0.016783 | 0.013184 | 0.030273 | 0.041504 | 0.053711 | 0.066895 | 0.075195 | 0.111328 |
| answer_doc_count | 642 | 2.750779 | 3.076928 | 1.000000 | 1.000000 | 2.000000 | 3.000000 | 6.000000 | 8.000000 | 31.000000 |
| best_answer_score | 642 | 0.826792 | 0.023358 | 0.768555 | 0.809082 | 0.822754 | 0.843750 | 0.858398 | 0.866699 | 0.890137 |
| best_answer_rank | 642 | 18.242991 | 12.025032 | 6.000000 | 8.000000 | 14.000000 | 26.000000 | 38.000000 | 44.000000 | 50.000000 |
| best_top5_nonanswer_score | 642 | 0.856853 | 0.025882 | 0.800781 | 0.836914 | 0.856445 | 0.876465 | 0.891602 | 0.899902 | 0.933594 |
| score_gap_top5_nonanswer_minus_answer | 642 | 0.030060 | 0.015672 | 0.004395 | 0.018555 | 0.026367 | 0.038574 | 0.050781 | 0.061523 | 0.109863 |

## Group B: top5_hit

| metric | n | mean | std | min | p25 | p50 | p75 | p90 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top1_score | 3250 | 0.867319 | 0.026383 | 0.777344 | 0.850098 | 0.870117 | 0.886230 | 0.899414 | 0.906738 | 0.940918 |
| top5_mean_score | 3250 | 0.853759 | 0.024089 | 0.775488 | 0.837500 | 0.856055 | 0.871387 | 0.883398 | 0.890527 | 0.923633 |
| top5_min_score | 3250 | 0.843839 | 0.023783 | 0.774902 | 0.826660 | 0.845215 | 0.861816 | 0.873535 | 0.880859 | 0.913086 |
| top50_mean_score | 3250 | 0.825901 | 0.020813 | 0.768086 | 0.810215 | 0.825947 | 0.841709 | 0.852900 | 0.859727 | 0.889814 |
| top50_score_range | 3250 | 0.053250 | 0.018636 | 0.012695 | 0.039551 | 0.051758 | 0.064941 | 0.077637 | 0.086914 | 0.131348 |
| answer_doc_count | 3250 | 12.309538 | 12.452032 | 1.000000 | 3.000000 | 7.000000 | 17.000000 | 34.000000 | 40.000000 | 50.000000 |
| best_answer_score | 3250 | 0.864221 | 0.027247 | 0.775391 | 0.846191 | 0.866211 | 0.884277 | 0.897949 | 0.905762 | 0.940918 |
| best_answer_rank | 3250 | 1.550769 | 1.029731 | 1.000000 | 1.000000 | 1.000000 | 2.000000 | 3.000000 | 4.000000 | 5.000000 |
| best_top5_nonanswer_score | 2810 | 0.855746 | 0.025625 | 0.777344 | 0.838379 | 0.857422 | 0.874512 | 0.887695 | 0.895996 | 0.929199 |
| score_gap_top5_nonanswer_minus_answer | 2810 | -0.007645 | 0.017846 | -0.090820 | -0.018555 | -0.006348 | 0.003906 | 0.013184 | 0.019043 | 0.064453 |

## Group C: top50_miss

| metric | n | mean | std | min | p25 | p50 | p75 | p90 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top1_score | 1208 | 0.847044 | 0.026363 | 0.790039 | 0.827637 | 0.845215 | 0.865234 | 0.883789 | 0.893555 | 0.930176 |
| top5_mean_score | 1208 | 0.835464 | 0.022999 | 0.787695 | 0.818359 | 0.832812 | 0.850586 | 0.866895 | 0.876855 | 0.910547 |
| top5_min_score | 1208 | 0.827266 | 0.021595 | 0.783691 | 0.810059 | 0.824219 | 0.841309 | 0.857910 | 0.867188 | 0.903320 |
| top50_mean_score | 1208 | 0.813351 | 0.018395 | 0.771592 | 0.799785 | 0.810791 | 0.824023 | 0.838711 | 0.847979 | 0.885781 |
| top50_score_range | 1208 | 0.043006 | 0.018799 | 0.008789 | 0.028320 | 0.039551 | 0.055664 | 0.069824 | 0.078613 | 0.110352 |
| answer_doc_count | 1208 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| best_answer_score | 0 |  |  |  |  |  |  |  |  |  |
| best_answer_rank | 0 |  |  |  |  |  |  |  |  |  |
| best_top5_nonanswer_score | 1208 | 0.847044 | 0.026363 | 0.790039 | 0.827637 | 0.845215 | 0.865234 | 0.883789 | 0.893555 | 0.930176 |
| score_gap_top5_nonanswer_minus_answer | 0 |  |  |  |  |  |  |  |  |  |

## Key Comparisons

| metric | Group A top50_hit_top5_miss | Group B top5_hit | delta A-B |
| --- | ---: | ---: | ---: |
| top1_score mean | 0.856853 | 0.867319 | -0.010466 |
| top5_mean_score mean | 0.846183 | 0.853759 | -0.007576 |
| best_answer_score mean | 0.826792 | 0.864221 | -0.037429 |
| best_answer_rank mean | 18.242991 | 1.550769 | 16.692221 |
| answer_doc_count mean | 2.750779 | 12.309538 | -9.558760 |

Interpretation:

- Group A is not a recall-miss group; answer evidence is in top50 but dense score ranks it below top5.
- Group A's top1/top5 dense scores are close to Group B, but its best answer-evidence score is much lower.
- The key failure mode is a positive gap between top5 non-answer dense score and best answer-evidence dense score.
- For Group A, this gap quantifies how much reranker must overcome dense retriever ordering.

## Rank Count Snapshots

### Group A best answer rank top20

```json
[
  [
    6,
    76
  ],
  [
    7,
    48
  ],
  [
    8,
    44
  ],
  [
    10,
    35
  ],
  [
    9,
    31
  ],
  [
    11,
    31
  ],
  [
    12,
    24
  ],
  [
    14,
    23
  ],
  [
    15,
    21
  ],
  [
    18,
    18
  ],
  [
    13,
    18
  ],
  [
    17,
    17
  ],
  [
    20,
    16
  ],
  [
    22,
    15
  ],
  [
    19,
    15
  ],
  [
    27,
    14
  ],
  [
    26,
    13
  ],
  [
    32,
    12
  ],
  [
    21,
    12
  ],
  [
    30,
    11
  ]
]
```

### Group B best answer rank top20

```json
[
  [
    1,
    2308
  ],
  [
    2,
    459
  ],
  [
    3,
    226
  ],
  [
    4,
    149
  ],
  [
    5,
    108
  ]
]
```

### Group A examples with largest dense score gap

- sample_id: `sample-003505:step:1`
  step_index=1 baseline_reward=0.0 best_answer_rank=37 best_answer_score=0.796387 top5_nonanswer_score=0.906250 gap=0.109863
  query: director of Magnificent Desolation: Walking on the Moon 3D
  answer_targets: ['Ron Howard']

- sample_id: `sample-003238:step:2`
  step_index=2 baseline_reward=0.0 best_answer_rank=13 best_answer_score=0.796875 top5_nonanswer_score=0.886230 gap=0.089355
  query: Robert Dornhelm Austrian writer film-producer film-director
  answer_targets: ['Barbara Albert']

- sample_id: `sample-001339:step:3`
  step_index=3 baseline_reward=0.0 best_answer_rank=20 best_answer_score=0.815918 top5_nonanswer_score=0.898438 gap=0.082520
  query: What is the global GDP ranking of Rome as a city?
  answer_targets: ['eighth', 'Eighth']

- sample_id: `sample-000559:step:1`
  step_index=1 baseline_reward=0.0 best_answer_rank=42 best_answer_score=0.806641 top5_nonanswer_score=0.888672 gap=0.082031
  query: Who did George Harrison write Miss O'Dell about
  answer_targets: ['his wife, Pattie Boyd', 'Pattie Boyd']

- sample_id: `sample-001065:step:1`
  step_index=1 baseline_reward=1.0 best_answer_rank=38 best_answer_score=0.789062 top5_nonanswer_score=0.866211 gap=0.077148
  query: Into Another band origin
  answer_targets: ['yes']


### Group A near-boundary examples

- sample_id: `sample-002637:step:2`
  step_index=2 baseline_reward=0.0 best_answer_rank=6 best_answer_score=0.827637 top5_nonanswer_score=0.832031 gap=0.004395
  query: who sang rock in the usa song
  answer_targets: ['John Mellencamp']

- sample_id: `sample-004507:step:2`
  step_index=2 baseline_reward=1.0 best_answer_rank=7 best_answer_score=0.832520 top5_nonanswer_score=0.836914 gap=0.004395
  query: Biggest terrorist attacks in Philadelphia
  answer_targets: ['the 9/11 attacks', '9/11', 'September 11', 'September 11 attacks']

- sample_id: `sample-001873:step:2`
  step_index=2 baseline_reward=0.6666666666666666 best_answer_rank=6 best_answer_score=0.820312 top5_nonanswer_score=0.825195 gap=0.004883
  query: Yongle Emperor Yang Sanbao 5th Dalai Lama 1642
  answer_targets: ['In 1642']

- sample_id: `sample-003461:step:2`
  step_index=2 baseline_reward=1.0 best_answer_rank=6 best_answer_score=0.811035 top5_nonanswer_score=0.815918 gap=0.004883
  query: Charles Eastman Mary Wells Lawrence nationality
  answer_targets: ['yes']

- sample_id: `sample-002111:step:1`
  step_index=1 baseline_reward=0.0 best_answer_rank=7 best_answer_score=0.854492 top5_nonanswer_score=0.859863 gap=0.005371
  query: who played christian on one life to live
  answer_targets: ['David Fumero', 'Yorlin Madera']


### Group B examples with low answer-evidence score

- sample_id: `sample-003801:step:3`
  step_index=3 baseline_reward=1.0 best_answer_rank=2 best_answer_score=0.775391 top5_nonanswer_score=0.777344 gap=0.001953
  query: Paul Belverstone country of birth
  answer_targets: ['no']

- sample_id: `sample-003112:step:1`
  step_index=1 baseline_reward=1.0 best_answer_rank=3 best_answer_score=0.779785 top5_nonanswer_score=0.784180 gap=0.004395
  query: John Peers and Jared Palmer occupation
  answer_targets: ['professional tennis player']

- sample_id: `sample-000203:step:2`
  step_index=2 baseline_reward=1.0 best_answer_rank=1 best_answer_score=0.788086 top5_nonanswer_score=0.782715 gap=-0.005371
  query: The Five Cents Of Lavarede (1913 Film) country
  answer_targets: ['no']

- sample_id: `sample-001152:step:3`
  step_index=3 baseline_reward=0.0 best_answer_rank=1 best_answer_score=0.791016 top5_nonanswer_score=0.790039 gap=-0.000977
  query: Jon Amiel Kurt Gerron common
  answer_targets: ['film director']

- sample_id: `sample-003124:step:2`
  step_index=2 baseline_reward=0.0 best_answer_rank=4 best_answer_score=0.791504 top5_nonanswer_score=0.840332 gap=0.048828
  query: Justin Bartha birth date
  answer_targets: ['Andrew Rannells']
