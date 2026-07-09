# AIR End-Point Dense Rank Score Analysis

- run_id: `260708-170127-dense_rank_endpoint_score_analysis`
- generated_at: `2026-07-08T17:01:51`
- source_manifest: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/llm_reranker_branch_train_set/260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_end_point_top50_top5_short_reason/manifest.json`
- source_dataset_jsonl: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/llm_reranker_branch_train_set/260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_end_point_top50_top5_short_reason/dataset.jsonl`
- prompt_template_version: `cosearch_rerank_topm_v1_short_reason_fixed_example`
- candidate_top_n: `50`
- visible_top_m: `5`
- answer_hit_rule: `cosearch_boundary_normalized_phrase_match`
- excluded_answer_types: `['yes', 'no']`
- excluded_yes_no_count: `315`
- json_summary: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/pipelines/formal/agenticIterRag/dense_rank_analysis/reports/260708-170127-dense_rank_endpoint_score_analysis.summary.json`
- row_metrics_csv: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/pipelines/formal/agenticIterRag/dense_rank_analysis/reports/260708-170127-dense_rank_endpoint_score_analysis.row_metrics.csv`

## Analysis Objective

Compare dense E5 recall score properties for end-point search queries between:

- Group A: `top50_hit_top5_miss`, where top50 contains answer evidence but original dense top5 does not.
- Group B: `top5_hit`, where original dense top5 already contains answer evidence.
- Group C: `top50_miss`, where top50 contains no answer evidence.

The query object is the final search query of each AIR trajectory (`step_policy=end_point`).

## Group Counts

| group | count | ratio |
| --- | ---: | ---: |
| A top50_hit_top5_miss | 631 | 0.1319 |
| B top5_hit | 3005 | 0.6280 |
| C top50_miss | 1149 | 0.2401 |
| total | 4785 | 1.0000 |

## Group A: top50_hit_top5_miss

| metric | n | mean | std | min | p25 | p50 | p75 | p90 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top1_score | 631 | 0.858391 | 0.025462 | 0.800781 | 0.839355 | 0.858887 | 0.876953 | 0.892090 | 0.899902 | 0.933594 |
| top5_mean_score | 631 | 0.847618 | 0.023412 | 0.792969 | 0.830273 | 0.848535 | 0.865625 | 0.878613 | 0.885156 | 0.915625 |
| top5_min_score | 631 | 0.839933 | 0.022842 | 0.788086 | 0.823242 | 0.840332 | 0.857422 | 0.868652 | 0.877441 | 0.903809 |
| top50_mean_score | 631 | 0.824662 | 0.020316 | 0.782158 | 0.809717 | 0.823008 | 0.840430 | 0.851758 | 0.857998 | 0.885225 |
| top50_score_range | 631 | 0.044015 | 0.016694 | 0.013184 | 0.031250 | 0.041992 | 0.054199 | 0.066895 | 0.075195 | 0.111328 |
| answer_doc_count | 631 | 2.892235 | 3.077105 | 1.000000 | 1.000000 | 2.000000 | 4.000000 | 6.000000 | 9.000000 | 22.000000 |
| best_answer_score | 631 | 0.828233 | 0.022692 | 0.779297 | 0.811035 | 0.825195 | 0.844727 | 0.858398 | 0.866699 | 0.890137 |
| best_answer_rank | 631 | 17.714739 | 11.745091 | 6.000000 | 8.000000 | 14.000000 | 24.000000 | 37.000000 | 43.000000 | 50.000000 |
| best_top5_nonanswer_score | 631 | 0.858391 | 0.025462 | 0.800781 | 0.839355 | 0.858887 | 0.876953 | 0.892090 | 0.899902 | 0.933594 |
| score_gap_top5_nonanswer_minus_answer | 631 | 0.030159 | 0.015563 | 0.004395 | 0.018555 | 0.026367 | 0.039062 | 0.050781 | 0.060059 | 0.109863 |

## Group B: top5_hit

| metric | n | mean | std | min | p25 | p50 | p75 | p90 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top1_score | 3005 | 0.870161 | 0.024339 | 0.784180 | 0.854004 | 0.872559 | 0.887207 | 0.900879 | 0.907227 | 0.940918 |
| top5_mean_score | 3005 | 0.856344 | 0.022372 | 0.781250 | 0.841406 | 0.857910 | 0.872461 | 0.884082 | 0.891113 | 0.923633 |
| top5_min_score | 3005 | 0.846208 | 0.022458 | 0.778320 | 0.830566 | 0.847168 | 0.862793 | 0.874512 | 0.881348 | 0.913086 |
| top50_mean_score | 3005 | 0.827779 | 0.019933 | 0.769541 | 0.813281 | 0.827764 | 0.842461 | 0.853789 | 0.860137 | 0.889814 |
| top50_score_range | 3005 | 0.054530 | 0.018142 | 0.014648 | 0.041016 | 0.052734 | 0.065918 | 0.078613 | 0.087402 | 0.131348 |
| answer_doc_count | 3005 | 10.172712 | 10.571984 | 1.000000 | 3.000000 | 6.000000 | 14.000000 | 26.000000 | 34.000000 | 50.000000 |
| best_answer_score | 3005 | 0.866975 | 0.025401 | 0.779785 | 0.850098 | 0.868652 | 0.885254 | 0.898926 | 0.906250 | 0.940918 |
| best_answer_rank | 3005 | 1.571381 | 1.053311 | 1.000000 | 1.000000 | 1.000000 | 2.000000 | 3.000000 | 4.000000 | 5.000000 |
| best_top5_nonanswer_score | 2635 | 0.858123 | 0.024167 | 0.784180 | 0.841309 | 0.859375 | 0.875488 | 0.888672 | 0.896484 | 0.929199 |
| score_gap_top5_nonanswer_minus_answer | 2635 | -0.007774 | 0.018039 | -0.090820 | -0.019043 | -0.006836 | 0.003906 | 0.013184 | 0.019043 | 0.064453 |

## Group C: top50_miss

| metric | n | mean | std | min | p25 | p50 | p75 | p90 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top1_score | 1149 | 0.848168 | 0.026321 | 0.790039 | 0.828613 | 0.846191 | 0.867188 | 0.884277 | 0.894043 | 0.930176 |
| top5_mean_score | 1149 | 0.836539 | 0.023103 | 0.787695 | 0.819141 | 0.834082 | 0.851660 | 0.868652 | 0.878223 | 0.910547 |
| top5_min_score | 1149 | 0.828307 | 0.021771 | 0.783691 | 0.811035 | 0.825684 | 0.842773 | 0.858398 | 0.868164 | 0.903320 |
| top50_mean_score | 1149 | 0.814241 | 0.018566 | 0.771592 | 0.800479 | 0.811855 | 0.825059 | 0.839912 | 0.849229 | 0.885781 |
| top50_score_range | 1149 | 0.043309 | 0.018777 | 0.008789 | 0.028320 | 0.039551 | 0.055664 | 0.069824 | 0.078613 | 0.110352 |
| answer_doc_count | 1149 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| best_answer_score | 0 |  |  |  |  |  |  |  |  |  |
| best_answer_rank | 0 |  |  |  |  |  |  |  |  |  |
| best_top5_nonanswer_score | 1149 | 0.848168 | 0.026321 | 0.790039 | 0.828613 | 0.846191 | 0.867188 | 0.884277 | 0.894043 | 0.930176 |
| score_gap_top5_nonanswer_minus_answer | 0 |  |  |  |  |  |  |  |  |  |

## Key Comparisons

| metric | Group A top50_hit_top5_miss | Group B top5_hit | delta A-B |
| --- | ---: | ---: | ---: |
| top1_score mean | 0.858391 | 0.870161 | -0.011769 |
| top5_mean_score mean | 0.847618 | 0.856344 | -0.008725 |
| best_answer_score mean | 0.828233 | 0.866975 | -0.038743 |
| best_answer_rank mean | 17.714739 | 1.571381 | 16.143357 |
| answer_doc_count mean | 2.892235 | 10.172712 | -7.280478 |

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
    74
  ],
  [
    8,
    50
  ],
  [
    7,
    45
  ],
  [
    10,
    35
  ],
  [
    9,
    32
  ],
  [
    11,
    32
  ],
  [
    14,
    24
  ],
  [
    12,
    22
  ],
  [
    17,
    20
  ],
  [
    15,
    20
  ],
  [
    13,
    20
  ],
  [
    19,
    16
  ],
  [
    20,
    16
  ],
  [
    18,
    15
  ],
  [
    22,
    14
  ],
  [
    21,
    13
  ],
  [
    27,
    13
  ],
  [
    16,
    13
  ],
  [
    26,
    13
  ],
  [
    32,
    11
  ]
]
```

### Group B best answer rank top20

```json
[
  [
    1,
    2117
  ],
  [
    2,
    420
  ],
  [
    3,
    216
  ],
  [
    4,
    143
  ],
  [
    5,
    109
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

- sample_id: `sample-002626:step:2`
  step_index=2 baseline_reward=0.0 best_answer_rank=20 best_answer_score=0.819336 top5_nonanswer_score=0.896973 gap=0.077637
  query: when does dean and rory break up in gilmore girls season 5
  answer_targets: ['7']


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

- sample_id: `sample-002111:step:1`
  step_index=1 baseline_reward=0.0 best_answer_rank=7 best_answer_score=0.854492 top5_nonanswer_score=0.859863 gap=0.005371
  query: who played christian on one life to live
  answer_targets: ['David Fumero', 'Yorlin Madera']

- sample_id: `sample-003959:step:2`
  step_index=2 baseline_reward=0.0 best_answer_rank=9 best_answer_score=0.821289 top5_nonanswer_score=0.826660 gap=0.005371
  query: American singer banjo player guitarist Appalachian folk music The Moonshiner
  answer_targets: ['Kentucky']


### Group B examples with low answer-evidence score

- sample_id: `sample-003112:step:1`
  step_index=1 baseline_reward=1.0 best_answer_rank=3 best_answer_score=0.779785 top5_nonanswer_score=0.784180 gap=0.004395
  query: John Peers and Jared Palmer occupation
  answer_targets: ['professional tennis player']

- sample_id: `sample-001152:step:3`
  step_index=3 baseline_reward=0.0 best_answer_rank=1 best_answer_score=0.791016 top5_nonanswer_score=0.790039 gap=-0.000977
  query: Jon Amiel Kurt Gerron common
  answer_targets: ['film director']

- sample_id: `sample-003124:step:2`
  step_index=2 baseline_reward=0.0 best_answer_rank=4 best_answer_score=0.791504 top5_nonanswer_score=0.840332 gap=0.048828
  query: Justin Bartha birth date
  answer_targets: ['Andrew Rannells']

- sample_id: `sample-004127:step:1`
  step_index=1 baseline_reward=0.0 best_answer_rank=4 best_answer_score=0.794434 top5_nonanswer_score=0.800781 gap=0.006348
  query: Pascal Chaumeil country
  answer_targets: ['French']

- sample_id: `sample-004244:step:2`
  step_index=2 baseline_reward=0.0 best_answer_rank=4 best_answer_score=0.795410 top5_nonanswer_score=0.805664 gap=0.010254
  query: Okkupert Kalmar Union history
  answer_targets: ['1814']
