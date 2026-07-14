# Teacher Prompt Replica Stability

## Fresh Top 5 three-repeat comparison

Each strategy was run three new times over all 237 cases on three different replicas. No response was reused: every included run has `cache_hits=0`. Accuracy columns are holdout mean [min, max]. Cost columns are means over the three full runs.

| Rank | Strategy | I precision | I recall | I F1 | Parse rate | Wall s / 237 | Mean request s | Avg completion tokens |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | question-tail evidence-only | 0.8556 [0.8400, 0.8750] | 0.8667 [0.8400, 0.9200] | 0.8606 [0.8400, 0.8846] | 1.0000 [1.0000, 1.0000] | 117.88 | 7.71 | 83.1 |
| 2 | question-tail with sub_query | 0.8426 [0.8077, 0.8800] | 0.8533 [0.8400, 0.8800] | 0.8478 [0.8235, 0.8800] | 1.0000 [1.0000, 1.0000] | 119.59 | 7.84 | 82.3 |
| 3 | title-free question-tail | 0.8494 [0.8333, 0.8750] | 0.8267 [0.8000, 0.8400] | 0.8378 [0.8163, 0.8571] | 1.0000 [1.0000, 1.0000] | 118.60 | 7.75 | 82.3 |
| 4 | short focused question-tail | 0.8832 [0.8400, 0.9048] | 0.7867 [0.7600, 0.8400] | 0.8307 [0.8261, 0.8400] | 0.9831 [0.9831, 0.9831] | 104.05 | 6.82 | 53.6 |
| 5 | evidence-only without tail | 0.8309 [0.7917, 0.8750] | 0.7867 [0.7600, 0.8400] | 0.8081 [0.7755, 0.8571] | 1.0000 [1.0000, 1.0000] | 105.87 | 6.99 | 82.8 |

The fresh three-repeat leader is `question-tail evidence-only` with mean holdout I precision/recall/F1 `0.8556/0.8667/0.8606`.

`short focused question-tail` is the fastest Top 5 candidate, but its lower I recall and non-perfect parse rate make it a speed-only option rather than the best accuracy/cost production choice. The balanced production candidate remains `question-tail evidence-only`.

## Cumulative historical stability

All entries below are one teacher call per sample. Values are holdout mean [min, max] across the independent runs completed before the fresh Top 5 comparison.

| Strategy | Runs | I precision | I recall | I F1 | Parse rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| question-tail evidence-only | 6 | 0.8545 [0.8077, 0.9200] | 0.8533 [0.8000, 0.9200] | 0.8530 [0.8163, 0.9200] | 1.0000 [1.0000, 1.0000] |
| question-tail with sub_query | 4 | 0.8160 [0.7692, 0.8800] | 0.8400 [0.8000, 0.8800] | 0.8276 [0.7843, 0.8800] | 1.0000 [1.0000, 1.0000] |
| title-free question-tail | 5 | 0.8392 [0.8077, 0.8750] | 0.8320 [0.8000, 0.8400] | 0.8354 [0.8163, 0.8571] | 1.0000 [1.0000, 1.0000] |
| short focused question-tail | 5 | 0.8862 [0.8400, 0.9091] | 0.8000 [0.7600, 0.8400] | 0.8403 [0.8085, 0.8511] | 0.9898 [0.9831, 1.0000] |
| evidence-only without tail | 2 | 0.8238 [0.8077, 0.8400] | 0.8400 [0.8400, 0.8400] | 0.8318 [0.8235, 0.8400] | 1.0000 [1.0000, 1.0000] |

The repeated-run spread is material even with temperature=0. Selection must use repeated-run means and cache-free requests rather than the best single replica result.
