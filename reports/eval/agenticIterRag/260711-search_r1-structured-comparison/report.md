# Search-R1 Structured Answer Offline Rescore

- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/structured_answer/260711a_search_r1_512_350/search_r1_structured.eval.parquet`
- Dataset SHA256: `ce01777aabcbfae4e48343b09fc76bb6f043f500177a8f51df039beea47453db`
- Structured denominator excludes rows marked ineligible.

## Per Run

| Run | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_run1 | 350 | 0.1143 | 0.1835 | 342 | 0.1170 | 0.1848 | 0.1179 |
| base_run2 | 350 | 0.1000 | 0.1770 | 342 | 0.1023 | 0.1772 | 0.1033 |
| base_run3 | 350 | 0.0971 | 0.1771 | 342 | 0.0994 | 0.1762 | 0.1004 |
| legacy_search_r1_run1 | 350 | 0.1314 | 0.1979 | 342 | 0.1316 | 0.1934 | 0.1347 |
| legacy_search_r1_run2 | 350 | 0.1400 | 0.2038 | 342 | 0.1345 | 0.1977 | 0.1377 |
| legacy_search_r1_run3 | 350 | 0.1371 | 0.1969 | 342 | 0.1374 | 0.1957 | 0.1406 |
| structured_search_r1_run1 | 350 | 0.1343 | 0.2021 | 342 | 0.1374 | 0.2009 | 0.1406 |
| structured_search_r1_run2 | 350 | 0.1429 | 0.2114 | 342 | 0.1462 | 0.2093 | 0.1494 |
| structured_search_r1_run3 | 350 | 0.1400 | 0.2067 | 342 | 0.1404 | 0.2025 | 0.1435 |

## Repeat Aggregate

| Model | Runs | Legacy EM | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|
| base | 3 | 0.1038 +/- 0.0092 | 0.1062 +/- 0.0094 | 0.1794 +/- 0.0047 | 0.1072 +/- 0.0094 |
| legacy_search_r1 | 3 | 0.1362 +/- 0.0044 | 0.1345 +/- 0.0029 | 0.1956 +/- 0.0021 | 0.1377 +/- 0.0029 |
| structured_search_r1 | 3 | 0.1390 +/- 0.0044 | 0.1413 +/- 0.0045 | 0.2042 +/- 0.0045 | 0.1445 +/- 0.0045 |
