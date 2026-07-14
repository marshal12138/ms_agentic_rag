# SPAD Stage1 Rollout Audit

- Rollout manifest: `log/agenticIterRag/260711-220950-337984-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_em_teacher_backoff_dev/outputs/stages/train_agent/spad_rag/search_policy_rl/rollout_data/manifest.json`
- Manifest SHA256: `f0b98a8f01db4550123889dd2ed6ab894620aa07d9bc38e64a0146eb9735f8c3`
- Validation: **PASS**

| Scope | Rollouts | Groups | EM=1 | Complete answer | Evidence | Teacher calls | All-zero groups | Backoff nonconstant | Final nonconstant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| total | 4096 | 512 | 515 | 2943 | 4070 | 2967 | 373 | 138 | 262 |
