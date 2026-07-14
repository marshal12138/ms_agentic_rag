# SPAD Stage1 Rollout Audit

- Rollout manifest: `log/agenticIterRag/260711-103304-616277-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512/outputs/stages/train_agent/spad_rag/search_policy_rl/rollout_data/manifest.json`
- Manifest SHA256: `e5ef0c9d777bd2aabf11fc8d1a456b3ad9432085268f0d26ded8cc23d1dd5e44`
- Validation: **PASS**

| Scope | Rollouts | Groups | EM=1 | Complete answer | Evidence | Teacher calls | All-zero groups | Backoff nonconstant | Final nonconstant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| total | 4096 | 512 | 515 | 3012 | 4068 | 2995 | 377 | 167 | 291 |
