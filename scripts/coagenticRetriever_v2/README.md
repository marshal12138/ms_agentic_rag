# CoAgenticRetriever v2 Scripts

This directory contains the canonical launcher entries for CoAgenticRetriever training and evaluation.

## Files

```text
00_start_dense_retriever_server.sh
01_train_launcher.sh
02_infer_launcher.sh
assets/
evaluate_coagentic_vllm.py
```

## Training

Training is driven by `01_train_launcher.sh`. Task scripts select a `main_run` config and overlays; the Python compiler resolves Hydra groups, resource config, tool config and runtime env before Bash starts services and training.

Example:

```bash
bash scripts/coagenticRetriever_v2/01_train_launcher.sh \
  --main_run_config=coAgenticRetriever_main
```

## Evaluation

Evaluation is driven by `02_infer_launcher.sh`. The launcher is config-compiled from eval runtime, budget and resource groups.

Canonical task shape:

```bash
bash scripts/coagenticRetriever_v2/02_infer_launcher.sh \
  --main_run_config=coAgenticRetriever_main \
  --EVAL_RUNTIME_CONFIG=coagentic_retriever_vllm \
  --EVAL_BUDGET_CONFIG=coagentic_retriever_aligned_budget \
  --RESOURCE_CONFIG=local_eval_4gpu_0_3 \
  --OVERLAY_YAML=tasks/eval_tasks/coAgenticRetriever/configs/eval_CAR_asy_labl_v0701a_npu_fix_overlay.yaml
```

The user-facing eval task name is `identity.eval_task_name` in the eval overlay. Runtime reports and audit env files use `EVAL_TASK_NAME` and `EVAL_TASK_SLUG`.

## Eval Config Groups

`CoAgenticRetriever/config/main_run/coAgenticRetriever_main.yaml` selects eval groups:

```yaml
eval_config_groups:
  eval_runtime: coagentic_retriever_vllm
  eval_budget: coagentic_retriever_aligned_budget
  resource: local_eval_4gpu_0_3
```

The runtime config owns evaluator/vLLM/retrieval/tool/artifact defaults. The budget config owns prompt, response and multi-turn limits. The resource config owns device layout and service start/stop behavior.

## Eval Outputs

For each eval run, the compiler writes audit files under the runtime log directory:

```text
<RUN_NAME>.eval_runtime_env.sh
<RUN_NAME>.env
<RUN_NAME>.eval_args.txt
<RUN_NAME>.eval_overlay_yamls.txt
<RUN_NAME>.eval_passthrough_args.txt
<RUN_NAME>.final_eval_config.yaml
<RUN_NAME>.final_eval_config.json
<RUN_NAME>.tool_config.yaml
```

The evaluator writes metrics, traces, summary and report files under `log/eval_res/<group>/<task_name>/` and `reports/eval/<group>/`.
