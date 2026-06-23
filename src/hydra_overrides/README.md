# Hydra YAML Overrides

Utilities for converting partial YAML files into Hydra dotlist overrides that
can be appended to training script command lines.

## YAML format

Use ordinary partial YAML with the same key hierarchy as the target Hydra
config:

```yaml
ranker_training:
  trajectory_selector:
    type: best_and_worst_f1
    top_k: 1
    bottom_n: 2
    min_final_reward: 0.0
```

The converter emits:

```text
++ranker_training.trajectory_selector.type=best_and_worst_f1
++ranker_training.trajectory_selector.top_k=1
++ranker_training.trajectory_selector.bottom_n=2
++ranker_training.trajectory_selector.min_final_reward=0.0
```

The `++` prefix is used so each key can either override an existing value or
create a new value.

## Python CLI

```bash
PY=/data04/envs/ms/ms_cosearch_official/bin/python
"${PY}" CoSearch_derevitives/src/hydra_overrides/yaml_to_dotlist.py \
  CoSearch_derevitives/scripts/coagenticRetriever_local/strategies_yaml/ranker_contrastive_new_sampling.yaml
```

Multiple YAML files are allowed. They are emitted in order, so later files have
higher Hydra precedence when the generated dotlist is appended to a command.

## Shell helper

Training scripts can source the helper and convert YAML files into an array:

```bash
source "${ROOT}/src/hydra_overrides/hydra_overrides.sh"

hydra_collect_yaml_override_files HYDRA_YAML_FILES \
  "${HYDRA_OVERRIDE_YAMLS:-}" \
  "${RANKER_STRATEGY_YAML:-}"

hydra_yaml_overrides_to_array HYDRA_YAML_ARGS "${PY}" "${HYDRA_YAML_FILES[@]}"

exec "${PY}" "${MAIN}" \
  ... \
  "${HYDRA_YAML_ARGS[@]}" \
  "${USER_EXTRA_ARGS[@]}" \
  "$@"
```

Recommended precedence:

```text
base Hydra config
< script defaults
< YAML override files
< explicit user CLI overrides
```

This means YAML strategy files can override script defaults, while command-line
arguments remain the highest-priority override surface.

## Scope

This helper supports value overrides only. It intentionally rejects YAML files
with a top-level `defaults:` section because Hydra config composition should be
handled by the target app's own config tree.
