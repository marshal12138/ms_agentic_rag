#!/bin/bash
set -euo pipefail

cat >&2 <<'EOF'
ERROR: CoAgenticRetriever/scripts/train_coagentic_retriever_grpo.sh is retired.

This legacy training entry read ranker/model/top-k fields from the static tool
config. Those fields are now owned by the canonical Hydra chain:

  scripts/coagenticRetriever_v2/01_train_launcher.sh
  CoAgenticRetriever/config/experimental/ranker_base/ranker_contrastive.yaml

Use the v2 canonical launcher or the task scripts under:

  tasks/train_tasks/coAgenticRetriever/
EOF

exit 2
