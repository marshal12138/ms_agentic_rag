#!/usr/bin/env bash
set -euo pipefail

ports=(8067 8068 8069 8070)
for port in "${ports[@]}"; do
  name="teacher_pe_glm47_${port}"
  running=$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null || echo false)
  if curl -fsS --max-time 3 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    ready=true
  else
    ready=false
  fi
  echo -e "$name\tport=$port\trunning=$running\tready=$ready"
done
