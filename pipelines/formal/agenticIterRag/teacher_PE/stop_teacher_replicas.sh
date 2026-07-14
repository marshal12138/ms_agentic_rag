#!/usr/bin/env bash
set -euo pipefail

for port in 8067 8068 8069 8070; do
  docker rm -f "teacher_pe_glm47_${port}" >/dev/null 2>&1 || true
done
