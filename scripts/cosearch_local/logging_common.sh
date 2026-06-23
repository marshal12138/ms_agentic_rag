#!/usr/bin/env bash

if [[ -z "${ROOT:-}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

source "${ROOT}/src/logs/report_system/logging_reports.sh"
