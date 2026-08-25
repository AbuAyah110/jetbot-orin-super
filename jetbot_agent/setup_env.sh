#!/usr/bin/env bash
# Thin wrapper — canonical script is repo-root setup_env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec sudo "${ROOT}/setup_env.sh"
