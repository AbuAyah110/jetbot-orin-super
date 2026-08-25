#!/usr/bin/env bash
# Stage D / F1: name-resolved SSS1629 sequential record + playback. See f1_alsa_baseline.py.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${ROOT}/scripts/bringup/f1_alsa_baseline.sh" "$@"
