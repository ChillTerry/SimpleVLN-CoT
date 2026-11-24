#!/usr/bin/env bash
# Start a Ray head with configurable log/spill directories and disk threshold.
set -euo pipefail

export RAY_TMPDIR="${RAY_LOG_DIR:-${RAY_TMPDIR:-/data/simplevln-ray}}"
RAY_SPILL_DIR="${RAY_SPILL_DIR:-${RAY_TMPDIR}/spill}"
RAY_LOCAL_FS_CAPACITY_THRESHOLD="${RAY_LOCAL_FS_CAPACITY_THRESHOLD:-0.99}"

exec "${RAY_BIN:-ray}" start --head \
  "--temp-dir=${RAY_TMPDIR}" \
  "--object-spilling-directory=${RAY_SPILL_DIR}" \
  "--system-config={\"local_fs_capacity_threshold\":${RAY_LOCAL_FS_CAPACITY_THRESHOLD}}" \
  "$@"
