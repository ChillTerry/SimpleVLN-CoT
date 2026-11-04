#!/usr/bin/env bash
set -euo pipefail
SFT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(dirname -- "$SFT_DIR")"
SFT_BASE_PYTHON="${SFT_BASE_PYTHON:-/opt/ac2/bin/python}"
SFT_VENV="${SFT_VENV:-$REPO_ROOT/.venv-sft}"
command -v uv >/dev/null

if [[ ! -f "$SFT_VENV/pyvenv.cfg" ]]; then
    uv venv --python "$SFT_BASE_PYTHON" --system-site-packages "$SFT_VENV"
fi
if ! rg -q '^include-system-site-packages = true$' "$SFT_VENV/pyvenv.cfg"; then
    echo "SFT_VENV must inherit the PPU system site-packages" >&2
    exit 1
fi
# The lock is the complete overlay dependency closure, excluding the vendor stack.
uv pip install --python "$SFT_VENV/bin/python" --no-deps --require-hashes \
    -r "$SFT_DIR/requirements.lock"
echo "SFT environment ready: $SFT_VENV"
