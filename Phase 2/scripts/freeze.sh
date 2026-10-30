#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m pip freeze > "$ROOT/pip_freeze.txt"
printf 'Wrote %s\n' "$ROOT/pip_freeze.txt"
