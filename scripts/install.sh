#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${VENV:-${ROOT}/.venv}"

"${PYTHON_BIN}" -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install -e "${ROOT}[test]"
echo "Installed embodied-infer in ${VENV}"
echo "Try: ${VENV}/bin/embodied-infer sim"
