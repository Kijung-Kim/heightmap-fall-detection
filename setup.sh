#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
EXTERNAL_ROOT="${REPO_ROOT}/external"
CLONE_EXTERNAL=0

print_help() {
  cat <<'EOF'
Usage: bash setup.sh [options]

Options:
  --clone-external       Clone Grounded-SAM-2 and ml-depth-pro into ./external
  --venv-dir PATH        Override the virtual environment path
  --external-root PATH   Override the external dependency directory
  -h, --help             Show this help message

Examples:
  bash setup.sh
  bash setup.sh --clone-external
  bash setup.sh --venv-dir /tmp/hmfd-venv --external-root /data/external
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clone-external)
      CLONE_EXTERNAL=1
      shift
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --external-root)
      EXTERNAL_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "[setup] Unknown argument: $1" >&2
      print_help
      exit 1
      ;;
  esac
done

echo "[setup] Repository root: ${REPO_ROOT}"
echo "[setup] Virtual environment: ${VENV_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[setup] python3 is required but not installed." >&2
  exit 1
fi

python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e "${REPO_ROOT}"

if [[ "${CLONE_EXTERNAL}" -eq 1 ]]; then
  mkdir -p "${EXTERNAL_ROOT}"

  if [[ ! -d "${EXTERNAL_ROOT}/Grounded-SAM-2/.git" ]]; then
    git clone https://github.com/IDEA-Research/Grounded-SAM-2.git "${EXTERNAL_ROOT}/Grounded-SAM-2"
  else
    echo "[setup] Grounded-SAM-2 already exists at ${EXTERNAL_ROOT}/Grounded-SAM-2"
  fi

  if [[ ! -d "${EXTERNAL_ROOT}/ml-depth-pro/.git" ]]; then
    git clone https://github.com/apple/ml-depth-pro.git "${EXTERNAL_ROOT}/ml-depth-pro"
  else
    echo "[setup] ml-depth-pro already exists at ${EXTERNAL_ROOT}/ml-depth-pro"
  fi
fi

cat <<EOF

[setup] Done.

Next steps:
1. Activate the environment:
   source "${VENV_DIR}/bin/activate"

2. Prepare external repositories if you did not use --clone-external:
   - Grounded-SAM-2
   - ml-depth-pro

3. Download the required checkpoints:
   - depth_pro.pt
   - sam2.1_hiera_large.pt (or another compatible SAM2 checkpoint)

4. Generate height maps:
   python scripts/generate_heightmaps.py --help

5. Train the model:
   python scripts/train.py --help

EOF
