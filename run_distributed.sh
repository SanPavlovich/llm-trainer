#!/usr/bin/env bash
# Launch training via torchrun for single- or multi-GPU (DDP) runs.
#
# Number of processes, in priority order:
#   1. explicit nproc_per_node argument
#   2. ddp.dp_size from the config, if ddp.enabled: true
#   3. auto-detected GPU count (nvidia-smi)
#
# Usage:
#   ./run_train.sh [config_path] [nproc_per_node]
#
# Examples:
#   ./run_train.sh                                              # configs/medium_pretrain.yaml, ddp.dp_size or all visible GPUs
#   ./run_train.sh configs/medium_pretrain.yaml                  # explicit config, ddp.dp_size or all visible GPUs
#   ./run_train.sh configs/medium_pretrain.yaml 2                # explicit config, 2 processes (2 GPUs)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-configs/medium_pretrain.yaml}"

if [[ ! -f "${SCRIPT_DIR}/${CONFIG}" ]]; then
    echo "error: config not found: ${CONFIG}" >&2
    exit 1
fi

if [[ -n "${2:-}" ]]; then
    NPROC="$2"
else
    NPROC="$(cd "${SCRIPT_DIR}" && python3 -c "
from src.schemas import RunConfig
config = RunConfig.from_yaml('${CONFIG}')
print(config.ddp.dp_size if config.ddp.enabled else '')
")"
    if [[ -z "$NPROC" ]]; then
        NPROC="$(nvidia-smi -L 2>/dev/null | wc -l)"
        if [[ "$NPROC" -lt 1 ]]; then
            NPROC=1
        fi
    fi
fi

echo "config:         ${CONFIG}"
echo "nproc_per_node: ${NPROC}"

cd "${SCRIPT_DIR}"
torchrun --standalone --nproc_per_node="${NPROC}" train.py --config "${CONFIG}"
