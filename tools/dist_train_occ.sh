#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 CONFIG GPUS [TRAIN_OPTIONS...]" >&2
    exit 2
fi

CONFIG="$1"
GPUS="$2"
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
SCRIPT_DIR=$(dirname "$0")

PYTHONPATH="$SCRIPT_DIR/..:${PYTHONPATH:-}" \
python -m torch.distributed.run \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    "$SCRIPT_DIR/train_occ.py" \
    "$CONFIG" \
    --seed 0 \
    --launcher pytorch "${@:3}"
