#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 CONFIG CHECKPOINT GPUS [TEST_OPTIONS...]" >&2
    exit 2
fi

CONFIG="$1"
CHECKPOINT="$2"
GPUS="$3"
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
    "$SCRIPT_DIR/test_occ.py" \
    "$CONFIG" \
    "$CHECKPOINT" \
    --launcher pytorch \
    "${@:4}"
