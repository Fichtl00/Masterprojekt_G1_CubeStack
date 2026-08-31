#!/usr/bin/env bash
# fetch_checkpoints.sh -- Holt Checkpoints nach einem KISSKI-Trainingslauf per
# rsync vom Scratch-Storage (ueber den Transfer-Node, nicht den Login-Node) auf
# diese Maschine und pusht sie optional (mit funktionierendem Internetzugang
# HIER, nicht auf dem Compute-Node) nach Hugging Face.
#
# Usage:
#   KISSKI_LOGIN_HOST=<username>@glogin10 \
#   RUN_ID=g1_dex3_blockstacking_full \
#   ./fetch_checkpoints.sh
#
#   # Optional zusaetzlich nach HF pushen:
#   HF_UPLOAD_REPO=<namespace>/unifolm-vla-g1-dex3-full ./fetch_checkpoints.sh
#
# NOCH NICHT gegen einen echten KISSKI-Login-Node getestet.

set -euo pipefail

KISSKI_LOGIN_HOST="${KISSKI_LOGIN_HOST:?Setze KISSKI_LOGIN_HOST=<username>@glogin10}"
KISSKI_USER="${KISSKI_LOGIN_HOST%@*}"
KISSKI_TRANSFER_HOST="${KISSKI_TRANSFER_HOST:-${KISSKI_USER}@transfer.hpc.gwdg.de}"
KISSKI_SCRATCH_DIR="${KISSKI_SCRATCH_DIR:-/scratch/${KISSKI_USER}}"
RUN_ID="${RUN_ID:-g1_dex3_blockstacking_full}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-./checkpoints_${RUN_ID}}"

mkdir -p "$LOCAL_OUTPUT_DIR"

echo "==> rsync Checkpoints vom Transfer-Node ($KISSKI_TRANSFER_HOST) ..."
rsync -avz --progress \
    "${KISSKI_TRANSFER_HOST}:${KISSKI_SCRATCH_DIR}/data/outputs_unifolm_vla/${RUN_ID}/" \
    "${LOCAL_OUTPUT_DIR}/"

echo "==> Lokal verfügbar unter: $LOCAL_OUTPUT_DIR"

if [[ -n "${HF_UPLOAD_REPO:-}" ]]; then
    echo "==> Push nach Hugging Face: $HF_UPLOAD_REPO"
    hf upload "$HF_UPLOAD_REPO" "$LOCAL_OUTPUT_DIR" . --repo-type model
fi
