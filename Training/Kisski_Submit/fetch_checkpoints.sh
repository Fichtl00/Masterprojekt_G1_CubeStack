#!/usr/bin/env bash
# fetch_checkpoints.sh -- Holt Checkpoints nach einem KISSKI-Trainingslauf per
# rsync vom Cluster-Projektspeicher auf diese Maschine und pusht sie optional
# (mit funktionierendem Internetzugang HIER, nicht auf dem Compute-Node) nach
# Hugging Face.
#
# Usage:
#   KISSKI_HOST=<username>@glogin-gpu.hpc.gwdg.de \
#   KISSKI_PROJECT_DIR=/mnt/vast-kisski/projects/<projekt> \
#   RUN_ID=g1_dex3_blockstacking_full \
#   ./fetch_checkpoints.sh
#
#   # Optional zusaetzlich nach HF pushen:
#   HF_UPLOAD_REPO=<namespace>/unifolm-vla-g1-dex3-full ./fetch_checkpoints.sh
#
# NOCH NICHT gegen einen echten KISSKI-Login-Node getestet.

set -euo pipefail

KISSKI_HOST="${KISSKI_HOST:?Setze KISSKI_HOST=<username>@glogin-gpu.hpc.gwdg.de}"
KISSKI_PROJECT_DIR="${KISSKI_PROJECT_DIR:-/mnt/vast-kisski/projects/kisski-humrob}"
RUN_ID="${RUN_ID:-g1_dex3_blockstacking_full}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-./checkpoints_${RUN_ID}}"

mkdir -p "$LOCAL_OUTPUT_DIR"

echo "==> rsync Checkpoints von $KISSKI_HOST ..."
rsync -avz --progress \
    "${KISSKI_HOST}:${KISSKI_PROJECT_DIR}/data/outputs_unifolm_vla/${RUN_ID}/" \
    "${LOCAL_OUTPUT_DIR}/"

echo "==> Lokal verfügbar unter: $LOCAL_OUTPUT_DIR"

if [[ -n "${HF_UPLOAD_REPO:-}" ]]; then
    echo "==> Push nach Hugging Face: $HF_UPLOAD_REPO"
    hf upload "$HF_UPLOAD_REPO" "$LOCAL_OUTPUT_DIR" . --repo-type model
fi
