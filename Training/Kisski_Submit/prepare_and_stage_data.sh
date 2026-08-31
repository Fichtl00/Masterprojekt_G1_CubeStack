#!/usr/bin/env bash
# prepare_and_stage_data.sh -- Lädt Basis-VLM + Datensatz von Hugging Face und
# konvertiert sie (LeRobot -> HDF5 -> RLDS) LOKAL (auf dieser Maschine, mit
# Internetzugang), dann rsync auf den KISSKI-Scratch-Storage. Läuft dafür
# genau denselben entrypoint.sh (Schritte 1+2) wie der Cluster-Job -- nur via
# `docker run` statt `apptainer run`, damit KEINE der schweren
# UnifoLM-VLA-Abhängigkeiten (tensorflow, mujoco, dlimp, ...) auf diesem Host
# installiert sein müssen.
#
# Grund fuer /scratch statt Projekt-/Home-Verzeichnis: grosse Datenmengen
# (Datensatz + Checkpoints, potenziell hunderte GB) gehoeren auf KISSKI auf
# /scratch/$USER/, nicht in $HOME oder das Projektverzeichnis (GWDG-Vorgabe).
# Der Transfer laeuft ueber den dedizierten Transfer-Node, nicht den Login-Node.
#
# Usage:
#   KISSKI_LOGIN_HOST=<username>@glogin10 \
#   HF_TOKEN=hf_... \
#   ./prepare_and_stage_data.sh
#
#   # Optional, falls abweichend von den Defaults:
#   KISSKI_TRANSFER_HOST=<username>@transfer.hpc.gwdg.de
#   KISSKI_SCRATCH_DIR=/scratch/<username>
#
# NOCH NICHT gegen einen echten KISSKI-Login-Node getestet.

set -euo pipefail

KISSKI_LOGIN_HOST="${KISSKI_LOGIN_HOST:?Setze KISSKI_LOGIN_HOST=<username>@glogin10}"
KISSKI_USER="${KISSKI_LOGIN_HOST%@*}"
KISSKI_TRANSFER_HOST="${KISSKI_TRANSFER_HOST:-${KISSKI_USER}@transfer.hpc.gwdg.de}"
KISSKI_SCRATCH_DIR="${KISSKI_SCRATCH_DIR:-/scratch/${KISSKI_USER}}"
IMAGE_NAME="${IMAGE_NAME:-unifolm-vla-kisski:latest}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-/tmp/unifolm_vla_kisski_data}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN muss gesetzt sein (Basis-VLM + Datensatz sind ggf. gated)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$LOCAL_DATA_DIR"

echo "==> 1/2 Download + Konvertierung lokal via Docker (SKIP_TRAIN=1) ..."
docker run --rm \
    -v "${LOCAL_DATA_DIR}:/data" \
    -v "${SCRIPT_DIR}:/scripts" \
    -e "HF_TOKEN=${HF_TOKEN}" \
    -e "HF_DATASET_REPO=${HF_DATASET_REPO:-unitreerobotics/G1_Dex3_BlockStacking_Dataset}" \
    -e "BASE_VLM_REPO=${BASE_VLM_REPO:-unitreerobotics/UnifoLM-VLM-Base}" \
    -e "SKIP_DOWNLOAD=0" \
    -e "SKIP_CONVERT=0" \
    -e "SKIP_TRAIN=1" \
    "$IMAGE_NAME"

echo "==> 2/2 rsync zum Transfer-Node ($KISSKI_TRANSFER_HOST) nach ${KISSKI_SCRATCH_DIR}/data/ ..."
ssh "$KISSKI_TRANSFER_HOST" "mkdir -p ${KISSKI_SCRATCH_DIR}/data"
rsync -avz --progress "${LOCAL_DATA_DIR}/" "${KISSKI_TRANSFER_HOST}:${KISSKI_SCRATCH_DIR}/data/"

echo "==> Fertig. DATA_DIR auf KISSKI (${KISSKI_SCRATCH_DIR}/data) ist bereit fuer SKIP_DOWNLOAD=1 (Default in kisski_submit.sh)."
