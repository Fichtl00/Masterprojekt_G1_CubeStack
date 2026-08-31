#!/usr/bin/env bash
# prepare_and_stage_data.sh -- Lädt Basis-VLM + Datensatz von Hugging Face und
# konvertiert sie (LeRobot -> HDF5 -> RLDS) LOKAL (auf dieser Maschine, mit
# Internetzugang), dann rsync nach KISSKI. Läuft dafür genau denselben
# entrypoint.sh (Schritte 1+2) wie der Cluster-Job -- nur via `docker run`
# statt `apptainer run`, damit KEINE der schweren UnifoLM-VLA-Abhängigkeiten
# (tensorflow, mujoco, dlimp, ...) auf diesem Host installiert sein müssen.
#
# Grund: KISSKI-Compute-Nodes haben kein Internet (siehe
# Dokumentation/Training/kisski_hpc_ausweichen.md) -- der eigentliche
# Trainings-Job (kisski_submit.sh) läuft daher mit SKIP_DOWNLOAD=1 und erwartet
# Basis-VLM + RLDS-Datensatz bereits fertig auf dem Cluster-Projektspeicher.
#
# Usage:
#   KISSKI_HOST=<username>@glogin-gpu.hpc.gwdg.de \
#   KISSKI_PROJECT_DIR=/mnt/vast-kisski/projects/<projekt> \
#   HF_TOKEN=hf_... \
#   ./prepare_and_stage_data.sh
#
# NOCH NICHT gegen einen echten KISSKI-Login-Node getestet.

set -euo pipefail

KISSKI_HOST="${KISSKI_HOST:?Setze KISSKI_HOST=<username>@glogin-gpu.hpc.gwdg.de}"
KISSKI_PROJECT_DIR="${KISSKI_PROJECT_DIR:-/mnt/vast-kisski/projects/kisski-humrob}"
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

echo "==> 2/2 rsync nach $KISSKI_HOST:${KISSKI_PROJECT_DIR}/data/ ..."
ssh "$KISSKI_HOST" "mkdir -p ${KISSKI_PROJECT_DIR}/data"
rsync -avz --progress "${LOCAL_DATA_DIR}/" "${KISSKI_HOST}:${KISSKI_PROJECT_DIR}/data/"

echo "==> Fertig. DATA_DIR auf KISSKI ist bereit fuer SKIP_DOWNLOAD=1 (Default in kisski_submit.sh)."
