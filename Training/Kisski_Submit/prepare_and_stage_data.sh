#!/usr/bin/env bash
# prepare_and_stage_data.sh -- Lädt Basis-VLM + Datensatz von Hugging Face und
# konvertiert sie (LeRobot -> HDF5 -> RLDS) LOKAL (auf dieser Maschine, mit
# Internetzugang), dann rsync auf den KISSKI VAST-Projekt-Storage. Läuft dafür
# genau denselben entrypoint.sh (Schritte 1+2) wie der Cluster-Job -- nur via
# `docker run` statt `apptainer run`, damit KEINE der schweren
# UnifoLM-VLA-Abhängigkeiten (tensorflow, mujoco, dlimp, ...) auf diesem Host
# installiert sein müssen.
#
# KORRIGIERT gegenueber einer frueheren Version dieses Skripts (siehe git-History):
# `/scratch/` auf KISSKI wurde am 2026-03-31 abgeschaltet -- bestaetigt durch die
# aktuelle, datierte Doku eines Kollegen mit bereits produktivem KISSKI-Workflow.
# Grosse Datenmengen (Datensatz + Checkpoints, potenziell hunderte GB) gehoeren
# stattdessen auf den GWDG VAST-Projekt-Storage (/mnt/vast-kisski/projects/<projekt>/).
# Der Transfer laeuft ueber den dedizierten Transfer-Node, nicht den Login-Node.
#
# KISSKI_PROJECT_DIR ist das GEMEINSAME Projektverzeichnis (z.B. kisski-humrob) --
# eine ANDERE Gruppe (Gruppe 1, GR00T-Workflow) hat dort ggf. schon eigene Daten/
# Repos liegen. Landet deshalb bewusst in einem eigenen Unterordner
# (KISSKI_GROUP_SUBDIR, Default "gruppe2"), nicht direkt in KISSKI_PROJECT_DIR.
#
# Usage:
#   KISSKI_LOGIN_HOST=<username>@glogin-gpu.hpc.gwdg.de \
#   KISSKI_PROJECT_DIR=/mnt/vast-kisski/projects/<projekt> \
#   HF_TOKEN=hf_... \
#   ./prepare_and_stage_data.sh
#
#   # Optional, falls abweichend von den Defaults:
#   KISSKI_TRANSFER_HOST=<username>@transfer.hpc.gwdg.de
#   KISSKI_GROUP_SUBDIR=gruppe2

set -euo pipefail

KISSKI_LOGIN_HOST="${KISSKI_LOGIN_HOST:?Setze KISSKI_LOGIN_HOST=<username>@glogin-gpu.hpc.gwdg.de}"
KISSKI_USER="${KISSKI_LOGIN_HOST%@*}"
KISSKI_TRANSFER_HOST="${KISSKI_TRANSFER_HOST:-${KISSKI_USER}@transfer.hpc.gwdg.de}"
KISSKI_PROJECT_DIR="${KISSKI_PROJECT_DIR:?Setze KISSKI_PROJECT_DIR=/mnt/vast-kisski/projects/<projekt>}"
KISSKI_GROUP_SUBDIR="${KISSKI_GROUP_SUBDIR:-gruppe2}"
KISSKI_GROUP_DIR="${KISSKI_GROUP_DIR:-$KISSKI_PROJECT_DIR/$KISSKI_GROUP_SUBDIR}"
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

echo "==> 2/2 rsync zum Transfer-Node ($KISSKI_TRANSFER_HOST) nach ${KISSKI_GROUP_DIR}/data/ ..."
ssh "$KISSKI_TRANSFER_HOST" "mkdir -p ${KISSKI_GROUP_DIR}/data"
rsync -avz --progress "${LOCAL_DATA_DIR}/" "${KISSKI_TRANSFER_HOST}:${KISSKI_GROUP_DIR}/data/"

echo "==> Fertig. DATA_DIR auf KISSKI (${KISSKI_GROUP_DIR}/data) ist bereit fuer SKIP_DOWNLOAD=1 (Default in kisski_submit.sh)."
