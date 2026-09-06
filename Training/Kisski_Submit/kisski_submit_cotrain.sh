#!/usr/bin/env bash
# kisski_submit_cotrain.sh -- Co-Training-Variante von kisski_submit.sh: mischt
# echte Trainingsdaten (unitreerobotics/G1_Dex3_BlockStacking_Dataset) mit
# synthetischen, per CloudXR/OpenXR teleoperierten Isaac-Lab-Aufnahmen
# (Fichtl00/Cube_Stacking_synth), um deren Einfluss aufs Finetuning zu testen.
#
# Setzt USE_COTRAIN=1 + sinnvolle Defaults und reicht dann an kisski_submit.sh
# weiter -- SLURM-Direktiven (#SBATCH-Zeilen), Pfad-/Token-Logik, Apptainer-Bind-
# Muster etc. bleiben zentral in kisski_submit.sh, nicht dupliziert.
#
# Voraussetzung: Fichtl00/Cube_Stacking_synth muss VORHER genauso wie der echte
# Datensatz gestaged sein -- prepare_and_stage_data.sh unterstuetzt das ueber
# dieselben USE_COTRAIN/COTRAIN_HF_REPO/COTRAIN_MIX_RATIO-Variablen:
#   USE_COTRAIN=1 KISSKI_LOGIN_HOST=... KISSKI_PROJECT_DIR=... HF_TOKEN=... \
#       ./prepare_and_stage_data.sh
#
# Usage:
#   REPO_DIR=... KISSKI_PROJECT_DIR=/mnt/vast-kisski/projects/<projekt> \
#   ./kisski_submit_cotrain.sh
#
#   # Anteil synthetisch anpassen (Default 0.25 = 25% synthetisch / 75% echt):
#   COTRAIN_MIX_RATIO=0.5 REPO_DIR=... KISSKI_PROJECT_DIR=... ./kisski_submit_cotrain.sh
#
#   # Erster Testlauf:
#   MAX_STEPS=10 REPO_DIR=... KISSKI_PROJECT_DIR=... ./kisski_submit_cotrain.sh
#
# NOCH NICHT auf echter KISSKI-Hardware getestet.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# kisski_submit.sh schreibt logs/ relativ zum sbatch-Aufrufverzeichnis (SLURM kann in
# #SBATCH-Zeilen keine Variablen auflösen) -- deshalb hierher wechseln, unabhängig
# davon, von wo dieses Wrapper-Skript selbst aufgerufen wurde.
cd "$SCRIPT_DIR"
mkdir -p logs

export USE_COTRAIN=1
export COTRAIN_HF_REPO="${COTRAIN_HF_REPO:-Fichtl00/Cube_Stacking_synth}"
export COTRAIN_MIX_RATIO="${COTRAIN_MIX_RATIO:-0.25}"

echo "==> Co-Training: ${COTRAIN_HF_REPO} (${COTRAIN_MIX_RATIO} synthetisch / $(python3 -c "print(1-${COTRAIN_MIX_RATIO})") echt)"

sbatch --export=ALL --job-name=unifolm-vla-cotrain ./kisski_submit.sh
