#!/usr/bin/env bash
# kisski_submit.sh -- SLURM-Job-Script fuer UnifoLM-VLA (volles Finetuning, VLM-
# Backbone NICHT eingefroren) auf der KISSKI-HPC.
#
# Warum KISSKI: siehe Dokumentation/Training/kisski_hpc_ausweichen.md -- lokal
# steht nur eine einzelne GPU mit begrenztem VRAM zur Verfuegung, ausreichend fuer
# das Action-Head-Finetuning (VLM-Backbone eingefroren), aber nicht fuer ein
# volles Finetuning inkl. Backbone (deutlich hoeherer Speicherbedarf fuer
# Optimizer-States + Aktivierungen ueber das gesamte Qwen2.5-VL-Backbone).
#
# Mechanik (SLURM-Direktiven, Token-Dateien, portable Pfadaufloesung, Apptainer-
# Bind-Muster) uebernommen/adaptiert aus dem GR00T-Workflow eines Kollegen
# (kisski_submit.sh, https://github.com/Docboter/projektarbeit_humanoider_roboter,
# Branch training-luca-IKR-IS6.0) -- dort fuer GR00T N1.6, hier auf unser eigenes
# UnifoLM-VLA-Image + Training/Kisski_Submit/entrypoint.sh umgeschrieben.
#
# NOCH NICHT auf echter KISSKI-Hardware getestet -- vor dem ersten produktiven
# Lauf mit kleinen MAX_STEPS/SKIP_*-Werten verifizieren (siehe README.md in
# diesem Ordner).
#
# Voraussetzungen (einmalig) -- Details + genaue Befehle: README.md in diesem Ordner.
# KORRIGIERT gegenueber einer frueheren Version dieses Skripts (siehe git-History):
# `/scratch/` auf KISSKI wurde am 2026-03-31 abgeschaltet -- alle Daten muessen auf
# dem VAST-Projekt-Storage liegen (/mnt/vast-kisski/projects/<projekt>/). Ausserdem
# laeuft `apptainer pull` direkt auf dem LOGIN-Node (nicht per `srun` auf einem
# Compute-Node) -- bestaetigt durch die aktuelle, datierte Doku eines Kollegen mit
# bereits produktivem KISSKI-Workflow.
#   1. Image LOKAL bauen + zu Docker Hub pushen, dann auf dem LOGIN-Node ziehen:
#      build_and_transfer_sif.sh.
#   2. Basis-VLM + Datensatz LOKAL herunterladen/konvertieren (Internet noetig)
#      und auf VAST-Projekt-Storage hochladen (nicht $HOME/Scratch -- letzteres
#      existiert nicht mehr): prepare_and_stage_data.sh.
#   3. Dieses Repo auf den Cluster klonen (git braucht Internet -- auf dem
#      Login-Node ausfuehren, NICHT im Compute-Job):
#        git clone --depth 1 https://github.com/Fichtl00/Masterprojekt_G1_CubeStack.git \
#            /mnt/vast-kisski/projects/<projekt>/gruppe2/repo
#   4. Tokens EINMALIG in Dateien hinterlegen (mode 600) -- KISSKI setzt
#      SBATCH_EXPORT=none auf dem Login-Node, das ueberstimmt --export=ALL,
#      daher werden Tokens robust aus Dateien statt aus der Env gelesen:
#        printf 'hf_DEIN_TOKEN\n'  > ~/.hf_token  && chmod 600 ~/.hf_token
#        printf 'DEIN_WANDB_KEY\n' > ~/.wandb_key && chmod 600 ~/.wandb_key
#      (Wird hier nur noch fuer optionales W&B-Logging + im Optional-Fall
#      SKIP_DOWNLOAD=0 gebraucht -- der Compute-Job selbst laedt per Default nichts
#      herunter, siehe Punkt 2.)
#      Danach genuegt zum Einreichen: sbatch kisski_submit.sh
#   5. Nach dem Lauf: Checkpoints vom Cluster zurueckholen: fetch_checkpoints.sh.
#
# Ueberschreibbare Variablen (als Inline-Prefix vor sbatch, MIT --export=ALL):
#   KISSKI_PROJECT_DIR, KISSKI_GROUP_SUBDIR, REPO_DIR, DATA_DIR, SIF_IMAGE
#   MAX_STEPS, SAVE_STEPS, GLOBAL_BATCH_SIZE, NUM_GPUS, LEARNING_RATE
#   FREEZE_BACKBONE (=1 -> wie lokal nur Action-Head, Default 0 = volles Finetuning)
#   HF_DATASET_REPO, BASE_VLM_REPO, RUN_ID, WANDB_PROJECT
#   SKIP_DOWNLOAD, SKIP_CONVERT, SKIP_TRAIN

#SBATCH --job-name=unifolm-vla-finetune
#SBATCH -p kisski
#SBATCH -G A100:4
#SBATCH -c 96
#SBATCH --mem=384G
#SBATCH -t 48:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --export=ALL

set -euo pipefail

# ── Portable Pfad-Auflösung ───────────────────────────────────────────────────
# Bewusst dupliziert statt in ein lib-Skript ausgelagert: SLURM kopiert das
# Batch-Skript vor der Ausfuehrung in sein Spool-Verzeichnis, darum zeigen $0 und
# BASH_SOURCE im Job NICHT mehr ins Repo.
#
# Storage: alles auf dem VAST-Projekt-Storage (persistente SSD, kein Scratch --
# `/scratch/` wurde am 2026-03-31 abgeschaltet). SIF-Image bevorzugt in $HOME/images
# (klein, pro Nutzer eh schon getrennt), sonst im Projektverzeichnis.
#
# KISSKI_PROJECT_DIR ist das GEMEINSAME Projektverzeichnis (z.B. kisski-humrob) --
# dort liegen ggf. schon Daten/Repos einer ANDEREN Gruppe (Gruppe 1, GR00T-Workflow).
# Alles, was diesem Projekt (UnifoLM-VLA, "Gruppe 2") gehoert, liegt bewusst in einem
# eigenen Unterordner (KISSKI_GROUP_SUBDIR), damit nichts mit Gruppe 1's data/,
# repo/, repo-groot/ kollidiert.
KISSKI_PROJECT_DIR="${KISSKI_PROJECT_DIR:-/mnt/vast-kisski/projects/<projekt>}"
KISSKI_GROUP_SUBDIR="${KISSKI_GROUP_SUBDIR:-gruppe2}"
KISSKI_GROUP_DIR="${KISSKI_GROUP_DIR:-$KISSKI_PROJECT_DIR/$KISSKI_GROUP_SUBDIR}"
KISSKI_SIF_DIR="${KISSKI_SIF_DIR:-$HOME/images}"
pick_sif() {
    local f
    for f in "$@"; do [[ -f "$f" ]] && { printf '%s\n' "$f"; return; }; done
    printf '%s\n' "$1"
}
# REPO_DIR: wohin das Repo geklont wurde (kann vom Default abweichen -- siehe
# Dein tatsaechlicher Clone-Pfad, per REPO_DIR=... vor sbatch --export=ALL setzen).
REPO_DIR="${REPO_DIR:-$KISSKI_GROUP_DIR/repo}"

# ── Konfiguration ─────────────────────────────────────────────────────────────
SIF_IMAGE="${SIF_IMAGE:-$(pick_sif \
    "$KISSKI_SIF_DIR/unifolm-vla-kisski.sif" \
    "$KISSKI_GROUP_DIR/images/unifolm-vla-kisski.sif")}"
DATA_DIR="${DATA_DIR:-$KISSKI_GROUP_DIR/data}"

if [[ ! -d "$(dirname "$DATA_DIR")" ]]; then
    echo "FEHLER: Elternverzeichnis von DATA_DIR nicht erreichbar: $(dirname "$DATA_DIR")" >&2
    echo "       /mnt/vast-kisski ist auf diesem Node moeglicherweise nicht gemountet." >&2
    exit 1
fi

# 4x A100, DeepSpeed ZeRO-2. GLOBAL_BATCH_SIZE muss durch NUM_GPUS teilbar sein.
MAX_STEPS="${MAX_STEPS:-30000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-4}"
PER_DEVICE_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NUM_GPUS))
LEARNING_RATE="${LEARNING_RATE:-4e-5}"
SAVE_STEPS="${SAVE_STEPS:-2000}"

# FREEZE_BACKBONE=0 (Default): volles Finetuning -- der eigentliche Grund fuer
# KISSKI. FREEZE_BACKBONE=1 erlaubt trotzdem einen Action-Head-only-Lauf mit
# groesserem Batch als lokal, zum Vergleich.
FREEZE_BACKBONE="${FREEZE_BACKBONE:-0}"

HF_DATASET_REPO="${HF_DATASET_REPO:-unitreerobotics/G1_Dex3_BlockStacking_Dataset}"
BASE_VLM_REPO="${BASE_VLM_REPO:-unitreerobotics/UnifoLM-VLM-Base}"
DATA_MIX="${DATA_MIX:-g1_dex3_blockstacking}"
RUN_ID="${RUN_ID:-g1_dex3_blockstacking_full}"
WANDB_PROJECT="${WANDB_PROJECT:-unifolm_vla_g1_dex3}"

# SKIP_DOWNLOAD=1 per Default: Compute-Nodes haben KEIN Internet (siehe
# Dokumentation/Training/kisski_hpc_ausweichen.md) -- Basis-VLM + Datensatz muessen
# VORHER auf DATA_DIR liegen (siehe prepare_and_stage_data.sh in diesem Ordner).
# SKIP_CONVERT bleibt an (0): reine lokale HDF5->RLDS-Konvertierung braucht kein
# Netz und darf im Job laufen.
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-1}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

# ── Tokens aus Dateien einlesen ───────────────────────────────────────────────
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.hf_token}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$HOME/.wandb_key}"

if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
    [[ -n "$HF_TOKEN" ]] && export HF_TOKEN && echo "    HF_TOKEN aus $HF_TOKEN_FILE gelesen."
fi
if [[ -z "${WANDB_API_KEY:-}" && -f "$WANDB_KEY_FILE" ]]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_KEY_FILE")"
    [[ -n "$WANDB_API_KEY" ]] && export WANDB_API_KEY && echo "    WANDB_API_KEY aus $WANDB_KEY_FILE gelesen."
fi

if [[ -z "${HF_TOKEN:-}" && "$SKIP_DOWNLOAD" != "1" ]]; then
    echo "FEHLER: HF_TOKEN ist nicht gesetzt. In \$HOME/.hf_token hinterlegen (siehe Kopf dieses Skripts)." >&2
    exit 1
fi

if [[ ! -f "$SIF_IMAGE" ]]; then
    echo "FEHLER: SIF-Image nicht gefunden: $SIF_IMAGE" >&2
    echo "Einmalig erstellen (siehe README.md in diesem Ordner):" >&2
    echo "    module load apptainer && mkdir -p \$HOME/images" >&2
    echo "    apptainer pull \$HOME/images/unifolm-vla-kisski.sif docker://<namespace>/unifolm-vla-kisski:latest" >&2
    exit 1
fi

if [[ ! -d "${REPO_DIR}/Training/Kisski_Submit" ]]; then
    echo "FEHLER: ${REPO_DIR}/Training/Kisski_Submit nicht gefunden." >&2
    echo "       Repo einmalig klonen: git clone --depth 1 https://github.com/Fichtl00/Masterprojekt_G1_CubeStack.git $REPO_DIR" >&2
    exit 1
fi

mkdir -p logs "$DATA_DIR"

# ── Laufumgebung anzeigen ─────────────────────────────────────────────────────
echo "==> SLURM Job: ${SLURM_JOB_ID:-local} auf $(hostname)"
echo "    SIF-Image:          $SIF_IMAGE"
echo "    DATA_DIR:           $DATA_DIR"
echo "    REPO_DIR:           $REPO_DIR"
echo "    MAX_STEPS:          $MAX_STEPS"
echo "    GLOBAL_BATCH_SIZE:  $GLOBAL_BATCH_SIZE  (per_device = $PER_DEVICE_BATCH_SIZE)"
echo "    NUM_GPUS:           $NUM_GPUS"
echo "    LEARNING_RATE:      $LEARNING_RATE"
echo "    FREEZE_BACKBONE:    $FREEZE_BACKBONE  (0 = volles Finetuning, Grund fuer KISSKI)"
echo "    HF_DATASET_REPO:    $HF_DATASET_REPO"
echo "    RUN_ID:             $RUN_ID"
echo "    SKIP_DOWNLOAD/CONVERT/TRAIN: $SKIP_DOWNLOAD / $SKIP_CONVERT / $SKIP_TRAIN"
echo ""

module load apptainer

# ── Container starten ────────────────────────────────────────────────────────
APPTAINER_ARGS=(
    --nv
    --bind "$DATA_DIR:/data"
    --bind "${REPO_DIR}/Training/Kisski_Submit:/scripts"
    --env "TMPDIR=/tmp"
    ${HF_TOKEN:+--env "HF_TOKEN=$HF_TOKEN"}
    --env "MAX_STEPS=$MAX_STEPS"
    --env "SAVE_STEPS=$SAVE_STEPS"
    --env "PER_DEVICE_BATCH_SIZE=$PER_DEVICE_BATCH_SIZE"
    --env "NUM_GPUS=$NUM_GPUS"
    --env "LEARNING_RATE=$LEARNING_RATE"
    --env "FREEZE_BACKBONE=$FREEZE_BACKBONE"
    --env "HF_DATASET_REPO=$HF_DATASET_REPO"
    --env "BASE_VLM_REPO=$BASE_VLM_REPO"
    --env "DATA_MIX=$DATA_MIX"
    --env "RUN_ID=$RUN_ID"
    --env "WANDB_PROJECT=$WANDB_PROJECT"
    --env "SKIP_DOWNLOAD=$SKIP_DOWNLOAD"
    --env "SKIP_CONVERT=$SKIP_CONVERT"
    --env "SKIP_TRAIN=$SKIP_TRAIN"
)
[[ -n "${WANDB_API_KEY:-}" ]] && APPTAINER_ARGS+=(--env "WANDB_API_KEY=$WANDB_API_KEY")
# Compute-Nodes haben kein Internet -- W&B immer im Offline-Modus betreiben.
APPTAINER_ARGS+=(--env "WANDB_MODE=offline")

apptainer run "${APPTAINER_ARGS[@]}" "$SIF_IMAGE"
