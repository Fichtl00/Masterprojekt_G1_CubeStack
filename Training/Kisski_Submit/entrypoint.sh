#!/usr/bin/env bash
# entrypoint.sh -- Container-Entrypoint fuer UnifoLM-VLA-Finetuning auf KISSKI.
#
# Wird von kisski_submit.sh per --bind .../Training/Kisski_Submit:/scripts in den
# Container gemountet (nicht ins Image gebacken -- siehe Dockerfile-Kommentar).
# Erwartet: /data (Bind-Mount von DATA_DIR), Env-Variablen aus kisski_submit.sh.
#
# ACHTUNG: Noch NICHT auf echter KISSKI-Hardware getestet (adaptiert aus dem
# GR00T-Workflow des Kollegen, siehe kisski_submit.sh-Kommentare für Details zu
# offenen Punkten -- insbesondere der RLDS-Build-Schritt).

set -euo pipefail

cd /app/unifolm-vla

DATA_ROOT=/data
LEROBOT_SOURCE_DIR="${DATA_ROOT}/lerobot_source"
HDF5_DIR="${DATA_ROOT}/hdf5_out"
RLDS_ROOT="${DATA_ROOT}/g1_dex3_blockstacking_rlds"
BASE_VLM="${DATA_ROOT}/UnifoLM-VLM-Base"
OUTPUT_ROOT="${DATA_ROOT}/outputs_unifolm_vla"

# unitreerobotics/G1_Dex3_BlockStacking_Dataset's `main` branch ist LeRobot v3.0
# (Pfadschema mit `chunk_index`), die hier verwendete LeRobotDataset-Klasse
# erwartet aber v2.1 (`episode_chunk`/`episode_index`) -- ohne Pin bricht
# convert_lerobot_to_hdf5_g1_dex3.py mit `KeyError: 'chunk_index'` ab. Fix
# (schon einmal in diesem Projekt geloest, siehe
# Dokumentation/Training/unifolm_vla_scratch_finetuning_g1_dex3_blockstacking.md):
# auf die Commit-SHA des `v2.1`-Tags pinnen (nicht den String "v2.1" selbst --
# das haelt den Hub-Versions-Check bei jeder Instanziierung aktiv).
HF_DATASET_REVISION="${HF_DATASET_REVISION:-88d465cc0d73659d0899c74eef053a2f4c2a5cff}"

# ── Schritt 1: Basis-VLM + Datensatz von Hugging Face laden ──────────────────
if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
    echo "==> Schritt 1/3 -- Download Basis-VLM + Datensatz"
    [[ -d "$BASE_VLM" ]] || hf download "${BASE_VLM_REPO:-unitreerobotics/UnifoLM-VLM-Base}" --local-dir "$BASE_VLM"
    hf download "${HF_DATASET_REPO:-unitreerobotics/G1_Dex3_BlockStacking_Dataset}" \
        --repo-type dataset --revision "$HF_DATASET_REVISION" --local-dir "$LEROBOT_SOURCE_DIR"
else
    echo "==> Schritt 1/3 -- SKIP_DOWNLOAD=1, übersprungen"
fi

# ── Schritt 2: LeRobot -> HDF5 -> RLDS konvertieren ───────────────────────────
if [[ "${SKIP_CONVERT:-0}" != "1" ]]; then
    echo "==> Schritt 2/3 -- Konvertiere LeRobot -> HDF5"
    python prepare_data/convert_lerobot_to_hdf5_g1_dex3.py \
        --repo-id "${HF_DATASET_REPO:-unitreerobotics/G1_Dex3_BlockStacking_Dataset}" \
        --root "$LEROBOT_SOURCE_DIR" \
        --revision "$HF_DATASET_REVISION" \
        --output_dir "$HDF5_DIR" \
        --num-workers "${CONVERT_WORKERS:-8}"

    # rlds_dataset_g1_dex3.py liest den HDF5-Ordner aus einer FEST verdrahteten
    # Konstante (HDF5_DATA_DIR), nicht aus einer Env-Variable -- hier per sed auf
    # den tatsaechlichen Container-Pfad umgebogen (nicht getestet!).
    BUILDER_FILE="prepare_data/hdf5_to_rlds/rlds_dataset_g1_dex3/rlds_dataset_g1_dex3.py"
    sed -i "s#^HDF5_DATA_DIR = .*#HDF5_DATA_DIR = \"${HDF5_DIR}\"#" "$BUILDER_FILE"

    echo "==> Schritt 2/3 -- Baue RLDS-Datensatz (tfds build)"
    (cd prepare_data/hdf5_to_rlds/rlds_dataset_g1_dex3 && tfds build --data_dir "$RLDS_ROOT")
else
    echo "==> Schritt 2/3 -- SKIP_CONVERT=1, übersprungen"
fi

# ── Schritt 3: Training ───────────────────────────────────────────────────────
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    echo "==> Schritt 3/3 -- Starte Training (FREEZE_BACKBONE=${FREEZE_BACKBONE:-0})"

    FREEZE_ARGS=()
    if [[ "${FREEZE_BACKBONE:-0}" == "1" ]]; then
        FREEZE_ARGS=(--trainer.freeze_modules qwen_vl_interface)
    fi

    [[ -n "${WANDB_API_KEY:-}" ]] && export WANDB_API_KEY
    export WANDB_MODE="${WANDB_MODE:-offline}"   # Compute-Nodes ohne Internet -> immer offline

    accelerate launch \
        --config_file src/unifolm_vla/config/deepseeds/deepspeed_zero2.yaml \
        --num_processes "${NUM_GPUS:-4}" \
        --mixed_precision bf16 \
        src/unifolm_vla/training/train_unifolm_vla.py \
        --config_yaml ./src/unifolm_vla/config/training/unifolm_vla_train.yaml \
        --framework.framework_py unifolm_vla \
        --framework.qwenvl.base_vlm "$BASE_VLM" \
        --framework.qwenvl.model_type qwen2_5_vl \
        --framework.action_model.action_dim 28 \
        --framework.action_model.state_dim 28 \
        --framework.action_model.action_horizon 16 \
        --framework.action_model.future_action_window_size 15 \
        --datasets.vla_data.data_root_dir "$RLDS_ROOT" \
        --datasets.vla_data.data_mix "${DATA_MIX:-g1_dex3_blockstacking}" \
        --datasets.vla_data.window_size 1 \
        --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-2}" \
        "${FREEZE_ARGS[@]}" \
        --trainer.max_train_steps "${MAX_STEPS:-30000}" \
        --trainer.shuffle_buffer_size 500 \
        --trainer.save_interval "${SAVE_STEPS:-2000}" \
        --trainer.use_wrist_image True \
        --trainer.use_proprio True \
        --trainer.logging_frequency 50 \
        --trainer.eval_interval 500 \
        --trainer.learning_rate.base "${LEARNING_RATE:-4e-5}" \
        --run_root_dir "$OUTPUT_ROOT" \
        --run_id "${RUN_ID:-g1_dex3_blockstacking_full}" \
        --wandb_project "${WANDB_PROJECT:-unifolm_vla_g1_dex3}" \
        --wandb_entity local
else
    echo "==> Schritt 3/3 -- SKIP_TRAIN=1, übersprungen"
fi
