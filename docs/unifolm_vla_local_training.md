# UnifoLM-VLA lokales Training (Action-Head-Finetuning)

Finetuning von UnifoLM-VLA **from scratch** (kein vortrainierter G1-VLA-Checkpoint) auf `unitreerobotics/G1_Dex3_BlockStacking_Dataset`, mit **eingefrorenem VLM-Backbone** (`qwen_vl_interface`) -- trainiert wird nur der Action-Head + der übrige Framework-Anteil.

Details/Hintergrund/Debugging-Historie: [`unifolm_vla_scratch_finetuning_g1_dex3_blockstacking.md`](unifolm_vla_scratch_finetuning_g1_dex3_blockstacking.md).

## Trainingsbefehl

Datensatz muss vorab als RLDS unter `<oxe_data_root>/<data_mix>/1.0.0/` vorliegen (siehe `Training/UnifoLM-VLA/prepare_data/`).

```bash
cd Training/UnifoLM-VLA

accelerate launch \
  --config_file src/unifolm_vla/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  --mixed_precision bf16 \
  src/unifolm_vla/training/train_unifolm_vla.py \
  --config_yaml ./src/unifolm_vla/config/training/unifolm_vla_train.yaml \
  --framework.framework_py unifolm_vla \
  --framework.qwenvl.base_vlm /pfad/zu/UnifoLM-VLM-Base \
  --framework.qwenvl.model_type qwen2_5_vl \
  --framework.action_model.action_dim 28 \
  --framework.action_model.state_dim 28 \
  --framework.action_model.action_horizon 16 \
  --framework.action_model.future_action_window_size 15 \
  --datasets.vla_data.data_root_dir /pfad/zu/g1_dex3_blockstacking_rlds \
  --datasets.vla_data.data_mix g1_dex3_blockstacking \
  --datasets.vla_data.window_size 1 \
  --datasets.vla_data.per_device_batch_size 2 \
  --trainer.freeze_modules qwen_vl_interface \
  --trainer.max_train_steps 30000 \
  --trainer.shuffle_buffer_size 500 \
  --trainer.save_interval 2000 \
  --trainer.use_wrist_image True \
  --trainer.use_proprio True \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 500 \
  --trainer.learning_rate.base 4e-5 \
  --run_root_dir /pfad/zu/outputs_unifolm_vla \
  --run_id g1_dex3_blockstacking_scratch \
  --wandb_project unifolm_vla_g1_dex3 \
  --wandb_entity local
```

Fertiges Run-Script: [`Training/UnifoLM-VLA/scripts/run_scripts/run_unifolm_vla_train_g1_dex3.sh`](../Training/UnifoLM-VLA/scripts/run_scripts/run_unifolm_vla_train_g1_dex3.sh) -- Pfade darin auf die eigene Umgebung anpassen.

## Wichtigste Flags erklärt

| Flag | Bedeutung |
|---|---|
| `--trainer.freeze_modules qwen_vl_interface` | Friert das VLM-Backbone (Qwen2.5-VL) ein -- nur Action-Head + Rest des Frameworks werden trainiert. |
| `--framework.action_model.action_dim 28` | G1-Dex3-Aktionsraum: 2×7 Arm-DoF + 2×7 Dex3-Hand-DoF. |
| `--datasets.vla_data.per_device_batch_size 2` | Klein gehalten wegen GPU-Speicherlimit bei eingefrorenem Backbone + DeepSpeed ZeRO-2. |
| `--trainer.max_train_steps 30000` | Angelehnt an akzeptierten Checkpoint `steps_24000` (siehe Doku). |

## Monitoring

TensorBoard läuft parallel zu wandb (offline-Modus) auf Port 6007 -- siehe Ergänzung in `train_unifolm_vla.py`.

## Evaluation

- **Open-Loop:** [`Training/UnifoLM-VLA/scripts/tools/open_loop_eval_g1_dex3.py`](../Training/UnifoLM-VLA/scripts/tools/open_loop_eval_g1_dex3.py)
- **Closed-Loop:** siehe [`groot_vs_unifolm_vla_vergleich.md`](groot_vs_unifolm_vla_vergleich.md) für die Architektur (HTTP-Server + Isaac-Lab-Bridge) und [`cube_stack_eval_env.md`](cube_stack_eval_env.md) für die Eval-Umgebung.
