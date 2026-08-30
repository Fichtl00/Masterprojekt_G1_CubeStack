# Masterprojekt G1 Cube Stacking

Vision-Language-Action-Finetuning (UnifoLM-VLA) für Unitree G1 + Dex3-Hand auf einer Cube-Stacking-Aufgabe in Isaac Lab -- von der Teleoperations-Datenaufnahme über LeRobot-Konvertierung bis zum lokalen Action-Head-Finetuning und der Open-/Closed-Loop-Evaluation.

## Überblick

- **Aufgabe:** G1 + Dex3-Hand stapelt drei Würfel in fester Reihenfolge (rot → gelb → grün) auf einem Tisch.
- **Datenaufnahme:** Teleoperation per CloudXR/OpenXR-Handtracking in Isaac Lab (Pink-IK-Steuerung).
- **Training:** UnifoLM-VLA, VLM-Backbone eingefroren, nur Action-Head trainiert ("from scratch" auf `unitreerobotics/G1_Dex3_BlockStacking_Dataset`, kein vortrainierter G1-Checkpoint).
- **Evaluation:** Open-Loop (MSE/MAE gegen Ground-Truth-Trajektorien) und Closed-Loop (Policy steuert die Simulation live über einen HTTP-Server).

## Schnellstart

```bash
# 1. Isaac Lab starten (siehe Simulation/IsaacLab/README.md für die volle Docker-Anleitung)
cd Simulation/IsaacLab
./isaaclab.sh -p scripts/tools/record_demos.py \
  --device cuda \
  --task Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-1_cube_stack \
  --teleop_device handtracking \
  --dataset_file ./datasets/1_cube_stack/batch01/dataset_g1_1_cube_stack.hdf5 \
  --num_demos 15 --enable_pinocchio --headless

# 2. UnifoLM-VLA lokal trainieren (siehe docs/unifolm_vla_local_training.md)
cd ../../Training/UnifoLM-VLA
bash scripts/run_scripts/run_unifolm_vla_train_g1_dex3.sh
```

## Dokumentation

| Thema | Datei |
|---|---|
| Teleop-Umgebung `1_cube_stack` (Portierung, Tuning, Aufnahme-Workflow) | [`Dokumentation/Simulation/1_cube_stack_teleop_env.md`](Dokumentation/Simulation/1_cube_stack_teleop_env.md) |
| Cube-Stack-Eval-Umgebung (Beobachtungs-/Aktionsformat) | [`Dokumentation/Simulation/cube_stack_eval_env.md`](Dokumentation/Simulation/cube_stack_eval_env.md) |
| UnifoLM-VLA Finetuning von scratch (vollständige Vorgehensweise) | [`Dokumentation/Training/unifolm_vla_scratch_finetuning_g1_dex3_blockstacking.md`](Dokumentation/Training/unifolm_vla_scratch_finetuning_g1_dex3_blockstacking.md) |
| UnifoLM-VLA lokaler Trainingsbefehl (Action-Head-Finetuning) | [`Dokumentation/Training/unifolm_vla_local_training.md`](Dokumentation/Training/unifolm_vla_local_training.md) |
| GR00T vs. UnifoLM-VLA -- Architekturvergleich | [`Dokumentation/Training/groot_vs_unifolm_vla_vergleich.md`](Dokumentation/Training/groot_vs_unifolm_vla_vergleich.md) |
| Eval-Videos/-Ergebnisse | [`Dokumentation/Training/eval_videos_unifolm_vla_action_head_only/`](Dokumentation/Training/eval_videos_unifolm_vla_action_head_only/) |
| Datensätze (Quellen + eigene Aufnahmen) | [`Data/README.md`](Data/README.md) |

## Struktur

```
Masterprojekt_G1_CubeStack/
├── Simulation/
│   ├── IsaacLab/                          # Isaac Lab Checkout (voll lauffähig)
│   ├── Custom_Scripts_Env_Anpassungen/    # Übersichts-Kopien: Env-Configs, Closed-Loop-Eval
│   └── Teleop/                            # Übersichts-Kopien: Aufnahme-/Replay-/Konvertierungs-Skripte
├── Training/
│   ├── UnifoLM-VLA/                       # UnifoLM-VLA Framework (Training, Eval, Isaac-Sim-Bridge)
│   └── Kisski_Submit/                     # Platzhalter -- KISSKI-Submit-Skripte folgen noch
├── Dokumentation/
│   ├── Simulation/                        # Doku zur Simulation-Seite
│   └── Training/                          # Doku zur Training-/Eval-Seite
├── Data/                                  # Verweise auf Datensätze (Hugging Face + lokale Pfade)
└── README.md
```

## Voraussetzungen (Kurzfassung)

- NVIDIA-GPU mit ausreichend VRAM (Finetuning + Isaac-Sim-Rendering), Docker mit NVIDIA Container Toolkit.
- Isaac Sim / Isaac Lab (siehe `Simulation/IsaacLab/README.md` für die offizielle Installationsanleitung).
- Für Teleoperation: CloudXR-fähiges Headset + `docker-cloudxr-runtime`.
- Für UnifoLM-VLA-Training: `accelerate` + DeepSpeed ZeRO-2, `UnifoLM-VLM-Base` als Basis-VLM.
- Hugging Face Account für Datensatz-/Modell-Up-/Downloads.

## Weiterführende Ressourcen

- Isaac Lab (offiziell): https://github.com/isaac-sim/IsaacLab
- UnifoLM-VLA (Unitree, offiziell): https://github.com/unitreerobotics/unifolm-vla
- Datensatz: https://huggingface.co/datasets/unitreerobotics/G1_Dex3_BlockStacking_Dataset
- Aufgenommene Demonstrationen (synthetisch/teleoperiert): https://huggingface.co/datasets/Fichtl00/Cube_Stacking_synth
