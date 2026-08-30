# Teleoperations-Umgebung `1_cube_stack` (G1 + Dex3, fixed base, Pink-IK)

Task-Gruppe innerhalb von `Simulation/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/`, gebaut zur Aufnahme neuer Demonstrationsdaten für das Cube-Stacking-Task per CloudXR/OpenXR-Handtracking.

## Herkunft

Portiert aus einer externen Referenz-Simulation (Gruppe-1-Repo, `g1_dex3_sim`), aber **nur** die Szenen-Objekte + Reset-Logik:

- Tisch als einfache Box statt Nucleus-`packing_table.usd`.
- 3 Würfel (rot/gelb/grün), randomisierter Reset mit disjunkten y-Bändern pro Würfel (verhindert Überlappung).
- Schwarzes Stapel-Band als Platzier-Landmarke (rein visuell, kein Kollisionskörper).

Kamera-Konfiguration (4 Kameras: `cam_left_high`, `cam_right_high`, `cam_left_wrist`, `cam_right_wrist`) stammt unverändert aus der bereits Teleop-getesteten `fixed_base_upper_body_ik_g1_env_cfg_CubeStack.py` (ROS-Convention, an `torso_link`/Wrist-Links montiert).

Roboter-Config `g1_dex3_cfg_1_cube_stack.py`: Dataset-Startpose (Frame 0 aus `unitreerobotics/G1_Dex3_BlockStacking_Dataset`) + eigene Aktuator-Gains.

## Stacking-Reihenfolge

Erfolgs-Check und Aufgabenbeschreibung: **rot unten → gelb Mitte → grün oben**.

## Iterativ angepasste Parameter (Nutzer-Feedback während Teleop-Tests)

| Parameter | Ausgangswert | Finaler Wert | Grund |
|---|---|---|---|
| XR-Anker (`anchor_pos.z`) | -0.35 | -0.25 | Robotpelvis 10 cm höher als Referenz-Robot → Spawnpunkt musste nachjustiert werden |
| Arm-Stiffness | 100 | 750 | Ursprungswert 30x schwächer als Referenz-Config → Roboter folgte IK-Ziel zu träge |
| Arm-Velocity-Limit | 20 | 25 | siehe oben |
| Tisch-/Würfelhöhe | Basis | +5 cm ggü. Zwischenstand | Ergonomie/Erreichbarkeit beim Teleoperieren |
| Würfelgröße | 5 cm | 4 cm | Griffigkeit für Dex3-Hand |
| Stapel-Band-Marker | 1 Balken, unrotiert | 1 Balken, 90° gedreht, halbe Größe | Bessere Sichtbarkeit/Platzierung |
| Würfel-Spawn vs. Ablagebereich | ungebremst | Rejection-Sampling-Sperrzone um Marker | Würfel sollen nicht im Ablagebereich spawnen |

## Aufnahme-Workflow

1. **Ohne Kameras aufnehmen** (schneller, kein Rendering-Overhead während Teleop):
   ```bash
   ./isaaclab.sh -p scripts/tools/record_demos.py \
     --device cuda \
     --task Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-1_cube_stack \
     --teleop_device handtracking \
     --dataset_file ./datasets/1_cube_stack/batch01/dataset_g1_1_cube_stack.hdf5 \
     --num_demos 15 \
     --enable_pinocchio \
     --headless
   ```
2. **Nachträglich rendern** (Kein Mimic nötig -- reines Action-Replay via `reset_to` + `env.step()`, siehe `scripts/tools/replay_render_hdf5.py`):
   ```bash
   ./isaaclab.sh -p scripts/tools/replay_render_hdf5.py \
     --device cpu \
     --headless \
     --enable_cameras \
     --task Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-1_cube_stack \
     --input_file ./datasets/1_cube_stack/batch01/dataset_g1_1_cube_stack.hdf5 \
     --output_file ./datasets/1_cube_stack/batch01/dataset_g1_1_cube_stack_annotated.hdf5 \
     --enable_pinocchio
   ```
   **Oder direkt mit Kameras aufnehmen** (einfach `--enable_cameras` zum Record-Befehl hinzufügen -- läuft während der Teleoperation spürbar schwerer, spart aber den Replay-Schritt).
3. **Konvertieren nach LeRobot + Push nach Hugging Face**:
   ```bash
   ./isaaclab.sh -p scripts/tools/convert_hdf5_to_Lerobot.py \
     --input_files ./datasets/1_cube_stack/batch01/dataset_g1_1_cube_stack_annotated.hdf5 \
     --output_dir ./datasets/1_cube_stack/lerobot_out \
     --language_instruction "Stack the cubes: red on the bottom, yellow in the middle, green on top." \
     --fps 50

   ./isaaclab.sh -p -m huggingface_hub.cli.hf upload <namespace>/<dataset-repo> ./datasets/1_cube_stack/lerobot_out . --repo-type dataset
   ```

## CloudXR/OpenXR-Setup

Docker-Container-Start (aus `IsaacLab/docker/`):
```bash
./docker/container.py start --files docker-compose.cloudxr-runtime.patch.yaml --env-file .env.cloudxr-runtime
```
Bei Problemen: die Container existieren i.d.R. schon (nach dem ersten Build) -- einfacher direkter Neustart reicht:
```bash
docker start isaac-lab-base
docker start docker-cloudxr-runtime-1
```
`container.py start` ruft `docker compose ... up --build` auf und **rebuildet das Image jedes Mal**, auch wenn nichts geändert wurde -- das kann je nach Build-Cache-Zustand über eine Stunde dauern.
