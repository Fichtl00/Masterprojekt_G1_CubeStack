# Data

Rohdaten/Datensätze liegen wegen ihrer Größe **nicht** in diesem Git-Repo, sondern auf Hugging Face bzw. lokal auf der Trainingsmaschine. Diese Datei listet, was wo liegt.

## Quell-Datensatz (Basis für das Finetuning)

- [`unitreerobotics/G1_Dex3_BlockStacking_Dataset`](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_BlockStacking_Dataset) -- echte Roboter-Demonstrationen, Basis für das UnifoLM-VLA-Scratch-Finetuning.

## Eigene Teleop-Aufnahmen (Isaac Lab, `1_cube_stack`)

- [`Fichtl00/Cube_Stacking_synth`](https://huggingface.co/datasets/Fichtl00/Cube_Stacking_synth) -- per CloudXR/OpenXR-Handtracking teleoperierte Cube-Stacking-Demonstrationen (LeRobot-Format, gerenderte Kamerabilder). Aufnahme-/Konvertierungs-Workflow siehe [`../Dokumentation/Simulation/1_cube_stack_teleop_env.md`](../Dokumentation/Simulation/1_cube_stack_teleop_env.md).

## Lokale Verzeichnisstruktur (auf der Trainingsmaschine, nicht Teil dieses Repos)

```
IsaacLab/datasets/1_cube_stack/
├── batch01/dataset_g1_1_cube_stack.hdf5              # Rohaufnahme (ohne Kameras)
├── batch01/dataset_g1_1_cube_stack_annotated.hdf5     # nach replay_render_hdf5.py (mit Kameras)
├── batch02/dataset_g1_1_cube_stack.hdf5               # Rohaufnahme (mit Kameras, live gerendert)
└── lerobot_all/                                       # LeRobot-Export, Basis für Data/-Push nach HF
```
