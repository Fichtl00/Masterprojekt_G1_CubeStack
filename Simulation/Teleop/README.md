# Teleoperation: Datenaufnahme-Skripte

Kopien der Skripte für den Aufnahme-Workflow (aufnehmen → nachträglich rendern → nach LeRobot konvertieren). **Lauffähig sind sie nur an ihrem Original-Ort** in `Simulation/IsaacLab/scripts/tools/`.

Vollständiger Workflow inkl. Befehlen: [`../../Dokumentation/Simulation/1_cube_stack_teleop_env.md`](../../Dokumentation/Simulation/1_cube_stack_teleop_env.md).

- `replay_render_hdf5.py` — spielt eine ohne Kameras aufgenommene HDF5-Episode erneut ab und rendert dabei die Kamera-Sensoren nach (kein Mimic nötig).
- `convert_hdf5_to_Lerobot.py` — konvertiert eine (annotierte) HDF5-Datei ins LeRobot-Format zum Push nach Hugging Face.
