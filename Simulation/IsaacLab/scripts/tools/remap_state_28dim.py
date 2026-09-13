#!/usr/bin/env python3
"""Remap an already-recorded HDF5 dataset's ``robot_joint_pos`` from 43-dim
(unsortierte interne USD-Artikulationsreihenfolge, alle DOF inkl. Beine/Huefte)
auf 28-dim (nur die aktuierten Arm+Hand-Gelenke, sortiert exakt nach der
Action-Reihenfolge ``ALL_JOINTS_ORDERED``).

Hintergrund: der urspruengliche ObsTerm fuer ``robot_joint_pos`` filterte nicht
nach Gelenknamen (``SceneEntityCfg("robot")`` ohne ``joint_names``/``preserve_order``)
und lieferte deshalb alle 43 DOF in einer Reihenfolge, die von der Action-Reihenfolge
abweicht -- siehe Fix in ``fixed_base_upper_body_ik_g1_env_cfg_1_cube_stack.py``
(ObservationsCfg.PolicyCfg.robot_joint_pos).

Fuer bereits aufgezeichnete Datensaetze (Kameras schon live mitgerendert, echte
Teleop-Erfolgs-Flags schon vorhanden) ist eine komplette Neusimulation via
``replay_render_hdf5.py`` nicht noetig, um den State-Fehler zu beheben: die 28
benoetigten Gelenke sind in den bereits aufgezeichneten 43 Werten vollstaendig
enthalten, nur eben in falscher Reihenfolge/mit Ueberschuss. Ein reines
Array-Remapping (dieses Skript) reicht aus und ist deutlich schneller als eine
GPU-Neusimulation (die zudem durch Kontaktphysik-Nichtdeterminismus abweichende
Erfolgsergebnisse liefern kann).

Die Zielreihenfolge (43 -> 28) wird NICHT hartkodiert, sondern bei jedem Lauf aus
einer kurzen, echten Env-Instanz abgeleitet (robust gegen USD-/Artikulations-
Aenderungen).

Usage:
    ./isaaclab.sh -p scripts/tools/remap_state_28dim.py \
        --task Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-1_cube_stack \
        --input_file /path/to/dataset.hdf5 \
        --output_file /path/to/dataset_28dim.hdf5 \
        --enable_pinocchio --enable_cameras --headless
"""

import argparse
import shutil

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, help="Name of the task (gym-registered env id).")
parser.add_argument("--input_file", type=str, required=True, help="HDF5 dataset with 43-dim robot_joint_pos.")
parser.add_argument("--output_file", type=str, required=True, help="Output HDF5 with 28-dim robot_joint_pos.")
parser.add_argument("--enable_pinocchio", action="store_true", default=False, help="Required for Pink-IK envs.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py
import numpy as np

import gymnasium as gym

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401

from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex3_cfg_1_cube_stack import ALL_JOINTS_ORDERED
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    # Nur zum Auslesen der 43 Gelenknamen in interner Reihenfolge -- keine Replays,
    # kein Recorder noetig, ein einziger reset() reicht.
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task
    env_cfg.terminations = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    raw_joint_names = list(env.scene["robot"].joint_names)
    env.close()

    assert len(raw_joint_names) == 43, f"Expected 43 raw joints, got {len(raw_joint_names)}"
    idx = [raw_joint_names.index(name) for name in ALL_JOINTS_ORDERED]
    assert len(set(idx)) == 28

    shutil.copyfile(args_cli.input_file, args_cli.output_file)
    f = h5py.File(args_cli.output_file, "r+")
    for ep_name in f["data"].keys():
        obs = f["data"][ep_name]["obs"]
        old = np.array(obs["robot_joint_pos"])
        assert old.shape[1] == 43, f"{ep_name}: expected 43-dim raw robot_joint_pos, got {old.shape}"
        new = old[:, idx]
        del obs["robot_joint_pos"]
        obs.create_dataset("robot_joint_pos", data=new)
        print(f"{ep_name}: remapped {old.shape} -> {new.shape}")
    f.close()
    print(f"Done: remapped robot_joint_pos (43->28) in {args_cli.output_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
