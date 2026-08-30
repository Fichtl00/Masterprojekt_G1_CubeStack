# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay recorded actions from an HDF5 dataset with cameras enabled and re-export.

Kein Mimic (keine ManagerBasedRLMimicEnv/Subtask-Signale noetig) -- reines
Action-Replay: fuer jede Episode wird der Ausgangszustand geladen
(``env.reset_to(initial_state)``), dann werden die aufgezeichneten Actions
Schritt fuer Schritt erneut ausgefuehrt (identisch zu record_demos.py's
Live-Teleop-Loop, nur eben mit vorgegebenen statt live erzeugten Actions).
Mit ``--enable_cameras`` werden die Kamera-Sensoren dabei live gerendert und
landen -- im Gegensatz zur Original-Aufnahme (dort ohne ``--enable_cameras``,
daher schwarze Kamerabilder) -- als echte RGB-Frames in der Ausgabedatei.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a recorded HDF5 dataset with cameras enabled and re-export.")
parser.add_argument("--task", type=str, required=True, help="Name of the task (gym-registered env id).")
parser.add_argument("--input_file", type=str, required=True, help="Recorded HDF5 dataset to replay.")
parser.add_argument("--output_file", type=str, required=True, help="Output HDF5 file with re-rendered cameras.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio (required for Pink-IK envs).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import torch

import gymnasium as gym

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.utils.datasets import HDF5DatasetFileHandler

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"Input dataset file does not exist: {args_cli.input_file}")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.input_file)
    episode_count = dataset_file_handler.get_num_episodes()
    if episode_count == 0:
        print("No episodes found in the dataset.")
        return

    output_dir = os.path.dirname(args_cli.output_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_name = args_cli.task
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)
    env_cfg.env_name = env_name

    # Erfolgs-Check manuell auswerten (fuer set_success_to_episodes), Terminations sonst
    # deaktivieren -- der Replay soll die volle aufgezeichnete Aktionssequenz durchlaufen,
    # unabhaengig von z.B. cube_dropping/time_out.
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
    env_cfg.terminations = None

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    # EXPORT_ALL statt EXPORT_SUCCEEDED_ONLY: keine der 15 echten Teleop-Demos durch einen
    # strikten Erfolgs-Check verwerfen -- der Erfolgsstatus wird trotzdem pro Episode annotiert.
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    exported_count = 0
    with torch.inference_mode():
        for episode_index, episode_name in enumerate(dataset_file_handler.get_episode_names()):
            print(f"\nReplaying episode #{episode_index} ({episode_name})")
            episode = dataset_file_handler.load_episode(episode_name, env.device)
            initial_state = episode.data["initial_state"]
            actions = episode.data["actions"]

            env.sim.reset()
            env.recorder_manager.reset()
            env.reset_to(initial_state, None, is_relative=True)

            for action in actions:
                action_tensor = torch.Tensor(action).reshape([1, action.shape[0]])
                env.step(action_tensor)

            is_success = False
            if success_term is not None:
                is_success = bool(success_term.func(env, **success_term.params)[0])

            env.recorder_manager.set_success_to_episodes(
                None, torch.tensor([[is_success]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes()
            exported_count += 1
            print(f"\tExported (success={is_success}).")

    print(f"\nExported {exported_count} (out of {episode_count}) replayed episodes to {args_cli.output_file}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
