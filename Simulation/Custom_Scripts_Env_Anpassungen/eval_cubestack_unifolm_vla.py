# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop evaluation of the from-scratch UnifoLM-VLA finetune (g1_dex3_blockstacking,
checkpoint steps_24000) on the CubeStack-JointSpace task (G1 + Dex3, direct joint-space
control, no Pink-IK).

Same architecture as eval_referenz_korb_ausladen_agile.py (GR00T): ONE process drives both
env.step() and the eval loop, talking synchronously to a policy server on the host. Unlike
the Referenz scripts, the server here is UnifoLM-VLA's own FastAPI HTTP server
(deployment/model_server/run_real_eval_server.py), not a ZMQ GR00T server -- and the env's
observation/action are already in the model's training order (arm_hand_joint_pos_ordered /
upper_body_joint_pos, see fixed_base_upper_body_ik_g1_env_cfg_CubeStack_JointSpace.py), so no
joint-name remap is needed the way the Referenz agile script needed one for hands.

This replaces the async SHM bridge (cubestack_jointspace_unifolm_vla_bridge.py +
vla_dds_client.py) for closed-loop testing: that bridge hung indefinitely in scene
initialization with unbounded host RAM growth (~90GB before being killed), reproducible with
both --device cpu and --device cuda, and was never actually verified end-to-end (see
Anleitung/Masterprojekt/unifolm_vla_scratch_finetuning_g1_dex3_blockstacking.md, Section 8).

Terminal 1 - UnifoLM-VLA HTTP server (auf dem HOST):
    conda activate unifolm-vla
    cd /home/omniverse-2/unifolm-vla/unifolm-vla
    python deployment/model_server/run_real_eval_server.py \
        --ckpt_path /home/omniverse-2/Groot/outputs_unifolm_vla/g1_dex3_blockstacking_scratch/checkpoints/steps_24000_pytorch_model.pt \
        --vlm_pretrained_path /home/omniverse-2/UnifoLM-VLM-Base \
        --unnorm_key g1_dex3_blockstacking \
        --host 0.0.0.0 --port 8777

Terminal 2 - Evaluation (im Docker Container):
    cd /workspace/isaaclab
    ./isaaclab.sh -p scripts/tools/eval_cubestack_unifolm_vla.py \
        --task Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-CubeStack-JointSpace \
        --device cuda --num_episodes 5 --max_steps 500 \
        --headless --enable_cameras --enable_pinocchio \
        --save_video --video_dir /workspace/isaaclab/eval_videos_unifolm_vla

--enable_pinocchio bleibt noetig, weil das pick_place-Package beim Import auch die
Pink-IK-CubeStack-Envs mit-importiert (siehe Docstring der JointSpace-Env-Datei) -- diese Env
selbst nutzt kein Pink-IK.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UnifoLM-VLA Closed-Loop Eval - CubeStack (JointSpace)")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-CubeStack-JointSpace")
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--server_host", type=str, default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=8777)
parser.add_argument("--action_horizon", type=int, default=16)
parser.add_argument("--instruction", type=str, default="stack three block")
parser.add_argument("--task_name", type=str, default="g1_dex3_blockstacking")
parser.add_argument("--results_file", type=str, default="/workspace/isaaclab/eval_results_unifolm_vla.json")
parser.add_argument("--save_video", action="store_true", default=False, help="Speichert Video pro Episode")
parser.add_argument("--video_dir", type=str, default="/workspace/isaaclab/eval_videos_unifolm_vla")
parser.add_argument("--video_fps", type=int, default=25)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help=(
        "No Pink IK is used by this task, but isaaclab_tasks.manager_based.locomanipulation.pick_place "
        "imports sibling modules (the Pink-IK CubeStack envs) at package-import time, which fail to import "
        "without Pinocchio available -- keep this flag on, same as cubestack_jointspace_unifolm_vla_bridge.py."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json_numpy
json_numpy.patch()

import numpy as np
import requests
import torch

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401  (registers gym task ids)
import gymnasium as gym
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# The model was trained on 3 of the dataset's 4 cameras -- cam_right_high was deliberately
# left out (see Masterprojekt doc, Abschnitt 2: matches Unitree's own G1 pretraining, which
# also only ever used primary + left_wrist + right_wrist).
CAMERA_KEYS = ["cam_left_high", "cam_left_wrist", "cam_right_wrist"]


# ---------------------------------------------------------------------------
# HTTP client for run_real_eval_server.py
# ---------------------------------------------------------------------------
class UnifolmVlaHttpClient:
    _HEADERS = {"content-type": "application/json"}

    def __init__(self, host="127.0.0.1", port=8777, timeout_s=30.0):
        self.url = f"http://{host}:{port}/act"
        self.timeout_s = timeout_s

    def ping(self, retries=15, delay=3.0):
        for i in range(retries):
            try:
                # The server only exposes /act, so a real request doubles as the readiness
                # probe -- any HTTP response (even a 4xx/500) means the process is up.
                requests.post(
                    self.url, data=json_numpy.dumps({"observations": []}), headers=self._HEADERS, timeout=self.timeout_s
                )
                print("Server erreichbar!")
                return True
            except requests.exceptions.RequestException:
                print(f"Warte auf Server... ({i + 1}/{retries})")
                time.sleep(delay)
        return False

    def get_action(self, observation: dict) -> np.ndarray:
        payload = {"observations": [observation]}
        resp = requests.post(self.url, data=json_numpy.dumps(payload), headers=self._HEADERS, timeout=self.timeout_s)
        if resp.status_code != 200:
            raise RuntimeError(f"{resp.status_code} {resp.reason}: {resp.text[:500]}")
        action = json_numpy.loads(resp.text)
        if isinstance(action, str) and action == "error":
            raise RuntimeError("UnifoLM-VLA server returned 'error' -- see server log")
        action = np.asarray(action)  # (action_horizon, 28)
        return action


# ---------------------------------------------------------------------------
# Video speichern
# ---------------------------------------------------------------------------
def save_video(frames, episode, video_dir, fps=25, suffix=""):
    if not frames:
        return
    try:
        import imageio

        os.makedirs(video_dir, exist_ok=True)
        path = os.path.join(video_dir, f"episode_{episode:03d}{suffix}.mp4")
        imageio.mimwrite(path, frames, fps=fps, quality=8)
        print(f"  Video gespeichert: {path}")
    except Exception as e:
        print(f"  Video-Fehler: {e}")


# ---------------------------------------------------------------------------
# Observation aufbauen -- Env liefert Kamera-Tensoren als CHW float [0,1] (B,3,H,W) und den
# 28-dim Zustand bereits in exakter Trainings-Reihenfolge (arm_hand_joint_pos_ordered).
# Server erwartet HWC uint8 pro Bild-Key und den rohen (unnormalisierten) Zustand.
# ---------------------------------------------------------------------------
def cam_to_hwc_uint8(cam_tensor: torch.Tensor) -> np.ndarray:
    img = cam_tensor[0].cpu().numpy()  # (3, H, W), float 0..1
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return np.transpose(img, (1, 2, 0))  # (H, W, 3)


def build_observation(env, instruction: str, task_name: str) -> dict:
    obs_buf = env.obs_buf["policy"]
    state = obs_buf["arm_hand_joint_pos_ordered"][0].cpu().numpy()  # (28,), dataset order
    return {
        "full_image": cam_to_hwc_uint8(obs_buf["cam_left_high"]),
        "left_wrist_image": cam_to_hwc_uint8(obs_buf["cam_left_wrist"]),
        "right_wrist_image": cam_to_hwc_uint8(obs_buf["cam_right_wrist"]),
        "instruction": instruction,
        "state": state,
        "task_name": task_name,
    }


def make_grid_frame(env) -> np.ndarray:
    obs_buf = env.obs_buf["policy"]
    top = np.concatenate(
        [cam_to_hwc_uint8(obs_buf["cam_left_high"]), cam_to_hwc_uint8(obs_buf["cam_right_high"])], axis=1
    )
    bottom = np.concatenate(
        [cam_to_hwc_uint8(obs_buf["cam_left_wrist"]), cam_to_hwc_uint8(obs_buf["cam_right_wrist"])], axis=1
    )
    return np.concatenate([top, bottom], axis=0)


# ---------------------------------------------------------------------------
# Einzelne Episode
# ---------------------------------------------------------------------------
def run_episode(env, client, episode):
    env.reset()

    step = 0
    success = False
    success_step = -1
    t_start = time.perf_counter()
    frames = []

    print(f"  Episode {episode}: starte...", flush=True)

    if args_cli.save_video:
        frames.append(make_grid_frame(env))

    while step < args_cli.max_steps:
        observation = build_observation(env, args_cli.instruction, args_cli.task_name)

        try:
            action_chunk = client.get_action(observation)
        except Exception as e:
            print(f"  Fehler bei get_action (Step {step}): {e}")
            break

        for t in range(min(args_cli.action_horizon, len(action_chunk))):
            action = torch.tensor(action_chunk[t], dtype=torch.float32, device=env.device).unsqueeze(0)
            _obs, _reward, terminated, truncated, _info = env.step(action)
            step += 1

            if args_cli.save_video:
                frames.append(make_grid_frame(env))

            if step % 50 == 0:
                print(f"    Step {step}/{args_cli.max_steps}...", flush=True)

            if env.termination_manager.get_term("success").any() and not success:
                success = True
                success_step = step
                print(f"  Erfolg bei Step {step}!")

            if terminated.any() or truncated.any():
                break
        else:
            continue
        break

    duration = time.perf_counter() - t_start
    status = "ERFOLG" if success else "nicht erfolgreich"
    print(f"  Episode {episode}: {status} | {step} Steps | {duration:.1f}s")

    if args_cli.save_video:
        save_video(frames, episode, args_cli.video_dir, args_cli.video_fps, suffix="_grid")

    return {
        "episode": episode,
        "success": success,
        "num_steps": step,
        "duration_s": round(duration, 2),
        "success_step": success_step,
    }


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("UnifoLM-VLA (steps_24000) - CubeStack Closed-Loop Eval (JointSpace)")
    print("=" * 60)
    print(f"  Task:        {args_cli.task}")
    print(f"  Server:      {args_cli.server_host}:{args_cli.server_port}")
    print(f"  Episoden:    {args_cli.num_episodes}")
    print(f"  Max Steps:   {args_cli.max_steps}")
    print(f"  Action Hor.: {args_cli.action_horizon}")
    print(f"  Instruction: {args_cli.instruction!r}")
    print(f"  Video:       {'ja -> ' + args_cli.video_dir if args_cli.save_video else 'nein'}")
    print()

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    print(f"Verbinde mit UnifoLM-VLA Server {args_cli.server_host}:{args_cli.server_port}...")
    client = UnifolmVlaHttpClient(host=args_cli.server_host, port=args_cli.server_port, timeout_s=30.0)

    if not client.ping():
        print("Server nicht erreichbar. Abbruch.")
        env.close()
        simulation_app.close()
        return

    print()

    results = [run_episode(env, client, ep) for ep in range(1, args_cli.num_episodes + 1)]

    n_success = sum(1 for r in results if r["success"])
    success_rate = n_success / len(results) if results else 0.0

    print()
    print("=" * 60)
    print(f"Ergebnis: {n_success}/{len(results)} Erfolge")
    print(f"Success Rate: {success_rate:.1%}")
    print("=" * 60)

    summary = {
        "task": args_cli.task,
        "num_episodes": len(results),
        "num_success": n_success,
        "success_rate": success_rate,
        "episodes": results,
    }
    Path(args_cli.results_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args_cli.results_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Ergebnisse gespeichert: {args_cli.results_file}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
