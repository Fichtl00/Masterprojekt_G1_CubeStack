# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared-memory bridge between the plain IsaacLab `CubeStack_JointSpace` env and the
UniFolm-VLA `vla_dds_client.py`, WITHOUT going through `unitree_sim_isaaclab`'s
`sim_main.py` (that script is specific to the unitree_sim_isaaclab task/env framework and
is not applicable to a plain `isaaclab_tasks` gym env like ours).

Plays the same role `sim_main.py` plays for the unitree_sim_isaaclab + DDS pipeline: steps
the env, publishes camera frames + robot/hand state into the exact same shared-memory
segments `vla_dds_client.py` already reads, and applies whatever action it wrote back --
so `vla_dds_client.py` runs completely UNCHANGED against this env.

IMPORTANT -- despite the "dds" naming inherited from the real-robot deployment code, none
of this uses actual DDS (no unitree_sdk2py ChannelPublisher/Subscriber) -- it is plain
`multiprocessing.shared_memory`, the same kind of lightweight embedded-protocol approach
already used for the GR00T ZMQ client in eval_referenz_korb_ausladen.py. See
Anleitung/Masterprojekt/cube_stack_eval_env.md for the full investigation that led here
(hand-command gap, proprio-composition bug, hardcoded 3-camera reader -- all fixed in
vla_dds_client.py / tools/shared_memory_utils.py alongside this script).

Joint/camera conventions here MUST stay in sync with vla_dds_client.py -- see the
constants below, each cross-referenced to the matching constant on the client side.

CONFIRMED WORKING end-to-end (SHM connects, vla_dds_client.py runs continuous inference
against this bridge -- required `ipc: host` on the isaac-lab-base container, see
Anleitung/Masterprojekt/cube_stack_eval_env.md). WebRTC livestream (`--livestream 2`) for
live viewing hits an unrelated Isaac-Sim-internal crash (omni.services.livestream.nvcf vs.
torch/libcusparseLt.so.0) -- use --save_video instead (see below), which needs no extra
extensions and doesn't touch the working headless path.

Run this first (Terminal 1, inside the isaac-lab-base container), then vla_dds_client.py
(Terminal 2, in the unifolm-vla conda env, exactly as in Unitree_VLA_stack/README.md but
with --image_order pointed at these 4 camera names and --camera_color_space rgb since this
bridge publishes raw RGB, not BGR):

    ./isaaclab.sh -p scripts/tools/cubestack_jointspace_unifolm_vla_bridge.py \
        --task Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-CubeStack-JointSpace \
        --device cpu --enable_cameras --enable_pinocchio --headless \
        --save_video --video_dir /workspace/isaaclab/eval_videos

    python deployment/isaaclab_bridge/vla_dds_client.py \
        --ckpt_path /home/omniverse-2/UnifoLM-VLA-Base/checkpoints/pytorch_model.pt \
        --vlm_pretrained_path unitreerobotics/UnifoLM-VLM-Base \
        --instruction "stack the red cube on the yellow cube, then the green cube on top" \
        --device cuda --camera_color_space rgb \
        --image_order cam_left_high cam_right_high cam_left_wrist cam_right_wrist \
        --log_actions

--save_video writes one MP4 per episode (episode boundary = success or
termination/truncation) to --video_dir: top-down view (env.scene["cam_top_down"], added to
the env's SceneCfg for this purpose only -- NOT part of ObservationsCfg, so it doesn't
change what the policy sees) stacked above a row of the 4 actual policy cameras, so you can
compare what the robot did against what the model actually saw.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import threading
import time
from multiprocessing import shared_memory
from typing import Any, Dict, Optional

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IsaacLab <-> UniFolm-VLA shared-memory bridge (CubeStack, joint-space)")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-CubeStack-JointSpace")
parser.add_argument("--max_steps", type=int, default=0, help="0 = run until interrupted")
parser.add_argument("--action_timeout_s", type=float, default=10.0, help="Reset if no action arrives for this long")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help=(
        "No Pink IK is used by this task, but isaaclab_tasks.manager_based.locomanipulation.pick_place "
        "imports sibling modules (the Pink-IK CubeStack envs) at package-import time, which fail to import "
        "without Pinocchio available -- keep this flag on, same as the other pick_place tasks."
    ),
)
parser.add_argument(
    "--save_video",
    action="store_true",
    default=False,
    help="Save a grid video (top-down + 4 policy cameras) per episode -- avoids needing the WebRTC livestream.",
)
parser.add_argument("--video_dir", type=str, default="/workspace/isaaclab/eval_videos")
parser.add_argument("--video_fps", type=int, default=25)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401

import gymnasium as gym

import isaaclab_tasks  # noqa: F401  (registers gym task ids)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# ---------------------------------------------------------------------------
# Minimal, dependency-free reimplementation of unitree_sim_isaaclab's
# dds/sharedmemorymanager.py -- same wire format (4-byte timestamp + 4-byte length +
# JSON payload), so vla_dds_client.py's SharedMemoryManager instances can read/write
# these segments transparently. Not imported directly from unitree_sim_isaaclab to avoid
# depending on that repo being mounted/importable inside this container.
# ---------------------------------------------------------------------------


class SharedMemoryManager:
    def __init__(self, name: str, size: int = 3072):
        self.size = size
        self.lock = threading.RLock()
        try:
            self.shm = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            self.shm = shared_memory.SharedMemory(create=True, size=size, name=name)
            try:
                os.chmod(f"/dev/shm/{name}", 0o666)
            except Exception:
                pass

    def write_data(self, data: Dict[str, Any]) -> bool:
        try:
            with self.lock:
                payload = json.dumps(data).encode("utf-8")
                if len(payload) > self.size - 8:
                    print(f"[SharedMemoryManager] Data too large ({len(payload)} > {self.size - 8})")
                    return False
                timestamp = int(time.time()) & 0xFFFFFFFF
                self.shm.buf[0:4] = timestamp.to_bytes(4, "little")
                self.shm.buf[4:8] = len(payload).to_bytes(4, "little")
                self.shm.buf[8 : 8 + len(payload)] = payload
                return True
        except Exception as e:
            print(f"[SharedMemoryManager] write error: {e}")
            return False

    def read_data(self) -> Optional[Dict[str, Any]]:
        try:
            with self.lock:
                data_len = int.from_bytes(self.shm.buf[4:8], "little")
                if data_len == 0:
                    return None
                payload = bytes(self.shm.buf[8 : 8 + data_len])
                return json.loads(payload.decode("utf-8"))
        except Exception as e:
            print(f"[SharedMemoryManager] read error: {e}")
            return None


class SimpleImageHeader(ctypes.LittleEndianStructure):
    _fields_ = [
        ("timestamp", ctypes.c_uint64),
        ("height", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("channels", ctypes.c_uint32),
        ("image_name", ctypes.c_char * 16),
        ("data_size", ctypes.c_uint32),
        ("encoding", ctypes.c_uint32),  # always 0 (raw) here -- no cv2/JPEG dependency
        ("quality", ctypes.c_uint32),
    ]


class ImageWriter:
    """Writes raw (uncompressed) RGB frames -- run vla_dds_client.py with
    --camera_color_space rgb to match (this bridge does not do a BGR conversion)."""

    _SHM_SIZE_PER_IMAGE = 640 * 480 * 3 + 128

    def __init__(self):
        self._shms: Dict[str, shared_memory.SharedMemory] = {}

    def _get_shm(self, image_name: str) -> shared_memory.SharedMemory:
        shm_name = f"isaac_{image_name}_image_shm"
        if shm_name not in self._shms:
            try:
                self._shms[shm_name] = shared_memory.SharedMemory(name=shm_name)
            except FileNotFoundError:
                self._shms[shm_name] = shared_memory.SharedMemory(
                    create=True, size=self._SHM_SIZE_PER_IMAGE, name=shm_name
                )
                try:
                    os.chmod(f"/dev/shm/{shm_name}", 0o666)
                except Exception:
                    pass
        return self._shms[shm_name]

    def write_image(self, image_name: str, image: np.ndarray) -> None:
        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)
        height, width, channels = image.shape
        data_bytes = image.tobytes()

        header = SimpleImageHeader()
        header.timestamp = int(time.time() * 1000)
        header.height = height
        header.width = width
        header.channels = channels
        header.image_name = image_name.encode("utf-8")[:15].ljust(16, b"\x00")
        header.data_size = len(data_bytes)
        header.encoding = 0
        header.quality = 0

        shm = self._get_shm(image_name)
        header_size = ctypes.sizeof(SimpleImageHeader)
        total_size = header_size + header.data_size
        if total_size > shm.size:
            print(f"[ImageWriter] Not enough space for {image_name}: need {total_size}, available {shm.size}")
            return
        shm.buf[0:header_size] = ctypes.string_at(ctypes.byref(header), header_size)
        shm.buf[header_size : header_size + header.data_size] = data_bytes


# ---------------------------------------------------------------------------
# Joint conventions -- MUST stay in sync with vla_dds_client.py's ARM_JOINT_NAMES /
# ARM_JOINT_TARGET_INDICES / _RIGHT_HAND_DATASET_TO_SIM_ORDER.
# ---------------------------------------------------------------------------

CAMERA_KEYS = ["cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist"]

# G1JointIndex body order (29 = 12 legs + 3 waist + 14 arms), matching the real robot's
# motor layout and vla_dds_client.py::ARM_JOINT_TARGET_INDICES = range(15, 29).
G1_BODY_JOINT_NAMES_ORDERED = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
ARM_JOINT_TARGET_INDICES = list(range(15, 29))

# Dex3 hand joints in the SIM's own order (thumb->middle->index, SAME for both hands --
# matches action_provider_dds.py::{left,right}_hand_joint_mapping and
# dex3_state.py::get_robot_girl_joint_names). The training dataset's RIGHT hand order is
# asymmetric (thumb->index->middle) -- vla_dds_client.py converts between the two via
# _RIGHT_HAND_DATASET_TO_SIM_ORDER, which is self-inverse, so the same constant is reused
# here in both directions (dataset->sim when reading a hand command, sim->dataset when
# publishing hand state).
LEFT_HAND_JOINT_NAMES_SIM_ORDER = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
]
RIGHT_HAND_JOINT_NAMES_SIM_ORDER = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
]
RIGHT_HAND_DATASET_TO_SIM_ORDER = [0, 1, 2, 5, 6, 3, 4]  # self-inverse permutation


def get_ordered_joint_pos(robot, joint_names_ordered) -> np.ndarray:
    joint_ids = [robot.data.joint_names.index(name) for name in joint_names_ordered]
    return robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()


def cam_to_uint8_hwc(cam_tensor: torch.Tensor) -> np.ndarray:
    """Our env's camera observations are CHW float in [0, 1] (see get_cam_* in the env
    cfg) -- convert to HWC uint8 RGB for the shared-memory image writer."""
    img = cam_tensor[0].detach().cpu().numpy()  # (3, H, W)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return np.transpose(img, (1, 2, 0))  # (H, W, 3)


def resize_nn(image: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Dependency-free nearest-neighbor resize (no cv2/PIL needed elsewhere in this
    script, so avoid pulling them in just for the debug video)."""
    h, w = image.shape[:2]
    row_idx = (np.arange(new_h) * h / new_h).astype(np.int64)
    col_idx = (np.arange(new_w) * w / new_w).astype(np.int64)
    return image[row_idx][:, col_idx]


def cam_raw_to_uint8_hwc(cam_sensor) -> np.ndarray:
    """For cameras read directly off env.scene (not through the observation manager,
    e.g. the visualization-only top-down camera) -- raw sensor output is already HWC
    uint8, unlike the CHW float [0,1] tensors in obs_buf['policy']."""
    img = cam_sensor.data.output["rgb"][0]
    img = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    return img[:, :, :3].astype(np.uint8)


def make_grid_frame(top_down: np.ndarray, policy_cams: Dict[str, np.ndarray]) -> np.ndarray:
    """Row 1: top-down view (upscaled, prominent). Row 2: the 4 policy cameras the model
    actually sees, shrunk and concatenated horizontally, same width as row 1."""
    row1 = resize_nn(top_down, 960, 1280)
    small = [resize_nn(policy_cams[key], 240, 320) for key in CAMERA_KEYS]
    row2 = np.concatenate(small, axis=1)  # (240, 1280, 3)
    return np.concatenate([row1, row2], axis=0)  # (1200, 1280, 3)


def save_episode_video(frames, episode: int, video_dir: str, fps: int) -> None:
    if not frames:
        return
    try:
        import imageio

        os.makedirs(video_dir, exist_ok=True)
        path = os.path.join(video_dir, f"episode_{episode:03d}_grid.mp4")
        imageio.mimwrite(path, frames, fps=fps, quality=8)
        print(f"  Video saved: {path} ({len(frames)} frames)")
    except Exception as e:
        print(f"  Video error: {e}")


def publish_state(robot, robot_state_shm: SharedMemoryManager, dex3_state_shm: SharedMemoryManager) -> None:
    body_positions = get_ordered_joint_pos(robot, G1_BODY_JOINT_NAMES_ORDERED)
    robot_state_shm.write_data({"joint_positions": body_positions.tolist()})

    left_positions = get_ordered_joint_pos(robot, LEFT_HAND_JOINT_NAMES_SIM_ORDER)
    right_positions = get_ordered_joint_pos(robot, RIGHT_HAND_JOINT_NAMES_SIM_ORDER)
    dex3_state_shm.write_data(
        {
            "left_hand": {"positions": left_positions.tolist(), "velocities": [0.0] * 7, "torques": [0.0] * 7},
            "right_hand": {"positions": right_positions.tolist(), "velocities": [0.0] * 7, "torques": [0.0] * 7},
        }
    )


def read_action(
    command_shm: SharedMemoryManager, dex3_cmd_shm: SharedMemoryManager, robot
) -> Optional[np.ndarray]:
    """Reassembles the 28-dim action in the exact order our env's ActionsCfg expects
    (DEX3_ARM_HAND_JOINT_NAMES_ORDERED in the env file: 14 arm + 7 left-hand [dataset
    order] + 7 right-hand [dataset order]). vla_dds_client.py writes arm targets into
    dds_robot_cmd (29-motor array, indices 15-28) and hand targets into isaac_dex3_cmd
    (already in the sim's thumb/middle/index order for both hands -- left needs no
    reorder, right needs converting back to dataset order).

    Requires the ARM command to be present (returns None / keeps waiting otherwise). The
    HAND command is treated as optional: if isaac_dex3_cmd has never been written (e.g.
    the loaded checkpoint's action_dim is 14/arm-only rather than the full 28), fall back
    to holding the robot's CURRENT hand joint positions instead of blocking forever -- lets
    you at least observe/record arm behavior while sorting out whether the checkpoint is
    actually the 28-dim arm+hand one this env was built against."""
    body_cmd = command_shm.read_data()
    if not body_cmd or not body_cmd.get("motor_cmd") or len(body_cmd["motor_cmd"]) < 29:
        return None
    arm_values = np.asarray([body_cmd["motor_cmd"][i]["q"] for i in ARM_JOINT_TARGET_INDICES], dtype=np.float32)

    hand_cmd = dex3_cmd_shm.read_data()
    left_positions = (hand_cmd or {}).get("left_hand_cmd", {}).get("positions")
    right_positions = (hand_cmd or {}).get("right_hand_cmd", {}).get("positions")
    if left_positions and right_positions and len(left_positions) >= 7 and len(right_positions) >= 7:
        left_values = np.asarray(left_positions[:7], dtype=np.float32)  # sim order == dataset order for left
        right_sim_order = np.asarray(right_positions[:7], dtype=np.float32)
        right_dataset_order = right_sim_order[RIGHT_HAND_DATASET_TO_SIM_ORDER]
    else:
        # No hand command yet -- hold current hand pose instead of blocking on it.
        left_values = get_ordered_joint_pos(robot, LEFT_HAND_JOINT_NAMES_SIM_ORDER)
        right_sim_order = get_ordered_joint_pos(robot, RIGHT_HAND_JOINT_NAMES_SIM_ORDER)
        right_dataset_order = right_sim_order[RIGHT_HAND_DATASET_TO_SIM_ORDER]

    return np.concatenate([arm_values, left_values, right_dataset_order], axis=0)


def main() -> None:
    print("=" * 60)
    print("IsaacLab <-> UniFolm-VLA Shared-Memory Bridge (CubeStack, Joint-Space)")
    print("=" * 60)
    print(f"  Task: {args_cli.task}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]

    robot_state_shm = SharedMemoryManager("isaac_robot_state", 3072)
    dex3_state_shm = SharedMemoryManager("isaac_dex3_state", 1180)
    command_shm = SharedMemoryManager("dds_robot_cmd", 3072)
    dex3_cmd_shm = SharedMemoryManager("isaac_dex3_cmd", 1180)
    image_writer = ImageWriter()

    print("Publishing isaac_robot_state / isaac_dex3_state / camera SHMs, waiting for actions on")
    print("dds_robot_cmd / isaac_dex3_cmd. Start vla_dds_client.py now (see docstring for the exact command).")
    if args_cli.save_video:
        print(f"  Recording grid videos (top-down + 4 cams) to {args_cli.video_dir}")

    step = 0
    episode = 1
    frames: list = []
    last_action_time = time.time()

    def flush_video():
        if args_cli.save_video:
            save_episode_video(frames, episode, args_cli.video_dir, args_cli.video_fps)
            frames.clear()

    try:
        while args_cli.max_steps == 0 or step < args_cli.max_steps:
            obs_buf = env.obs_buf["policy"]
            policy_cam_frames = {key: cam_to_uint8_hwc(obs_buf[key]) for key in CAMERA_KEYS}
            for key, img in policy_cam_frames.items():
                image_writer.write_image(key, img)
            publish_state(robot, robot_state_shm, dex3_state_shm)

            if args_cli.save_video:
                top_down = cam_raw_to_uint8_hwc(env.scene["cam_top_down"])
                frames.append(make_grid_frame(top_down, policy_cam_frames))

            action = read_action(command_shm, dex3_cmd_shm, robot)
            if action is None:
                if time.time() - last_action_time > args_cli.action_timeout_s:
                    print(f"  No action received for {args_cli.action_timeout_s}s, still waiting...")
                    last_action_time = time.time()
                time.sleep(0.02)
                continue
            last_action_time = time.time()

            action_tensor = torch.tensor(action, dtype=torch.float32, device=env.device).unsqueeze(0)
            _obs, _reward, terminated, truncated, _info = env.step(action_tensor)
            step += 1

            if step % 100 == 0:
                print(f"  Step {step}...", flush=True)

            if env.termination_manager.get_term("success").any():
                print(f"  Success at step {step}! Resetting.")
                flush_video()
                episode += 1
                env.reset()
            elif terminated.any() or truncated.any():
                flush_video()
                episode += 1
                env.reset()
    except KeyboardInterrupt:
        print("Interrupted, shutting down.")
    finally:
        flush_video()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
