#!/usr/bin/env python3
"""
Convert Isaac Lab HDF5 dataset to GR00T LeRobot v2 format, joint-space ("WBC") variant.

Unlike convert_hdf5_to_Lerobot.py (which keeps left_arm/right_arm as absolute EEF
Pink-IK poses matching examples/Referenz/referenz_config.py), this script slices the raw
43-dim robot_joint_pos vector into named joint groups and re-labels left_arm/right_arm
as absolute JOINT angles. Output matches examples/Referenz/referenz_wbc_config.py
(ActionType.NON_EEF for every group), mirroring the schema used by
isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py (unitree_g1_sim_wbc_config).

Rationale: applying a Pink-IK/FK conversion wrapper at inference time is too slow
for closed-loop control, so the joint-space representation is baked into the
training data instead and GR00T is fine-tuned directly on it.

Joint-space action labels are NOT copied from the recorded HDF5 "actions" field
(which holds the raw 28-dim EEF+hand env action sent to env.step, i.e. what the
Pink-IK controller was commanded, not what the arm joints actually did). Instead
they are relabeled from the ACHIEVED next-frame robot_joint_pos state
(standard behavior-cloning technique: state[t+1] becomes the joint-space action
target for state[t]), sliced by the same GROUP_INDICES as the state.

GROUP_INDICES below encodes the mapping from named joint groups to indices into
the flat 43-dim robot_joint_pos vector. This mapping was verified empirically
(NOT assumed) against dataset_g1_locomanip_annotated.hdf5 (batch01): every one of
the 14 columns of obs/hand_joint_state matches robot_joint_pos[29:43] exactly and
in order across a full episode -- this is only consistent with the joint ordering
in isaaclab_arena_g1/g1_env/config/lab_g1_joints_order_43dof.yaml (NOT the
loco_manip_g1_joints_order_43dof.yaml variant, which does not place the combined
14-joint hand block contiguously at the end). If this converter is ever pointed at
HDF5 data recorded from a different robot config/USD, re-verify GROUP_INDICES
before trusting the output -- see the hand_joint_state cross-check in the
conversation/commit history, or rerun the same column-matching check.

Usage:
    python convert_hdf5_to_Lerobot_wbc.py \
        --input_files /path/to/dataset_V1.hdf5 /path/to/dataset_V2.hdf5 \
        --output_dir /path/to/lerobot_dataset_wbc \
        --language_instruction "Pick up the object and place it on the holder." \
        --fps 50
"""

import argparse
import json
import os

import cv2
import h5py
import numpy as np
import pandas as pd

# Index of each named joint within the flat 43-dim robot_joint_pos vector,
# per lab_g1_joints_order_43dof.yaml. See module docstring for verification method.
LAB_JOINT_INDEX = {
    "left_hip_pitch_joint": 0, "right_hip_pitch_joint": 1, "waist_yaw_joint": 2,
    "left_hip_roll_joint": 3, "right_hip_roll_joint": 4, "waist_roll_joint": 5,
    "left_hip_yaw_joint": 6, "right_hip_yaw_joint": 7, "waist_pitch_joint": 8,
    "left_knee_joint": 9, "right_knee_joint": 10,
    "left_shoulder_pitch_joint": 11, "right_shoulder_pitch_joint": 12,
    "left_ankle_pitch_joint": 13, "right_ankle_pitch_joint": 14,
    "left_shoulder_roll_joint": 15, "right_shoulder_roll_joint": 16,
    "left_ankle_roll_joint": 17, "right_ankle_roll_joint": 18,
    "left_shoulder_yaw_joint": 19, "right_shoulder_yaw_joint": 20,
    "left_elbow_joint": 21, "right_elbow_joint": 22,
    "left_wrist_roll_joint": 23, "right_wrist_roll_joint": 24,
    "left_wrist_pitch_joint": 25, "right_wrist_pitch_joint": 26,
    "left_wrist_yaw_joint": 27, "right_wrist_yaw_joint": 28,
    "left_hand_index_0_joint": 29, "left_hand_middle_0_joint": 30, "left_hand_thumb_0_joint": 31,
    "right_hand_index_0_joint": 32, "right_hand_middle_0_joint": 33, "right_hand_thumb_0_joint": 34,
    "left_hand_index_1_joint": 35, "left_hand_middle_1_joint": 36, "left_hand_thumb_1_joint": 37,
    "right_hand_index_1_joint": 38, "right_hand_middle_1_joint": 39, "right_hand_thumb_1_joint": 40,
    "left_hand_thumb_2_joint": 41, "right_hand_thumb_2_joint": 42,
}

# Canonical per-group joint name order -- must match examples/Referenz/referenz_g1_joints.json
GROUP_JOINT_NAMES = {
    "left_arm": [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    ],
    "right_arm": [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ],
    "left_hand": [
        "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
        "left_hand_index_0_joint", "left_hand_index_1_joint",
        "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    ],
    "right_hand": [
        "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
        "right_hand_index_0_joint", "right_hand_index_1_joint",
        "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    ],
    "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
}
GROUP_INDICES = {
    group: [LAB_JOINT_INDEX[name] for name in names] for group, names in GROUP_JOINT_NAMES.items()
}
# State includes "waist" for proprioceptive context (non-trivial variance from arm-motion
# coupling). Action excludes it -- waist is not commanded by teleop/model (see
# examples/Referenz/referenz_wbc_config.py docstring), only passively coupled; legs are excluded
# from both (near-constant, fix_root_link=True, driven externally in the WBC reference design).
STATE_GROUP_ORDER = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]
ACTION_GROUP_ORDER = ["left_arm", "right_arm", "left_hand", "right_hand"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--language_instruction", type=str,
        default="Pick up the object and place it on the holder.",
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--cameras", nargs="+",
        default=["cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist"],
    )
    return parser.parse_args()


def save_video(frames, output_path, fps):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    T, C, H, W = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for frame in frames:
        img = (frame.transpose(1, 2, 0) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(img_bgr)
    writer.release()


def build_joint_state(robot_joint_pos):
    """Slice the flat (T, 43) robot_joint_pos array into named joint groups, each (T, D)."""
    return {group: robot_joint_pos[:, idx] for group, idx in GROUP_INDICES.items()}


def main():
    args = parse_args()

    all_episodes = []
    for hdf5_path in args.input_files:
        print(f"Loading {hdf5_path}...")
        with h5py.File(hdf5_path, "r") as f:
            for ep_name in sorted(f["data"].keys()):
                ep = {}
                ep["obs"] = {}
                for key in f["data"][ep_name]["obs"].keys():
                    ep["obs"][key] = np.array(f["data"][ep_name]["obs"][key])
                all_episodes.append(ep)
                print(f"  Loaded {ep_name}: {ep['obs']['robot_joint_pos'].shape[0]} steps")

    print(f"\nTotal episodes: {len(all_episodes)}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "meta"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "data", "chunk-000"), exist_ok=True)
    for cam in args.cameras:
        os.makedirs(
            os.path.join(args.output_dir, "videos", "chunk-000", f"observation.images.{cam}"),
            exist_ok=True,
        )

    state_dim = sum(len(GROUP_INDICES[g]) for g in STATE_GROUP_ORDER)  # 31
    action_dim = sum(len(GROUP_INDICES[g]) for g in ACTION_GROUP_ORDER)  # 28
    print(f"State dim: {state_dim} (groups: {STATE_GROUP_ORDER})")
    print(f"Action dim: {action_dim} (groups: {ACTION_GROUP_ORDER})")

    global_frame_index = 0
    episodes_meta = []
    tasks_meta = [
        {"task_index": 0, "task": args.language_instruction},
        {"task_index": 1, "task": "valid"},
    ]

    for ep_idx, ep in enumerate(all_episodes):
        robot_joint_pos = ep["obs"]["robot_joint_pos"]  # (T, 43)
        T = robot_joint_pos.shape[0]

        joint_groups = build_joint_state(robot_joint_pos)  # each (T, D)
        states = np.concatenate([joint_groups[g] for g in STATE_GROUP_ORDER], axis=-1)  # (T, 31)

        # Action = achieved joint state one step ahead (last frame repeats itself -- no t+1 exists)
        joint_groups_next = {g: np.concatenate([v[1:], v[-1:]], axis=0) for g, v in joint_groups.items()}
        actions = np.concatenate([joint_groups_next[g] for g in ACTION_GROUP_ORDER], axis=-1)  # (T, 28)

        for cam in args.cameras:
            if cam in ep["obs"]:
                frames = ep["obs"][cam]
                video_path = os.path.join(
                    args.output_dir, "videos", "chunk-000",
                    f"observation.images.{cam}", f"episode_{ep_idx:06d}.mp4",
                )
                save_video(frames, video_path, args.fps)

        rows = []
        for t in range(T):
            rows.append({
                "observation.state": states[t].tolist(),
                "action": actions[t].tolist(),
                "timestamp": t / args.fps,
                "annotation.human.action.task_description": 0,
                "task_index": 0,
                "annotation.human.validity": 1,
                "episode_index": ep_idx,
                "index": global_frame_index + t,
                "next.reward": 0.0,
                "next.done": t == T - 1,
            })

        df = pd.DataFrame(rows)
        parquet_path = os.path.join(args.output_dir, "data", "chunk-000", f"episode_{ep_idx:06d}.parquet")
        df.to_parquet(parquet_path, index=False)

        episodes_meta.append({"episode_index": ep_idx, "tasks": [args.language_instruction], "length": T})
        global_frame_index += T
        print(f"  Written episode {ep_idx}: {T} steps")

    with open(os.path.join(args.output_dir, "meta", "episodes.jsonl"), "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    with open(os.path.join(args.output_dir, "meta", "tasks.jsonl"), "w") as f:
        for task in tasks_meta:
            f.write(json.dumps(task) + "\n")

    def build_ranges(group_order):
        ranges, idx = {}, 0
        for group in group_order:
            dim = len(GROUP_INDICES[group])
            ranges[group] = {"start": idx, "end": idx + dim}
            idx += dim
        return ranges

    state_ranges = build_ranges(STATE_GROUP_ORDER)
    action_ranges = build_ranges(ACTION_GROUP_ORDER)

    modality = {
        "state": state_ranges,
        "action": action_ranges,
        "video": {cam: {"original_key": f"observation.images.{cam}"} for cam in args.cameras},
        "annotation": {
            "human.action.task_description": {},
            "human.validity": {},
        },
    }
    with open(os.path.join(args.output_dir, "meta", "modality.json"), "w") as f:
        json.dump(modality, f, indent=2)

    info = {
        "codebase_version": "v2.0",
        "robot_type": "unitree_g1",
        "total_episodes": len(all_episodes),
        "total_frames": global_frame_index,
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": args.fps,
        "splits": {"train": f"0:{len(all_episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [state_dim], "names": STATE_GROUP_ORDER},
            "action": {"dtype": "float32", "shape": [action_dim], "names": ACTION_GROUP_ORDER},
            **{
                f"observation.images.{cam}": {
                    "dtype": "video",
                    "shape": [3, 480, 640],
                    "names": ["channel", "height", "width"],
                    "video_info": {"video.fps": args.fps, "video.codec": "mp4v"},
                }
                for cam in args.cameras
            },
        },
    }
    with open(os.path.join(args.output_dir, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDone! Joint-space LeRobot dataset written to: {args.output_dir}")
    print(f"Total episodes: {len(all_episodes)}")
    print(f"Total frames: {global_frame_index}")


if __name__ == "__main__":
    main()
