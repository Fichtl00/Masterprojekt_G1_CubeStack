#!/usr/bin/env python3
"""
Convert Isaac Lab HDF5 dataset to GR00T LeRobot v2 format.

Usage:
    python convert_hdf5_to_lerobot.py \
        --input_files /path/to/dataset_V1.hdf5 /path/to/dataset_V2.hdf5 \
        --output_dir /path/to/lerobot_dataset \
        --language_instruction "Pick up the object and place it on the holder." \
        --fps 50
"""

import argparse
import json
import os
import shutil
import subprocess

import cv2
import h5py
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="Input HDF5 files (can be multiple)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for LeRobot dataset",
    )
    parser.add_argument(
        "--language_instruction",
        type=str,
        default="Pick up the object and place it on the holder.",
        help="Language instruction for the task",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Frames per second for video output",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist"],
        help="Camera names to include",
    )
    return parser.parse_args()


def save_video(frames, output_path, fps):
    """Save frames (T, C, H, W) float32 [0,1] as MP4."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    T, C, H, W = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for frame in frames:
        img = (frame.transpose(1, 2, 0) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(img_bgr)
    writer.release()


def main():
    args = parse_args()

    # Collect all episodes from all input files
    all_episodes = []
    for hdf5_path in args.input_files:
        print(f"Loading {hdf5_path}...")
        with h5py.File(hdf5_path, "r") as f:
            for ep_name in sorted(f["data"].keys()):
                ep = {}
                ep["actions"] = np.array(f["data"][ep_name]["actions"])
                ep["obs"] = {}
                for key in f["data"][ep_name]["obs"].keys():
                    ep["obs"][key] = np.array(f["data"][ep_name]["obs"][key])
                all_episodes.append(ep)
                print(f"  Loaded {ep_name}: {len(ep['actions'])} steps")

    print(f"\nTotal episodes: {len(all_episodes)}")

    # Create output directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "meta"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "data", "chunk-000"), exist_ok=True)
    for cam in args.cameras:
        os.makedirs(
            os.path.join(args.output_dir, "videos", "chunk-000", f"observation.images.{cam}"),
            exist_ok=True,
        )

    # Determine state and action dimensions from first episode
    first_ep = all_episodes[0]
    action_dim = first_ep["actions"].shape[1]  # 28
    state_keys = ["robot_joint_pos", "left_eef_pos", "left_eef_quat", "right_eef_pos", "right_eef_quat"]
    
    # Build state array: concatenate relevant observations
    def build_state(obs):
        parts = []
        for key in state_keys:
            if key in obs:
                parts.append(obs[key])
        return np.concatenate(parts, axis=-1)  # (T, state_dim)

    # Compute state_dim
    state_example = build_state(first_ep["obs"])
    state_dim = state_example.shape[1]
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Write episodes
    global_frame_index = 0
    episodes_meta = []
    tasks_meta = [
        {"task_index": 0, "task": args.language_instruction},
        {"task_index": 1, "task": "valid"},
    ]

    for ep_idx, ep in enumerate(all_episodes):
        T = len(ep["actions"])
        actions = ep["actions"]  # (T, 28)
        states = build_state(ep["obs"])  # (T, state_dim)

        # Save videos
        for cam in args.cameras:
            if cam in ep["obs"]:
                frames = ep["obs"][cam]  # (T, C, H, W)
                video_path = os.path.join(
                    args.output_dir,
                    "videos",
                    "chunk-000",
                    f"observation.images.{cam}",
                    f"episode_{ep_idx:06d}.mp4",
                )
                save_video(frames, video_path, args.fps)

        # Build parquet data
        rows = []
        for t in range(T):
            row = {
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
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        parquet_path = os.path.join(
            args.output_dir, "data", "chunk-000", f"episode_{ep_idx:06d}.parquet"
        )
        df.to_parquet(parquet_path, index=False)

        episodes_meta.append(
            {
                "episode_index": ep_idx,
                "tasks": [args.language_instruction],
                "length": T,
            }
        )
        global_frame_index += T
        print(f"  Written episode {ep_idx}: {T} steps")

    # Write meta/episodes.jsonl
    with open(os.path.join(args.output_dir, "meta", "episodes.jsonl"), "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    # Write meta/tasks.jsonl
    with open(os.path.join(args.output_dir, "meta", "tasks.jsonl"), "w") as f:
        for task in tasks_meta:
            f.write(json.dumps(task) + "\n")

    # Compute state index ranges
    state_ranges = {}
    idx = 0
    for key in state_keys:
        if key in first_ep["obs"]:
            dim = first_ep["obs"][key].shape[1]
            state_ranges[key] = {"start": idx, "end": idx + dim}
            idx += dim

    # Write meta/modality.json
    modality = {
        "state": state_ranges,
        "action": {
            "left_arm": {"start": 0, "end": 7},
            "right_arm": {"start": 7, "end": 14},
            "left_hand": {"start": 14, "end": 21},
            "right_hand": {"start": 21, "end": 28},
        },
        "video": {cam: {"original_key": f"observation.images.{cam}"} for cam in args.cameras},
        "annotation": {
            "human.action.task_description": {},
            "human.validity": {},
        },
    }
    with open(os.path.join(args.output_dir, "meta", "modality.json"), "w") as f:
        json.dump(modality, f, indent=2)

    # Write meta/info.json
    total_frames = global_frame_index
    info = {
        "codebase_version": "v2.0",
        "robot_type": "unitree_g1",
        "total_episodes": len(all_episodes),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": args.fps,
        "splits": {"train": f"0:{len(all_episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": list(state_ranges.keys()),
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": ["left_arm", "right_arm", "left_hand", "right_hand"],
            },
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

    print(f"\nDone! LeRobot dataset written to: {args.output_dir}")
    print(f"Total episodes: {len(all_episodes)}")
    print(f"Total frames: {total_frames}")


if __name__ == "__main__":
    main()