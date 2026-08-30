"""
Convert unitreerobotics/G1_Dex3_BlockStacking_Dataset (LeRobot v3.0, joint-space,
flat 28-dim observation.state/action, 4 cameras) to the HDF5 layout expected by
prepare_data/hdf5_to_rlds/rlds_dataset_g1_dex3/rlds_dataset.py.

Unlike prepare_data/convert_lerobot_to_hdf5.py (built for a different, bimanual
left_arm/right_arm/gripper/body-split schema), this dataset's observation.state/action
are already flat 28-dim joint vectors -- no key splitting or EE-pose derivation needed.
Only 3 of the 4 source cameras are written (cam_right_high is dropped), matching the
project decision documented in
Anleitung/Masterprojekt/unifolm_vla_scratch_finetuning_g1_dex3_blockstacking.md.

Usage:
    python convert_lerobot_to_hdf5_g1_dex3.py \
        --repo-id unitreerobotics/G1_Dex3_BlockStacking_Dataset \
        --output_dir /path/to/save/the/converted/data/directory
"""

import os
import argparse
import numpy as np
import h5py
from tqdm import tqdm
from multiprocessing import Pool
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAMERA_KEYS = ("cam_left_high", "cam_left_wrist", "cam_right_wrist")


def process_episode(dataset: LeRobotDataset, episode_index: int) -> dict:
    # `dataset` is always instantiated with `episodes=[episode_index]` (single-episode filter),
    # so episode_data_index is re-indexed to just this one entry -- use 0, not episode_index.
    from_idx = dataset.episode_data_index["from"][0].item()
    to_idx = dataset.episode_data_index["to"][0].item()

    states, actions = [], []
    cameras = {k: [] for k in CAMERA_KEYS}
    task = None

    for step_idx in tqdm(
        range(from_idx, to_idx), desc=f"Episode {episode_index}", position=1, leave=False, dynamic_ncols=True
    ):
        step = dataset[step_idx]
        states.append(step["observation.state"].numpy().astype(np.float32))
        actions.append(step["action"].numpy().astype(np.float32))
        for cam_key in CAMERA_KEYS:
            img = step[f"observation.images.{cam_key}"].numpy()  # (3, H, W) float in [0, 1]
            img = (img * 255).astype(np.uint8).transpose(1, 2, 0)  # -> (H, W, 3) uint8
            cameras[cam_key].append(img)
        task = step["task"]

    cam_height, cam_width = cameras[CAMERA_KEYS[0]][0].shape[:2]

    return {
        "episode_index": episode_index,
        "episode_length": to_idx - from_idx,
        "state": np.stack(states),
        "action": np.stack(actions),
        "cameras": {k: np.stack(v) for k, v in cameras.items()},
        "task": task,
        "cam_height": cam_height,
        "cam_width": cam_width,
    }


def write_episode_h5(episode: dict, output_dir: str) -> None:
    h5_path = os.path.join(output_dir, f"episode_{episode['episode_index']}.hdf5")
    episode_length = episode["episode_length"]
    state_dim = episode["state"].shape[1]
    action_dim = episode["action"].shape[1]

    with h5py.File(h5_path, "w", rdcc_nbytes=1024**2 * 2, libver="latest") as root:
        root.attrs["sim"] = False

        obs = root.create_group("observations")
        images = obs.create_group("images")
        for cam_key, imgs in episode["cameras"].items():
            images.create_dataset(
                cam_key,
                shape=(episode_length, episode["cam_height"], episode["cam_width"], 3),
                dtype="uint8",
                chunks=(1, episode["cam_height"], episode["cam_width"], 3),
                compression="gzip",
            )
            images[cam_key][...] = imgs

        obs.create_dataset("qpos", (episode_length, state_dim), dtype="float32", compression="gzip")
        obs["qpos"][...] = episode["state"]

        root.create_dataset("action", (episode_length, action_dim), dtype="float32", compression="gzip")
        root["action"][...] = episode["action"]

        root.create_dataset("language_raw", data=episode["task"])


def _convert_one_episode(args_tuple):
    episode_index, repo_id, root, output_dir, revision = args_tuple
    out_path = os.path.join(output_dir, f"episode_{episode_index}.hdf5")
    if os.path.exists(out_path):
        return episode_index, "skipped"
    # Each worker opens its own LeRobotDataset instance restricted to this one episode --
    # per-frame video decoding via pyav random access is the bottleneck (~118s/episode
    # sequentially for ~1200 frames), so this parallelizes across episodes instead.
    # `revision` should be a full commit SHA (not e.g. "v2.1"): lerobot's `is_valid_version()`
    # treats any PEP440-parseable revision string as a version tag and re-resolves it against
    # the Hub's list_repo_refs on every single instantiation -- with 16 parallel workers this
    # blows through HF's 1000-req/5min quota (429) even though all data is already local. A
    # commit SHA fails that version-parse check and is used as-is, skipping the Hub call.
    dataset = LeRobotDataset(repo_id=repo_id, root=root, revision=revision, episodes=[episode_index], video_backend="pyav")
    episode = process_episode(dataset, episode_index)
    write_episode_h5(episode, output_dir)
    return episode_index, "done"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="unitreerobotics/G1_Dex3_BlockStacking_Dataset")
    parser.add_argument("--root", type=str, default=None, help="Local LeRobot dataset root (optional, else HF cache)")
    parser.add_argument("--revision", type=str, default=None, help="Commit SHA to pin (avoids repeated Hub version lookups)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    num_episodes = LeRobotDataset(repo_id=args.repo_id, root=args.root, revision=args.revision).num_episodes

    tasks = [(i, args.repo_id, args.root, args.output_dir, args.revision) for i in range(num_episodes)]
    with Pool(args.num_workers) as pool:
        for episode_index, status in tqdm(
            pool.imap_unordered(_convert_one_episode, tasks), total=len(tasks), desc="Episodes", dynamic_ncols=True
        ):
            if status == "skipped":
                print(f"Episode {episode_index} already exists, skipping")


if __name__ == "__main__":
    main()
