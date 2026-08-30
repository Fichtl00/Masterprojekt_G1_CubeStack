"""
Open-loop evaluation for the g1_dex3_blockstacking finetune: for a handful of trajectories,
feed the *recorded* observations at each sampled timestep and compare the model's predicted
action chunk against the recorded ground-truth actions (no simulation, no closed-loop rollout).

Mirrors the pattern used for the GR00T Referenz open-loop eval (gr00t/eval/open_loop_eval.py /
referenz_agile_joint_space_finetuning_und_eval.md), adapted to UnifoLM-VLA's inference API
(baseframework.from_pretrained + predict_action), reading directly from the already-converted
HDF5 episodes (fast, no video decode needed).

NOTE: episodes evaluated here are part of the 301 training episodes -- this is NOT a held-out
split. Low error here shows good fit to training data, not generalization (same caveat as
documented in referenz_agile_joint_space_finetuning_und_eval.md, Section 5).

Usage:
    python open_loop_eval_g1_dex3.py \
        --ckpt_path /home/omniverse-2/Groot/outputs_unifolm_vla/g1_dex3_blockstacking_scratch/checkpoints/steps_24000_pytorch_model.pt \
        --vlm_pretrained_path /home/omniverse-2/UnifoLM-VLM-Base \
        --hdf5_dir /home/omniverse-2/IsaacLab/Datasets/g1_dex3_blockstacking_hdf5 \
        --episode-ids 0 1 2 3 4 \
        --stride 20 \
        --output-dir /home/omniverse-2/Groot/outputs_unifolm_vla/g1_dex3_blockstacking_scratch/eval_open_loop
"""

import os
import json
import argparse

import h5py
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qwen_vl_utils import process_vision_info

from unifolm_vla.model.framework.base_framework import baseframework
from unifolm_vla.rlds_dataloader.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

JOINT_NAMES = [
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
    "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0", "kLeftHandIndex1",
    "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
    "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1",
]


def unnormalize(values, stats):
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = np.array(stats.get("mask", [True] * len(stats["min"])))
        high, low = np.array(stats["max"]), np.array(stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = np.array(stats.get("mask", [True] * len(stats["q01"])))
        high, low = np.array(stats["q99"]), np.array(stats["q01"])
    else:
        raise ValueError(f"Unsupported normalization type: {ACTION_PROPRIO_NORMALIZATION_TYPE}")
    return np.where(mask, 0.5 * (values + 1) * (high - low + 1e-8) + low, values)


def normalize(values, stats):
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = np.array(stats.get("mask", [True] * len(stats["min"])))
        high, low = np.array(stats["max"]), np.array(stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = np.array(stats.get("mask", [True] * len(stats["q01"])))
        high, low = np.array(stats["q99"]), np.array(stats["q01"])
    else:
        raise ValueError(f"Unsupported normalization type: {ACTION_PROPRIO_NORMALIZATION_TYPE}")
    return np.clip(np.where(mask, 2 * (values - low) / (high - low + 1e-8) - 1, values), -1.0, 1.0)


def predict_chunk(vla, processor, images, instruction, state, norm_stats_proprio):
    text = f"The task is \"{instruction.lower()}\"."
    messages = [{
        "role": "user",
        "content": [*[{"type": "image", "image": img} for img in images], {"type": "text", "text": text}],
    }]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    batch_input = processor(text=chat_text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    normalized_state = normalize(state, norm_stats_proprio)
    batch_input["state"] = torch.from_numpy(normalized_state).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    batch_input["input_ids"] = batch_input["input_ids"].to(DEVICE)
    batch_input["attention_mask"] = batch_input["attention_mask"].to(DEVICE)
    batch_input["pixel_values"] = batch_input["pixel_values"].to(DEVICE)
    batch_input["image_grid_thw"] = batch_input["image_grid_thw"].to(DEVICE)

    with torch.no_grad():
        out = vla.predict_action(qwen_inputs=batch_input)
    return out["normalized_actions"][0]  # (action_horizon, action_dim)


def eval_episode(vla, processor, hdf5_path, episode_id, stride, action_horizon, norm_stats, output_dir):
    with h5py.File(hdf5_path, "r") as f:
        actions = f["action"][:]
        states = f["observations"]["qpos"][:]
        images_primary = f["observations"]["images"]["cam_left_high"][:]
        images_left_wrist = f["observations"]["images"]["cam_left_wrist"][:]
        images_right_wrist = f["observations"]["images"]["cam_right_wrist"][:]
        instruction = f["language_raw"][()]
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")

    T = actions.shape[0]
    timesteps = list(range(0, T - action_horizon, stride))

    all_preds, all_gts, all_ts = [], [], []
    for t in timesteps:
        images = [
            Image.fromarray(images_primary[t]),
            Image.fromarray(images_left_wrist[t]),
            Image.fromarray(images_right_wrist[t]),
        ]
        pred_normalized = predict_chunk(vla, processor, images, instruction, states[t], norm_stats["proprio"])
        pred = unnormalize(pred_normalized, norm_stats["action"])
        gt = actions[t:t + action_horizon]
        all_preds.append(pred[:len(gt)])
        all_gts.append(gt)
        all_ts.append(t)

    preds = np.concatenate(all_preds, axis=0)
    gts = np.concatenate(all_gts, axis=0)
    mse = float(np.mean((preds - gts) ** 2))
    mae = float(np.mean(np.abs(preds - gts)))

    # Plot against the *real* trajectory timestep, with a NaN-separated gap between each
    # sampled window -- these windows are independent inference calls at non-contiguous
    # times (spaced `stride` apart), so joining them without a gap would draw a misleading
    # straight line across time that never actually happened.
    fig, axes = plt.subplots(4, 7, figsize=(28, 12))
    for dim in range(28):
        ax = axes[dim // 7, dim % 7]
        for i, (pred, gt, t) in enumerate(zip(all_preds, all_gts, all_ts)):
            xs = np.arange(t, t + len(gt))
            ax.plot(xs, gt[:, dim], color="tab:blue", linewidth=1.2, label="GT" if i == 0 else None)
            ax.plot(xs, pred[:, dim], color="tab:orange", linewidth=1.2, alpha=0.8, label="Pred" if i == 0 else None)
            ax.axvline(t, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
        ax.set_title(JOINT_NAMES[dim], fontsize=8)
        ax.tick_params(labelsize=6)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Episode {episode_id} -- MSE={mse:.4f} MAE={mae:.4f}")
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"traj_{episode_id}_open_loop.jpeg"), dpi=100)
    plt.close(fig)

    return mse, mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--vlm_pretrained_path", type=str, required=True)
    parser.add_argument("--hdf5_dir", type=str, required=True)
    parser.add_argument("--episode-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--stride", type=int, default=20, help="Sample every N timesteps within a trajectory")
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    vla = baseframework.from_pretrained(args.ckpt_path, vlm_pretrained_path=args.vlm_pretrained_path)
    vla = vla.to(torch.bfloat16).to(DEVICE).eval()
    processor = vla.qwen_vl_interface.processor
    norm_stats = vla.norm_stats["g1_dex3_blockstacking"]

    results = {}
    for episode_id in args.episode_ids:
        hdf5_path = os.path.join(args.hdf5_dir, f"episode_{episode_id}.hdf5")
        mse, mae = eval_episode(vla, processor, hdf5_path, episode_id, args.stride, args.action_horizon, norm_stats, args.output_dir)
        results[episode_id] = {"mse": mse, "mae": mae}
        print(f"Episode {episode_id}: MSE={mse:.4f} MAE={mae:.4f}")

    avg_mse = float(np.mean([r["mse"] for r in results.values()]))
    avg_mae = float(np.mean([r["mae"] for r in results.values()]))
    results["average"] = {"mse": avg_mse, "mae": avg_mae}
    print(f"Average: MSE={avg_mse:.4f} MAE={avg_mae:.4f}")

    with open(os.path.join(args.output_dir, "open_loop_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
