#!/usr/bin/env python3
"""Run UnifoLM-VLA inference against a live Isaac Lab sim via shared memory.

The client reads camera frames from the sim's per-camera SHMs, reads robot state
from `isaac_robot_state`, runs the VLA checkpoint, and writes the resulting
motor command dictionary to `dds_robot_cmd` so the sim's DDS bridge can consume it.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info


VLA_ROOT = Path("/home/omniverse-2/unifolm-vla/unifolm-vla")
SIM_ROOT = Path("/home/omniverse-2/unitree_sim_isaaclab")

for path in (SIM_ROOT, VLA_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from dds.sharedmemorymanager import SharedMemoryManager
from tools.shared_memory_utils import MultiImageReader
from unifolm_vla.model.framework.base_framework import baseframework
from unifolm_vla.rlds_dataloader.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType


ARM_JOINT_TARGET_INDICES = list(range(15, 29))
ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_KP = [
    60, 60, 60, 100, 40, 40,
    60, 60, 60, 100, 40, 40,
    60, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
]

G1_KD = [
    1, 1, 1, 2, 1, 1,
    1, 1, 1, 2, 1, 1,
    1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
]


def _as_numpy(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return array.reshape(-1)


def _normalize_bounds(values: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = np.asarray(stats.get("mask", np.ones_like(stats["min"], dtype=bool)), dtype=bool)
        high = np.asarray(stats["max"], dtype=np.float32)
        low = np.asarray(stats["min"], dtype=np.float32)
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)), dtype=bool)
        high = np.asarray(stats["q99"], dtype=np.float32)
        low = np.asarray(stats["q01"], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported normalization type: {ACTION_PROPRIO_NORMALIZATION_TYPE}")

    values = np.asarray(values, dtype=np.float32)
    normalized = np.where(mask, 2.0 * (values - low) / (high - low + 1e-8) - 1.0, values)
    return np.clip(normalized, -1.0, 1.0)


def _unnormalize_action(values: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = np.asarray(stats.get("mask", np.ones_like(stats["min"], dtype=bool)), dtype=bool)
        high = np.asarray(stats["max"], dtype=np.float32)
        low = np.asarray(stats["min"], dtype=np.float32)
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)), dtype=bool)
        high = np.asarray(stats["q99"], dtype=np.float32)
        low = np.asarray(stats["q01"], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported normalization type: {ACTION_PROPRIO_NORMALIZATION_TYPE}")

    values = np.asarray(values, dtype=np.float32)
    return np.where(mask, 0.5 * (values + 1.0) * (high - low + 1e-8) + low, values)


def _resize_and_crop(image: np.ndarray, image_size: int, center_crop: bool) -> Image.Image:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    if pil_image.size != (image_size, image_size):
        resampling = getattr(Image, "Resampling", None)
        resample = getattr(resampling, "LANCZOS", getattr(Image, "LANCZOS"))
        pil_image = pil_image.resize((image_size, image_size), resample=resample)
    if center_crop:
        crop_scale = 0.9
        crop_w = int(round(pil_image.width * crop_scale ** 0.5))
        crop_h = int(round(pil_image.height * crop_scale ** 0.5))
        left = max(0, (pil_image.width - crop_w) // 2)
        top = max(0, (pil_image.height - crop_h) // 2)
        pil_image = pil_image.crop((left, top, left + crop_w, top + crop_h)).resize((image_size, image_size), resample=resample)
    return pil_image.convert("RGB")


def _compose_proprio(robot_state: Dict[str, Any], proprio_dim: int) -> np.ndarray:
    """Generischer Fallback (nicht Dex3-spezifisch) -- siehe _compose_dex3_proprio fuer den
    korrekten 28-dim Pfad, der tatsaechlich dem Trainingsdatensatz entspricht."""
    parts: List[np.ndarray] = []
    for key in ("joint_positions", "joint_velocities", "joint_torques", "imu_data"):
        if key in robot_state and robot_state[key] is not None:
            parts.append(_as_numpy(robot_state[key]))

    if not parts:
        composed = np.zeros(proprio_dim, dtype=np.float32)
    else:
        composed = np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    if composed.size < proprio_dim:
        padded = np.zeros(proprio_dim, dtype=np.float32)
        padded[: composed.size] = composed
        composed = padded
    elif composed.size > proprio_dim:
        composed = composed[:proprio_dim]

    return composed


def _compose_dex3_proprio(current_positions: np.ndarray, dex3_state: Dict[str, Any] | None) -> np.ndarray | None:
    """Baut das 28-dim Proprio EXAKT wie im Trainingsdatensatz (14 Arm- + 14
    Dex3-Fingergelenke, siehe LeRobot-Metadaten): `isaac_robot_state`'s "joint_positions"
    enthaelt nur die 29 Koerpermotoren (Beine+Taille+Arme, KEINE Hand); die Handwinkel
    kommen separat aus `isaac_dex3_state` (siehe dex3_dds.py::publish_hand_states). Ohne
    diese Funktion wuerde der generische `_compose_proprio`-Fallback stattdessen
    Beine/Taille/Armreste konkatenieren -- eine Verteilung, die nichts mit den
    Trainingsdaten zu tun hat.

    Gibt None zurueck, wenn der Hand-State noch nicht verfuegbar ist (z.B. kurz nach dem
    Start, bevor der erste Hand-State publiziert wurde) -- der Aufrufer soll in diesem Fall
    auf den naechsten Schritt warten statt mit Nullen zu inferieren.
    """
    if dex3_state is None:
        return None
    left_hand = dex3_state.get("left_hand", {})
    right_hand = dex3_state.get("right_hand", {})
    left_positions = left_hand.get("positions")
    right_positions = right_hand.get("positions")
    if not left_positions or not right_positions:
        return None
    if len(left_positions) < NUM_HAND_VALUES_PER_SIDE or len(right_positions) < NUM_HAND_VALUES_PER_SIDE:
        return None

    arm_values = np.asarray(current_positions[ARM_JOINT_TARGET_INDICES], dtype=np.float32)
    left_values = np.asarray(left_positions[:NUM_HAND_VALUES_PER_SIDE], dtype=np.float32)
    right_sim_order = np.asarray(right_positions[:NUM_HAND_VALUES_PER_SIDE], dtype=np.float32)
    right_dataset_order = right_sim_order[_RIGHT_HAND_DATASET_TO_SIM_ORDER]

    return np.concatenate([arm_values, left_values, right_dataset_order], axis=0)


# Dex3-Handgelenk-Reihenfolge im Trainingsdatensatz (siehe LeRobot-Metadaten): 14 Arm-Werte
# gefolgt von 14 Handgelenk-Werten, Haende jeweils 7 Werte. Links: Thumb0,1,2,Middle0,1,
# Index0,1. Rechts ASYMMETRISCH: Thumb0,1,2,Index0,1,Middle0,1 -- unitree_sim_isaaclab's
# action_provider_dds.py::right_hand_joint_mapping (fuers Schreiben) UND
# tasks/common_observations/dex3_state.py::get_robot_girl_joint_names (fuers Lesen)
# erwarten dagegen fuer BEIDE Haende dieselbe Reihenfolge (Thumb0,1,2,Middle0,1,Index0,1).
# Diese Permutation ist selbstinvers (vertauscht nur die 2 gleich grossen Middle/Index-
# Bloecke), daher fuer BEIDE Richtungen (Dataset->Sim beim Schreiben der Aktion, Sim->
# Dataset beim Lesen des Proprios) dieselbe Konstante verwendbar.
NUM_ARM_VALUES = 14
NUM_HAND_VALUES_PER_SIDE = 7
_RIGHT_HAND_DATASET_TO_SIM_ORDER = [0, 1, 2, 5, 6, 3, 4]  # thumb0,1,2,index0,1,middle0,1 <-> thumb0,1,2,middle0,1,index0,1


def _build_hand_cmd(positions: Sequence[float]) -> Dict[str, Any]:
    positions = [float(v) for v in positions]
    n = len(positions)
    return {
        "positions": positions,
        "velocities": [0.0] * n,
        "torques": [0.0] * n,
        "kp": [100.0] * n,
        "kd": [3.0] * n,
    }


def _build_dex3_hand_cmd(action: np.ndarray) -> Dict[str, Any] | None:
    """Baut das isaac_dex3_cmd-SHM-Payload aus den letzten 14 Aktionswerten (7 links + 7
    rechts), falls das Modell ein 28-dim Arm+Hand-Aktionsformat ausgibt. Gibt None zurueck,
    wenn die Aktion zu kurz ist (z.B. reine Arm-Checkpoints mit action_dim<=14)."""
    flat_action = np.asarray(action, dtype=np.float32).reshape(-1)
    required = NUM_ARM_VALUES + 2 * NUM_HAND_VALUES_PER_SIDE
    if flat_action.size < required:
        return None

    left_raw = flat_action[NUM_ARM_VALUES : NUM_ARM_VALUES + NUM_HAND_VALUES_PER_SIDE]
    right_raw = flat_action[NUM_ARM_VALUES + NUM_HAND_VALUES_PER_SIDE : required]
    right_reordered = right_raw[_RIGHT_HAND_DATASET_TO_SIM_ORDER]

    return {
        "left_hand_cmd": _build_hand_cmd(left_raw),
        "right_hand_cmd": _build_hand_cmd(right_reordered),
    }


def _build_command_dict(current_positions: np.ndarray, action: np.ndarray, target_indices: Sequence[int]) -> Dict[str, Any]:
    motor_count = 29
    positions = np.asarray(current_positions, dtype=np.float32).copy()
    if positions.size < motor_count:
        padded = np.zeros(motor_count, dtype=np.float32)
        padded[: positions.size] = positions
        positions = padded
    else:
        positions = positions[:motor_count]

    flat_action = np.asarray(action, dtype=np.float32).reshape(-1)
    use_count = min(flat_action.size, len(target_indices))
    for index, value in zip(target_indices[:use_count], flat_action[:use_count]):
        if 0 <= index < motor_count:
            positions[index] = float(value)

    motor_cmd = []
    for index in range(motor_count):
        motor_cmd.append(
            {
                "mode": 1,
                "q": float(positions[index]),
                "dq": 0.0,
                "tau": 0.0,
                "kp": float(G1_KP[index]),
                "kd": float(G1_KD[index]),
                "reserve": 0,
            }
        )

    return {
        "mode_pr": 0,
        "mode_machine": 0,
        "motor_cmd": motor_cmd,
    }


class VLAIsaacSimBridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        # image_order treibt jetzt sowohl die SHM-Namen (welche isaac_<name>_image_shm
        # gelesen werden) als auch die Auswahl/Reihenfolge fuer das Modell -- vorher war
        # MultiImageReader() intern hart auf ['head','left','right'] codiert, unabhaengig
        # von --image_order, wodurch z.B. --image_order cam_left_high ... nie etwas
        # gefunden haette.
        self.image_reader = MultiImageReader(image_names=self.args.image_order)
        self.robot_state_shm = SharedMemoryManager("isaac_robot_state", 3072)
        self.command_shm = SharedMemoryManager("dds_robot_cmd", 3072)
        # Separater Kanal fuer die 14 Dex3-Fingergelenke (siehe action_provider_dds.py /
        # dex3_dds.py::get_hand_commands -- unabhaengig von dds_robot_cmd, das nur die 29
        # Koerpermotoren traegt). Ohne dies werden Handgelenk-Anteile des Modell-Outputs
        # nirgends angewendet.
        self.dex3_cmd_shm = SharedMemoryManager("isaac_dex3_cmd", 1180)
        # Hand-STATE (fuer Proprio) -- separat von isaac_dex3_cmd (Hand-Kommandos, s.o.).
        # Publiziert vom Sim via dex3_dds.py::publish_hand_states.
        self.dex3_state_shm = SharedMemoryManager("isaac_dex3_state", 1180)
        self.model = self._load_model()
        self.processor = self.model.qwen_vl_interface.processor

        # Resolve normalization stats key.
        # - If the user provided one, validate it.
        # - Otherwise prefer the G1 stack-block split for Base checkpoints, since it is the
        #   most representative split in this multi-task checkpoint.
        # - If that does not exist, fall back to the only available key or the first key.
        available_keys = list(self.model.norm_stats.keys())
        resolved_key = getattr(self.args, "unnorm_key", None)
        if resolved_key is None:
            if "g1_stack_block" in available_keys:
                resolved_key = "g1_stack_block"
            elif len(available_keys) == 1:
                resolved_key = available_keys[0]
            else:
                resolved_key = available_keys[0]
            logging.info("No --unnorm_key provided; auto-selected %s from %s", resolved_key, available_keys)
        elif resolved_key not in available_keys:
            raise AssertionError(
                f"The `unnorm_key` you chose is not in the set of available dataset statistics, please choose from: {available_keys}"
            )

        logging.info("Using normalization stats key: %s", resolved_key)
        self.unnorm_key = resolved_key
        self.norm_stats_action = self.model.norm_stats[self.unnorm_key]["action"]
        self.norm_stats_proprio = self.model.norm_stats[self.unnorm_key]["proprio"]
        self.proprio_dim = len(np.asarray(self.norm_stats_proprio.get("max", self.norm_stats_proprio.get("q99")), dtype=np.float32))
        self.action_dim = len(np.asarray(self.norm_stats_action.get("max", self.norm_stats_action.get("q99")), dtype=np.float32))
        self.target_indices = self._resolve_target_indices()
        self.step_counter = 0
        self.wait_counter = 0
        self.image_log_counter = 0

    def _load_model(self):
        logging.info("Loading checkpoint from %s", self.args.ckpt_path)
        model = baseframework.from_pretrained(
            self.args.ckpt_path,
            vlm_pretrained_path=self.args.vlm_pretrained_path,
        )
        if self.args.use_bf16 and self.device.type == "cuda":
            model = model.to(torch.bfloat16)
        model = model.to(self.device).eval()
        return model

    def _resolve_target_indices(self) -> List[int]:
        if self.args.target_joint_indices:
            return [int(value) for value in self.args.target_joint_indices.split(",") if value.strip()]
        if self.action_dim <= len(ARM_JOINT_TARGET_INDICES):
            return ARM_JOINT_TARGET_INDICES[: self.action_dim]
        return ARM_JOINT_TARGET_INDICES

    def _read_images(self) -> Dict[str, np.ndarray] | None:
        images = self.image_reader.read_images()
        if not images:
            return None
        return images

    def _prepare_images(self, images: Dict[str, np.ndarray]) -> List[Image.Image]:
        prepared: List[Image.Image] = []
        for name in self.args.image_order:
            if name not in images:
                continue
            image = images[name]
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Unexpected image format for {name}: {image.shape}, {image.dtype}")
            if self.args.camera_color_space == "bgr":
                image = image[..., ::-1]
            prepared.append(_resize_and_crop(image, self.args.image_size, self.args.center_crop))
        return prepared

    def _build_batch(self, prepared_images: List[Image.Image], proprio: np.ndarray) -> Dict[str, torch.Tensor]:
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in prepared_images],
                    {"type": "text", "text": f'The task is "{self.args.instruction.lower()}".'},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        batch = self.processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        proprio_tensor = torch.from_numpy(_normalize_bounds(proprio, self.norm_stats_proprio)).unsqueeze(0)
        batch["state"] = proprio_tensor.to(self.device)
        batch["input_ids"] = batch["input_ids"].to(self.device)
        batch["attention_mask"] = batch["attention_mask"].to(self.device)
        batch["pixel_values"] = batch["pixel_values"].to(self.device)
        batch["image_grid_thw"] = batch["image_grid_thw"].to(self.device)
        return batch

    @torch.inference_mode()
    def step(self) -> bool:
        self.step_counter += 1
        robot_state = self.robot_state_shm.read_data()
        images = self._read_images()
        if robot_state is None or images is None:
            self.wait_counter += 1
            if self.wait_counter == 1 or self.wait_counter % 20 == 0:
                missing_parts: List[str] = []
                if robot_state is None:
                    missing_parts.append("isaac_robot_state")
                if images is None:
                    missing_parts.append("camera SHMs")
                logging.info("Waiting for %s (step=%d, wait=%d)", ", ".join(missing_parts), self.step_counter, self.wait_counter)
            return False

        if self.wait_counter:
            logging.info("Received first valid sim data after %d wait loops", self.wait_counter)
            self.wait_counter = 0

        current_positions = _as_numpy(robot_state.get("joint_positions", []))
        if current_positions.size == 0:
            logging.info("Sim data arrived but joint_positions is empty; waiting for valid robot state")
            return False

        dex3_expected_dim = NUM_ARM_VALUES + 2 * NUM_HAND_VALUES_PER_SIDE
        if self.proprio_dim == dex3_expected_dim:
            # 28-dim Dex3-Checkpoint: Arm-Proprio kommt aus isaac_robot_state, Hand-Proprio
            # separat aus isaac_dex3_state (siehe _compose_dex3_proprio-Docstring). Falls der
            # Hand-State noch nicht publiziert wurde, warten statt mit falschem/generischem
            # Proprio zu inferieren.
            dex3_state = self.dex3_state_shm.read_data()
            proprio = _compose_dex3_proprio(current_positions, dex3_state)
            if proprio is None:
                self.wait_counter += 1
                if self.wait_counter == 1 or self.wait_counter % 20 == 0:
                    logging.info(
                        "Waiting for isaac_dex3_state (step=%d, wait=%d)", self.step_counter, self.wait_counter
                    )
                return False
        else:
            proprio = _compose_proprio(robot_state, self.proprio_dim)
        prepared_images = self._prepare_images(images)
        if self.args.log_image_presence:
            available_names = sorted(images.keys())
            selected_names = [name for name in self.args.image_order if name in images]
            missing_names = [name for name in self.args.image_order if name not in images]
            self.image_log_counter += 1
            should_log = (
                self.image_log_counter == 1
                or bool(missing_names)
                or (self.image_log_counter % max(1, self.args.log_image_presence_interval) == 0)
            )
            if should_log:
                logging.info(
                    "Image inputs | available=%s | selected=%s | missing=%s",
                    available_names,
                    selected_names,
                    missing_names,
                )
        if not prepared_images:
            logging.info("Sim data arrived but no usable images matched image_order=%s", self.args.image_order)
            return False

        batch = self._build_batch(prepared_images, proprio)
        if self.device.type == "cuda":
            autocast_dtype = torch.bfloat16 if self.args.use_bf16 else torch.float16
            autocast_ctx = torch.autocast("cuda", dtype=autocast_dtype)
        else:
            autocast_ctx = torch.autocast("cpu", dtype=torch.float32)

        with autocast_ctx:
            prediction = self.model.predict_action(qwen_inputs=batch)

        normalized_actions = prediction["normalized_actions"][0]
        action = _unnormalize_action(normalized_actions[0], self.norm_stats_action)
        cmd = _build_command_dict(current_positions, action, self.target_indices)
        self.command_shm.write_data(cmd)

        hand_cmd = _build_dex3_hand_cmd(action)
        if hand_cmd is not None:
            self.dex3_cmd_shm.write_data(hand_cmd)
        if self.args.log_actions:
            action_preview = np.array2string(action[: min(10, action.size)], precision=4, suppress_small=True)
            normalized_preview = np.array2string(
                np.asarray(normalized_actions[0]).reshape(-1)[: min(10, action.size)],
                precision=4,
                suppress_small=True,
            )
            target_preview = [cmd["motor_cmd"][index]["q"] for index in self.target_indices[: min(10, len(self.target_indices))]]
            logging.info("Step %d | normalized action[:10]=%s", self.step_counter, normalized_preview)
            logging.info("Step %d | unnormalized action[:10]=%s", self.step_counter, action_preview)
            logging.info("Step %d | target q[:10]=%s", self.step_counter, np.array2string(np.asarray(target_preview), precision=4, suppress_small=True))
        logging.info("Published command with %d target joints", min(len(action.reshape(-1)), len(self.target_indices)))
        return True


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UnifoLM-VLA Isaac Lab shared-memory bridge")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--vlm_pretrained_path", type=str, default=None, help="Optional VLM base model path")
    parser.add_argument("--unnorm_key", type=str, default=None, help="Normalization stats key (optional). If omitted the key is inferred from the checkpoint's dataset statistics")
    parser.add_argument("--instruction", type=str, default="move the robot arm", help="Task instruction prompt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    parser.add_argument("--poll_interval", type=float, default=0.05, help="Loop sleep in seconds")
    parser.add_argument("--image_size", type=int, default=224, help="Vision input size")
    parser.add_argument("--center_crop", action="store_true", help="Apply center crop before inference")
    parser.add_argument("--use_bf16", action="store_true", default=True, help="Use bfloat16 on CUDA")
    parser.add_argument("--camera_color_space", choices=["rgb", "bgr"], default="bgr", help="Color layout in camera SHMs")
    parser.add_argument("--image_order", nargs="+", default=["head", "left", "right"], help="Image names to read from SHM")
    parser.add_argument("--log_image_presence", action="store_true", help="Log available/selected/missing camera inputs from shared memory")
    parser.add_argument("--log_image_presence_interval", type=int, default=50, help="How often to log camera input presence when --log_image_presence is set")
    parser.add_argument("--target_joint_indices", type=str, default=None, help="Comma-separated target motor indices")
    parser.add_argument("--log_actions", action="store_true", help="Log normalized/unnormalized actions and target joint values")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")

    bridge = VLAIsaacSimBridge(args)
    logging.info("Model action dim: %d | proprio dim: %d", bridge.action_dim, bridge.proprio_dim)
    logging.info("Target joints: %s", bridge.target_indices)

    while True:
        try:
            if not bridge.step():
                time.sleep(args.poll_interval)
                continue
            time.sleep(args.poll_interval)
        except KeyboardInterrupt:
            break
        except Exception as exc:
            logging.exception("Bridge loop error: %s", exc)
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()