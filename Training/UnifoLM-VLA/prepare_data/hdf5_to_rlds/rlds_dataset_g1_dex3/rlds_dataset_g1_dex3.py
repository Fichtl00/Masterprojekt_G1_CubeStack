from typing import Iterator, Tuple, Any

import os
import h5py
import glob
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
# Add the directory containing this file to sys.path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
from conversion_utils import MultiThreadedDatasetBuilder

# NOTE: unlike rlds_dataset/rlds_dataset.py (g1_stack_block etc.), this dataset has no
# EE-pose ground truth at all -- the source (unitreerobotics/G1_Dex3_BlockStacking_Dataset,
# LeRobot v3.0) only records joint-space state/action (28-dim: 14 arm + 14 Dex3 hand
# joints per side, no waist). No pose17->pose23 conversion is applicable/needed here;
# state/action are passed straight through by `unitree_g1_joint_dataset_transform`.
# Only 3 of the source's 4 cameras are used (cam_right_high dropped), matching Unitree's
# own G1 pretraining setup -- see Masterprojekt/unifolm_vla_scratch_finetuning_g1_dex3_blockstacking.md.

# IMPORTANT: update this path to the directory containing the converted *.hdf5 files
# before running `tfds build`.
HDF5_DATA_DIR = "/home/omniverse-2/IsaacLab/Datasets/g1_dex3_blockstacking_hdf5"


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""

    def _parse_example(episode_path):
        with h5py.File(episode_path, "r") as F:
            actions = F["action"][:]
            states = F["observations"]["qpos"][:]
            images_primary_high = F["observations"]["images"]["cam_left_high"][:]
            images_left_wrist = F["observations"]["images"]["cam_left_wrist"][:]
            images_right_wrist = F["observations"]["images"]["cam_right_wrist"][:]

            language_raw_data = F["language_raw"]
            if language_raw_data.shape == ():
                language_instruction = str(language_raw_data[()])
            else:
                language_instruction = str(language_raw_data[0])

        episode = []
        for i in range(actions.shape[0]):
            episode.append({
                "observation": {
                    "image_primary_high": images_primary_high[i],
                    "image_left_wrist": images_left_wrist[i],
                    "image_right_wrist": images_right_wrist[i],
                    "state": np.asarray(states[i], np.float32),
                },
                "action": np.asarray(actions[i], dtype=np.float32),
                "discount": 1.0,
                "is_first": i == 0,
                "is_last": i == (actions.shape[0] - 1),
                "is_terminal": i == (actions.shape[0] - 1),
                "language_instruction": language_instruction,
            })

        sample = {
            "steps": episode,
            "episode_metadata": {
                "file_path": episode_path
            }
        }
        return episode_path, sample

    # For smallish datasets, use single-thread parsing
    for sample in paths:
        ret = _parse_example(sample)
        yield ret


class rlds_dataset_g1_dex3(MultiThreadedDatasetBuilder):
    """DatasetBuilder for unitreerobotics/G1_Dex3_BlockStacking_Dataset (joint-space, 3 cameras)."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    N_WORKERS = 8
    MAX_PATHS_IN_MEMORY = 8
    PARSE_FCN = _generate_examples

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image_primary_high": tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="Left head camera RGB observation (cam_left_high).",
                        ),
                        "image_left_wrist": tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="Left wrist camera RGB observation.",
                        ),
                        "image_right_wrist": tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="Right wrist camera RGB observation.",
                        ),
                        "state": tfds.features.Tensor(
                            shape=(28,),
                            dtype=np.float32,
                            doc="Robot joint state: 14 arm joints (7 left + 7 right) + 14 Dex3 hand joints (7 left + 7 right), no waist.",
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(28,),
                        dtype=np.float32,
                        doc="Robot joint action, same layout as state.",
                    ),
                    "discount": tfds.features.Scalar(
                        dtype=np.float32,
                        doc="Discount if provided, default to 1."
                    ),
                    "is_first": tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="True on first step of the episode."
                    ),
                    "is_last": tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="True on last step of the episode."
                    ),
                    "is_terminal": tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="True on last step of the episode if it is a terminal step, True for demos."
                    ),
                    "language_instruction": tfds.features.Text(
                        doc="Language Instruction."
                    ),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(
                        doc="Path to the original data file."
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        return {
            "train": glob.glob(os.path.join(HDF5_DATA_DIR, "*.hdf5")),
        }
