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

# Co-training counterpart to rlds_dataset_g1_dex3.py: same 28-dim joint-space
# state/action layout, same 3 cameras (cam_left_high/cam_left_wrist/cam_right_wrist),
# same HDF5 schema -- but sourced from Fichtl00/Cube_Stacking_synth (our own
# CloudXR/OpenXR-teleoperated Isaac Lab recordings, converted via the SAME
# convert_lerobot_to_hdf5_g1_dex3.py) instead of the real-robot dataset. Registered
# as its own named TFDS dataset so it can be combined with the real dataset in an
# OXE mixture (see mixtures.py: g1_dex3_blockstacking_cotrain).

# IMPORTANT: update this path to the directory containing the converted *.hdf5 files
# before running `tfds build` -- entrypoint.sh patches this via sed, siehe dortigen
# Kommentar zu HDF5_DATA_DIR.
HDF5_DATA_DIR = "/home/omniverse-2/IsaacLab/Datasets/g1_dex3_cubestacking_synth_hdf5"


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


class rlds_dataset_g1_dex3_synth(MultiThreadedDatasetBuilder):
    """DatasetBuilder for Fichtl00/Cube_Stacking_synth (joint-space, 3 cameras, teleoperated sim data)."""

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
