# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


'''Cube-Stacking Teleop-Aufnahme-Umgebung fuer G1 + Dex3 (fixed base, upper body Pink-IK).
Task-Gruppe: ``1_cube_stack``.

Klon von fixed_base_upper_body_ik_g1_env_cfg_CubeStack.py (Pink-IK + OpenXR-Handtracking-
Teleop, dort bereits erfolgreich getestet). Kamera-Config ist UNVERAENDERT von dort
uebernommen (Referenz-Kalibrierung, ROS-Convention, an torso_link/Wrist-Links montiert).

Aus https://github.com/Docboter/projektarbeit_humanoider_roboter (Branch
training-luca-IKR-IS6.0, Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py) NUR die
Szenen-Objekte + Reset-Logik uebernommen:
  - Tisch als einfache Box (0.8 x 0.6 x 0.87 m) statt Nucleus-``packing_table.usd``.
  - 3 Wuerfel (5 cm statt 4.5 cm), Positionen/Farben aus dem Original.
  - Schwarzes Stapel-Band als Platzier-Landmarke (im Dataset vorhanden, in der bisherigen
    Referenz-Cube-Stack-Szene nicht).
  - Randomisierter Reset: jeder Wuerfel bekommt ein EIGENES, disjunktes y-Band innerhalb
    von block_y_range (statt Referenz's +-0.03 m Jitter um 3 fixe, 0.12 m auseinanderliegende
    Positionen) -- verhindert Ueberlappung beim Reset unabhaengig von der Anzahl Wuerfel.

Roboter-Config: G1_DEX3_CFG_1_CUBE_STACK (g1_dex3_cfg_1_cube_stack.py, ebenfalls aus dem
Gruppe-1-Repo) -- eigene Hand-Aktuator-Gains (empirisch fuer Dex3-Fingerkraft kalibriert)
+ Startpose aus Frame 0 des echten unitreerobotics/G1_Dex3_BlockStacking_Dataset.

NICHT uebernommen (siehe Nutzer-Anweisung "nur Szenen-Env + Cube-Reset + Objekte/Umgebung"):
Kameras (bleiben Referenz), sowie die RL-spezifische Env-Logik des Originals (g1_dex3_blockstack_env.py)
-- Domain-Randomization, Reach-Diagnostik, geshapte Rewards, USD-Kamera-Orientierungs-Fix,
Finger-Limit-Weitung, Hand-Schwarz-Recolor. Diese Env nutzt wie ihr Vorbild die Standard-
ObservationsCfg/EventCfg/TerminationsCfg-Manager statt eigener RL-Logik.

start recording (Vorlage, Pfade ggf. anpassen):

    ./isaaclab.sh -p scripts/tools/record_demos.py \
  --device cuda \
  --task Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-1_cube_stack \
  --teleop_device handtracking \
  --dataset_file ./datasets/1_cube_stack/batch01/dataset_g1_1_cube_stack.hdf5 \
  --num_demos 30 \
  --enable_pinocchio \
  --enable_cameras \
  --headless
'''

import os
import sys

import numpy as np
import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.trihand.g1_upper_body_retargeter import (
    G1TriHandUpperBodyRetargeterCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, save_images_to_file
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.math import sample_uniform

from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp

from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.pink_controller_cfg import (  # isort: skip
    G1_UPPER_BODY_IK_ACTION_CFG,
)

# Portierte Roboter-Config (Actuator-Gains + Dataset-Startpose) -- siehe Docstring oben.
from isaaclab_tasks.manager_based.locomanipulation.pick_place.g1_dex3_cfg_1_cube_stack import (
    G1_DEX3_CFG_1_CUBE_STACK,
)

##
# Konstanten - Tisch/Wuerfel/Stapel-Band, portiert aus g1_dex3_blockstack_env.py
# (Simulation/g1_dex3_sim/, siehe Docstring). Ersetzt die bisherigen Referenz-Konstanten
# (PACKING_TABLE_POS/CUBE_TABLE_HEIGHT/CUBE_*_INIT_POS) fuer diese Task-Gruppe.
##

# Tisch + Wuerfel 10 cm tiefer als vorherige Version gespawnt (Nutzer-Feedback), danach um
# 5 cm wieder angehoben (weiteres Nutzer-Feedback) -- Tischoberflaeche jetzt bei z=0.82.
# Wuerfel zudem von 5 cm ueber 4,5 cm auf 4 cm verkleinert.
TABLE_SIZE: tuple[float, float, float] = (0.8, 0.6, 0.87)
TABLE_POS: tuple[float, float, float] = (0.5, 0.0, 0.385)  # Tischoberflaeche bei z=0.82

CUBE_SIZE: float = 0.04
CUBE_RED_INIT_POS: list[float] = [0.34, -0.15, 0.865]
CUBE_GREEN_INIT_POS: list[float] = [0.36, 0.0, 0.865]
CUBE_YELLOW_INIT_POS: list[float] = [0.34, 0.15, 0.865]

# Stapel-Band-Markierung: EIN einzelnes Band, um 90 Grad um Z gedreht und halb so gross wie
# zuvor (Nutzer-Feedback), auf Tischoberflaeche z=0.82 platziert.
STACK_BAND_SIZE: tuple[float, float, float] = (0.06, 0.02, 0.003)
STACK_BAND_POS: tuple[float, float, float] = (0.35, 0.0, 0.8215)
# 90-Grad-Drehung um Z (w, x, y, z).
STACK_BAND_ROT: tuple[float, float, float, float] = (0.70710678, 0.0, 0.0, 0.70710678)
# Nach der Drehung liegt STACK_BAND_SIZE[1] (Breite) entlang Welt-X, STACK_BAND_SIZE[0]
# (Laenge) entlang Welt-Y -- fuer die Ablagebereich-Sperrzone unten verwendet.
STACK_BAND_WORLD_X_EXTENT: float = STACK_BAND_SIZE[1]
STACK_BAND_WORLD_Y_EXTENT: float = STACK_BAND_SIZE[0]

# Reset-Sampling-Bereich (erreichbarer Greifraum, aus dem Original uebernommen).
BLOCK_X_RANGE: tuple[float, float] = (0.30, 0.40)
BLOCK_Y_RANGE: tuple[float, float] = (-0.20, 0.20)
BLOCK_Z_SURFACE: float = 0.865

# Sperrzone (Ablagebereich) um das Stapel-Band -- Wuerfel duerfen dort nicht spawnen (Nutzer-
# Feedback: sonst muss der Ablagebereich vor dem Stapeln erst freigeraeumt werden).
STACK_EXCLUSION_HALF_X: float = STACK_BAND_WORLD_X_EXTENT / 2.0 + CUBE_SIZE / 2.0 + 0.01
STACK_EXCLUSION_HALF_Y: float = STACK_BAND_WORLD_Y_EXTENT / 2.0 + CUBE_SIZE / 2.0 + 0.01

TASK_DESCRIPTION: str = "Stack the cubes: red on the bottom, yellow in the middle, green on top."

##
# Kamera-Hilfsfunktionen (UNVERAENDERT aus fixed_base_upper_body_ik_g1_env_cfg_CubeStack.py --
# Kamera-Config bleibt Referenz, siehe Docstring oben)
##

_cameras_enabled_cache: bool | None = None


def _cameras_enabled() -> bool:
    global _cameras_enabled_cache
    if _cameras_enabled_cache is None:
        _cameras_enabled_cache = "--enable_cameras" in sys.argv or os.environ.get(
            "ENABLE_CAMERAS", "0"
        ) == "1"
    return _cameras_enabled_cache


def _get_sensor_rgb(env, sensor_names):
    import numpy as _np

    if not _cameras_enabled():
        return _np.zeros((3, 480, 640), dtype=_np.uint8)

    scene_keys = list(env.scene.keys())

    for name in sensor_names:
        if name in scene_keys and hasattr(env.scene[name], "data"):
            try:
                img = env.scene[name].data.output["rgb"][0]
            except Exception:
                continue
            try:
                arr = img.cpu().numpy() if hasattr(img, "cpu") else img
            except Exception:
                continue
            if isinstance(arr, _np.ndarray) and arr.ndim == 3 and arr.shape[2] == 3:
                return _np.transpose(arr, (2, 0, 1))

    if hasattr(env.scene, "sensors") and env.scene.sensors:
        for sensor in env.scene.sensors.values():
            try:
                img = sensor.data.output["rgb"][0]
            except Exception:
                continue
            try:
                arr = img.cpu().numpy() if hasattr(img, "cpu") else img
            except Exception:
                continue
            if isinstance(arr, _np.ndarray) and arr.ndim == 3 and arr.shape[2] == 3:
                return _np.transpose(arr, (2, 0, 1))

    import numpy as _np
    return _np.zeros((3, 480, 640), dtype=_np.uint8)


def get_cam_left_high(env):
    img = _get_sensor_rgb(env, ("cam_left_high", "front_camera", "PerspectiveCamera_robot", "d435_link"))
    n = getattr(env, "num_envs", 1)
    t = torch.from_numpy(img).float() / 255.0
    return t.unsqueeze(0).repeat(n, 1, 1, 1)


def get_cam_right_high(env):
    img = _get_sensor_rgb(env, ("cam_right_high", "right_front_camera", "PerspectiveCamera_robot", "d435_link"))
    n = getattr(env, "num_envs", 1)
    t = torch.from_numpy(img).float() / 255.0
    return t.unsqueeze(0).repeat(n, 1, 1, 1)


def get_cam_left_wrist(env):
    img = _get_sensor_rgb(env, ("cam_left_wrist", "left_wrist_camera"))
    n = getattr(env, "num_envs", 1)
    t = torch.from_numpy(img).float() / 255.0
    return t.unsqueeze(0).repeat(n, 1, 1, 1)


def get_cam_right_wrist(env):
    img = _get_sensor_rgb(env, ("cam_right_wrist", "right_wrist_camera"))
    n = getattr(env, "num_envs", 1)
    t = torch.from_numpy(img).float() / 255.0
    return t.unsqueeze(0).repeat(n, 1, 1, 1)


##
# Reset-Funktion -- PORTIERT aus g1_dex3_blockstack_env.py::_reset_idx (Gruppe 1): jeder
# Wuerfel bekommt ein eigenes, disjunktes y-Band innerhalb BLOCK_Y_RANGE, x bleibt voll
# randomisiert. Ersetzt die bisherige Referenz-Variante (+-0.03 m Jitter um 3 feste Positionen).
##

def reset_cubes_disjoint(env, env_ids: torch.Tensor) -> None:
    device = env.device
    cube_names = ["cube_red", "cube_green", "cube_yellow"]
    n_blocks = len(cube_names)
    y_lo, y_hi = BLOCK_Y_RANGE
    slot = (y_hi - y_lo) / n_blocks
    margin = min(0.03, 0.4 * slot)
    band_x, band_y = STACK_BAND_POS[0], STACK_BAND_POS[1]
    n = len(env_ids)

    for i, name in enumerate(cube_names):
        cube = env.scene[name]
        state = cube.data.default_root_state[env_ids].clone()

        y_min = y_lo + i * slot + margin
        y_max = y_lo + (i + 1) * slot - margin
        x = sample_uniform(BLOCK_X_RANGE[0], BLOCK_X_RANGE[1], (n,), device=device)
        y = sample_uniform(y_min, y_max, (n,), device=device)

        # Sperrzone um das Stapel-Band ausschliessen (Rejection-Sampling): verhindert, dass
        # Wuerfel im Ablagebereich spawnen und vorher zur Seite geraeumt werden muessen.
        for _ in range(20):
            violates = ((x - band_x).abs() < STACK_EXCLUSION_HALF_X) & (
                (y - band_y).abs() < STACK_EXCLUSION_HALF_Y
            )
            if not violates.any():
                break
            n_violate = int(violates.sum())
            x[violates] = sample_uniform(BLOCK_X_RANGE[0], BLOCK_X_RANGE[1], (n_violate,), device=device)
            y[violates] = sample_uniform(y_min, y_max, (n_violate,), device=device)

        state[:, 0] = x
        state[:, 1] = y
        state[:, 2] = BLOCK_Z_SURFACE
        state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        state[:, 7:13] = 0.0

        cube.write_root_state_to_sim(state, env_ids)


##
# Erfolgs-Check (unveraendert aus fixed_base_upper_body_ik_g1_env_cfg_CubeStack.py)
##

def cubes_stacked_trihand(
    env,
    # Stapel-Reihenfolge (Nutzer-Vorgabe): 1. rot (unten), 2. gelb (Mitte), 3. gruen (oben).
    cube_bottom_cfg: SceneEntityCfg = SceneEntityCfg("cube_red"),
    cube_middle_cfg: SceneEntityCfg = SceneEntityCfg("cube_yellow"),
    cube_top_cfg: SceneEntityCfg = SceneEntityCfg("cube_green"),
    xy_threshold: float = 0.03,
    height_threshold: float = 0.01,
    height_diff: float = CUBE_SIZE,
) -> torch.Tensor:
    bottom = env.scene[cube_bottom_cfg.name]
    middle = env.scene[cube_middle_cfg.name]
    top = env.scene[cube_top_cfg.name]

    def _stacked(lower, upper) -> torch.Tensor:
        pos_diff = upper.data.root_pos_w - lower.data.root_pos_w
        xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
        height_dist = pos_diff[:, 2]
        return torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff).abs() < height_threshold)

    return torch.logical_and(_stacked(bottom, middle), _stacked(middle, top))


##
# Scene definition
##

@configclass
class FixedBaseUpperBodyIKG1SceneCfg(InteractiveSceneCfg):
    """Scene configuration -- Task-Gruppe 1_cube_stack (G1 + Dex3, fixed base, Pink-IK).

    Tisch/Wuerfel/Stapel-Band aus dem Gruppe-1-Repo (siehe Modul-Docstring); Kameras
    unveraendert von der bestehenden Referenz-CubeStack-Env uebernommen.
    """

    # GEÄNDERT ggü. ...CubeStack.py: einfache Box statt Nucleus-``packing_table.usd``
    # (portiert aus g1_dex3_blockstack_env.py).
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.85, 0.85)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_POS),
    )

    cube_red = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CubeRed",
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_RED_INIT_POS, rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, retain_accelerations=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.01, rest_offset=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1), metallic=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max", restitution_combine_mode="min",
                static_friction=3.0, dynamic_friction=2.5, restitution=0.0,
            ),
        ),
    )

    cube_green = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CubeGreen",
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_GREEN_INIT_POS, rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, retain_accelerations=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.01, rest_offset=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.1), metallic=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max", restitution_combine_mode="min",
                static_friction=3.0, dynamic_friction=2.5, restitution=0.0,
            ),
        ),
    )

    cube_yellow = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CubeYellow",
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_YELLOW_INIT_POS, rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, retain_accelerations=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.01, rest_offset=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.70, 0.10), metallic=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max", restitution_combine_mode="min",
                static_friction=3.0, dynamic_friction=2.5, restitution=0.0,
            ),
        ),
    )

    # NEU ggü. ...CubeStack.py: Stapel-Band als Platzier-Landmarke (aus dem Gruppe-1-Repo --
    # im Dataset vorhanden, dort wird auf einen kleinen schwarzen Streifen gestapelt).
    # Zwischenzeitlich zu einem 4-Baender-Rahmen umgebaut, auf Nutzer-Feedback wieder auf ein
    # einzelnes Band zurueckgebaut.
    stack_band = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/StackBand",
        spawn=sim_utils.CuboidCfg(
            size=STACK_BAND_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            # Kein Kollisionskoerper -- rein visuelle Platzier-Landmarke, kein Hindernis.
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.02, 0.02)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=STACK_BAND_POS, rot=STACK_BAND_ROT),
    )

    # GEÄNDERT ggü. ...CubeStack.py: G1_DEX3_CFG_1_CUBE_STACK (portierte Actuator-Gains +
    # Dataset-Startpose) statt G1_29DOF_CFG. prim_path fehlt in der portierten Config
    # (das Original setzt es an anderer Stelle) -- hier ergaenzt, analog zu G1_29DOF_CFG.
    robot: ArticulationCfg = G1_DEX3_CFG_1_CUBE_STACK.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Kopf-Stereokameras -- UNVERAENDERT aus fixed_base_upper_body_ik_g1_env_cfg_CubeStack.py
    # (Referenz-Kalibrierung, ROS-Convention, am torso_link/d435_link montiert).
    cam_left_high = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/cam_left_high",
        update_period=1.0 / 30.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12, clipping_range=(0.1, 1.0e5)),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.03, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    cam_right_high = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/cam_right_high",
        update_period=1.0 / 30.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12, clipping_range=(0.1, 1.0e5)),
        offset=CameraCfg.OffsetCfg(
            pos=(0.03, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    cam_left_wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link/cam_left_wrist",
        update_period=1.0 / 30.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=10.0, clipping_range=(0.1, 1.0e5)),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.04012, -0.07441, 0.15711),
            rot=(0.00539, 0.86024, 0.0424, 0.50809),
            convention="ros",
        ),
    )

    cam_right_wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/cam_right_wrist",
        update_period=1.0 / 30.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=10.0, clipping_range=(0.1, 1.0e5)),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.04012, 0.07441, 0.15711),
            rot=(0.00539, 0.86024, 0.0424, 0.50809),
            convention="ros",
        ),
    )

    # KEIN __post_init__ mit generischem Hand-Stiffness-Boost mehr noetig -- die portierte
    # G1_DEX3_CFG_1_CUBE_STACK bringt bereits eigene, empirisch kalibrierte Hand-Gains mit.


@configclass
class ActionsCfg:
    """Action specifications for the MDP -- unveraendert: Pink-IK wie ...CubeStack.py."""

    upper_body_ik = G1_UPPER_BODY_IK_ACTION_CFG


@configclass
class EventCfg:
    """Event-Terms fuer die MDP."""

    # GEÄNDERT ggü. ...CubeStack.py: disjunkte y-Baender statt +-0.03m-Jitter (s. oben).
    reset_cubes = EventTerm(
        func=reset_cubes_disjoint,
        mode="reset",
    )

    increase_gripper_friction = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=r"(left|right)_hand_(palm|thumb_[0-2]|index_[01]|middle_[01])_link"
            ),
            "static_friction_range": (1.5, 1.5),
            "dynamic_friction_range": (1.2, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "make_consistent": True,
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP (unveraendert aus ...CubeStack.py)."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=manip_mdp.last_action)
        robot_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})

        cube_red_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube_red")})
        cube_red_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube_red")})
        cube_yellow_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube_yellow")})
        cube_yellow_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube_yellow")})
        cube_green_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube_green")})
        cube_green_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube_green")})

        robot_links_state = ObsTerm(func=manip_mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "left_wrist_yaw_link"})
        left_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "left_wrist_yaw_link"})
        right_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "right_wrist_yaw_link"})
        right_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "right_wrist_yaw_link"})

        hand_joint_state = ObsTerm(func=manip_mdp.get_robot_joint_state, params={"joint_names": [".*_hand.*"]})
        head_joint_state = ObsTerm(func=manip_mdp.get_robot_joint_state, params={"joint_names": []})

        cam_left_high = ObsTerm(func=get_cam_left_high)
        cam_right_high = ObsTerm(func=get_cam_right_high)
        cam_left_wrist = ObsTerm(func=get_cam_left_wrist)
        cam_right_wrist = ObsTerm(func=get_cam_right_wrist)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP (unveraendert aus ...CubeStack.py)."""

    time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)

    cube_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("cube_red")},
    )

    success = DoneTerm(func=cubes_stacked_trihand)


@configclass
class FixedBaseUpperBodyIKG1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration -- Task-Gruppe 1_cube_stack (G1 + Dex3, fixed base, Pink-IK, Teleop)."""

    scene: FixedBaseUpperBodyIKG1SceneCfg = FixedBaseUpperBodyIKG1SceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )

    terminations: TerminationsCfg = TerminationsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    commands = None
    rewards = None
    curriculum = None

    # anchor_pos.z: -0.35 (...CubeStack.py) -> -0.45 (Ausgleich Pelvis-Hoehe) war laut
    # Nutzer-Feedback zu niedrig ueberkorrigiert (Spawnpunkt danach hoeher als vorher --
    # anchor_pos.z ist der Offset zwischen XR-Ursprung und Robotersicht, ein staerker
    # negativer Wert hebt den wahrgenommenen Spawnpunkt an, kein Absenken). Auf -0.25
    # korrigiert.
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, -0.25),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 200  # 200 Hz
        self.sim.render_interval = 2

        urdf_omniverse_path = (
            f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/"
            "unitree_g1_kinematics_asset/g1_29dof_with_hand_only_kinematics.urdf"
        )
        self.actions.upper_body_ik.controller.urdf_path = retrieve_file_path(urdf_omniverse_path)

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        G1TriHandUpperBodyRetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
