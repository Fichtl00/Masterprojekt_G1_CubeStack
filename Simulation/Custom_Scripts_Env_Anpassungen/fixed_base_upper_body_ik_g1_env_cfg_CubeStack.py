# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


'''Cube-Stacking Evaluation Environment fuer GR00T / UniVLA (G1, fixed base, upper body IK).

Zweck: Zweite, schlankere Eval-Umgebung neben
fixed_base_upper_body_ik_g1_env_cfg_Referenz_Korb_ausladen.py, um GR00T/UniVLA-VLA-Policies
am Standard-Benchmark-Task "Cube Stacking" (3 Wuerfel: rot/gelb/gruen) zu evaluieren,
statt am Referenz-spezifischen Korb-Task. Beobachtungs-/Aktionsformat (4 Kameras,
robot_joint_pos, left/right eef pos+quat, G1_UPPER_BODY_IK_ACTION_CFG) ist bewusst
identisch zur Referenz-Ausladen-Umgebung gehalten, damit dasselbe Eval-Script
(scripts/tools/eval_referenz_korb_ausladen.py bzw. eine Cube-Stack-Kopie davon) und
derselbe GR00T-ZMQ-Client ohne Anpassung weiterverwendet werden koennen.

Vorbereitet, NOCH NICHT ausgefuehrt/registriert-getestet (siehe
Anleitung/Masterprojekt/ fuer die vollstaendige Dokumentation dieser Umgebung).

start recording (Vorlage, Pfade ggf. anpassen):

    ./isaaclab.sh -p scripts/tools/record_demos.py \
  --device cuda \
  --task Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-CubeStack \
  --teleop_device handtracking \
  --dataset_file ./datasets/CubeStack/batch01/dataset_g1_cubestack.hdf5 \
  --num_demos 15 \
  --enable_pinocchio \
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

from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.pink_controller_cfg import (  # isort: skip
    G1_UPPER_BODY_IK_ACTION_CFG,
)

##
# Konstanten - Wuerfel (Cube Stacking, analog zu unitree_sim_isaaclab stack_rgyblock)
##

CUBE_SIZE: float = 0.045
CUBE_HALF: float = CUBE_SIZE / 2.0

# Tischhoehe/-position uebernommen von fixed_base_upper_body_ik_g1_env_cfg.py (Basisversion),
# da dort bereits gegen den G1-Arbeitsbereich (fixed base) kalibriert.
PACKING_TABLE_POS: list[float] = [0.3, 0.55, -0.3]
CUBE_TABLE_HEIGHT: float = 0.6996

# Startpositionen der 3 Wuerfel auf dem Tisch, nebeneinander mit 0.12 m Abstand entlang Y,
# damit sie beim ersten Szenenaufbau nicht ueberlappen. Werden pro Reset per EventTerm
# innerhalb eines kleinen, disjunkten Fensters zufaellig verschoben (siehe EventCfg unten).
CUBE_RED_INIT_POS: list[float] = [-0.25, 0.23, CUBE_TABLE_HEIGHT]
CUBE_YELLOW_INIT_POS: list[float] = [-0.25, 0.35, CUBE_TABLE_HEIGHT]
CUBE_GREEN_INIT_POS: list[float] = [-0.25, 0.47, CUBE_TABLE_HEIGHT]

# Sprachinstruktion fuer GR00T/UniVLA (identischer Language-Key wie in der Referenz-Ausladen-
# Konvertierung: annotation.human.action.task_description -- siehe
# Anleitung/closed_loop_evaluation_zusammenfassung.md).
TASK_DESCRIPTION: str = "Stack the red cube on the yellow cube, then stack the green cube on top."

##
# Kamera-Hilfsfunktionen (1:1 uebernommen aus
# fixed_base_upper_body_ik_g1_env_cfg_Referenz_Korb_ausladen.py, da dieselbe G1-Hardware/
# Kamerabestueckung -- siehe dortige Kommentare fuer Details zur Kalibrierung)
##

_CAMERA_RECORD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_recordings")

_cameras_enabled_cache: bool | None = None


def _cameras_enabled() -> bool:
    """Ob dieser Prozess mit ``--enable_cameras`` (oder ``ENABLE_CAMERAS=1``) gestartet wurde."""
    global _cameras_enabled_cache
    if _cameras_enabled_cache is None:
        _cameras_enabled_cache = "--enable_cameras" in sys.argv or os.environ.get(
            "ENABLE_CAMERAS", "0"
        ) == "1"
    return _cameras_enabled_cache


def _get_sensor_rgb(env, sensor_names):
    """Gibt das erste verfuegbare Sensor-RGB als numpy-Array im CHW-Format (3, H, W) zurueck."""
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
                return _np.transpose(arr, (2, 0, 1))  # HWC -> CHW

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


# Channel-first (C, H, W) float32 in [0, 1] -- identisches Format zur Referenz-Ausladen-Env,
# damit RLinf's GR00T-Obs-Konverter (convert_referenz_obs_to_gr00t_format) unveraendert
# weiterverwendet werden kann.
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
# Reset-Funktion: 3 Wuerfel unabhaengig innerhalb disjunkter Fenster neu positionieren
##

def reset_cubes_disjoint(env, env_ids: torch.Tensor) -> None:
    """Setzt rot/gelb/gruen auf ihre Basisposition + kleinen Zufalls-Offset zurueck.

    Die Fenster je Wuerfel (siehe CUBE_*_INIT_POS, 0.12 m Y-Abstand) sind schmaler als der
    Basisabstand, damit die Wuerfel nach dem Reset nie ueberlappen -- der Roboter muss den
    Stack in jeder Episode von Grund auf neu bauen (kein Fixed-Distance-Pairing wie im
    unitree_sim_isaaclab-Referenz-Task, siehe stack_rgyblock_g1_29dof_dex3_joint_env_cfg.py).
    """
    device = env.device
    N = len(env_ids)

    for name, init_pos in (
        ("cube_red", CUBE_RED_INIT_POS),
        ("cube_yellow", CUBE_YELLOW_INIT_POS),
        ("cube_green", CUBE_GREEN_INIT_POS),
    ):
        cube = env.scene[name]
        state = cube.data.default_root_state[env_ids].clone()

        offset_xy = torch.empty(N, 2, device=device).uniform_(-0.03, 0.03)
        state[:, 0] = init_pos[0] + offset_xy[:, 0]
        state[:, 1] = init_pos[1] + offset_xy[:, 1]
        state[:, 2] = init_pos[2]
        state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        state[:, 7:13] = 0.0  # Geschwindigkeiten nullen

        cube.write_root_state_to_sim(state, env_ids)


##
# Erfolgs-Check: rot->gelb->gruen gestapelt (ohne Greifer-Zustandscheck, da G1-Trihand keinen
# einfachen "open_val" wie ein Parallelgreifer hat -- siehe object_stacked/cubes_stacked in
# isaaclab_tasks.manager_based.manipulation.stack.mdp, die genau das voraussetzen)
##

def cubes_stacked_trihand(
    env,
    cube_bottom_cfg: SceneEntityCfg = SceneEntityCfg("cube_red"),
    cube_middle_cfg: SceneEntityCfg = SceneEntityCfg("cube_yellow"),
    cube_top_cfg: SceneEntityCfg = SceneEntityCfg("cube_green"),
    xy_threshold: float = 0.03,
    height_threshold: float = 0.01,
    height_diff: float = CUBE_SIZE,
) -> torch.Tensor:
    """Bool-Tensor (N,): rot-gelb UND gelb-gruen jeweils xy-ausgerichtet und um genau
    eine Wuerfelhoehe versetzt gestapelt."""
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
    """Scene configuration fuer die Cube-Stacking-Eval-Umgebung mit G1 (fixed base, upper body IK)."""

    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=PACKING_TABLE_POS, rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    cube_red = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CubeRed",
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_RED_INIT_POS, rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, retain_accelerations=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.01, rest_offset=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
        ),
    )

    cube_yellow = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CubeYellow",
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_YELLOW_INIT_POS, rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, retain_accelerations=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.01, rest_offset=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
        ),
    )

    cube_green = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CubeGreen",
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_GREEN_INIT_POS, rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, retain_accelerations=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.01, rest_offset=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
        ),
    )

    # Unitree G1 Humanoid - fixed base
    robot: ArticulationCfg = G1_29DOF_CFG

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Kopf-Stereokameras (identische Kalibrierung wie in der Referenz-Ausladen-Umgebung, siehe
    # dortige Kommentare zu convention="ros" und den empirisch ermittelten Offsets)
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
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.1, 1.0e5)),
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
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.1, 1.0e5)),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.04012, 0.07441, 0.15711),
            rot=(0.00539, 0.86024, 0.0424, 0.50809),
            convention="ros",
        ),
    )

    def __post_init__(self):
        """Post initialization."""
        self.robot.spawn.articulation_props.fix_root_link = True

        # Greifkraft/Reibung der Hand wie in der Referenz-Ausladen-Umgebung erhoehen, damit die
        # (im Vergleich zum Korb-Inlay viel leichteren, aber kleineren) Wuerfel sicher
        # gehalten werden koennen.
        hands_actuator = self.robot.actuators["hands"]
        hands_actuator.stiffness = 100.0  # standard = 20.0 (5x)
        hands_actuator.damping = 4.47     # standard = 2.0 (~sqrt(5)x)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = G1_UPPER_BODY_IK_ACTION_CFG


@configclass
class EventCfg:
    """Event-Terms fuer die MDP."""

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
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

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

        # Kamera-Observations (CHW, 3x480x640) -- identische Schluessel wie in der
        # Referenz-Ausladen-Umgebung, siehe deren Docstring fuer das erwartete GR00T-Format.
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
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)

    cube_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("cube_red")},
    )

    success = DoneTerm(func=cubes_stacked_trihand)


@configclass
class FixedBaseUpperBodyIKG1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 fixed base upper body IK cube-stacking eval environment."""

    scene: FixedBaseUpperBodyIKG1SceneCfg = FixedBaseUpperBodyIKG1SceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )

    terminations: TerminationsCfg = TerminationsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    # Ungenutzte Manager
    commands = None
    rewards = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, -0.35),
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
