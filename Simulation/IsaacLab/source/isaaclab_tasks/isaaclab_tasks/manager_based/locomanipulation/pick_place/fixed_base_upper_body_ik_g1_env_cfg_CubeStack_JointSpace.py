# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


'''Cube-Stacking Eval-Umgebung mit G1 + Dex3-Hand, direkte Gelenkwinkel-Steuerung
(kein Pink-IK), speziell fuer das Testen des vortrainierten UniFolm-VLA.

Dritte Variante neben:
- fixed_base_upper_body_ik_g1_env_cfg_CubeStack.py (Dex3 + Pink-IK/6D-Pose, fuer GR00T --
  bleibt UNVERAENDERT, siehe Anleitung/Masterprojekt/cube_stack_eval_env.md).
- fixed_base_upper_body_ik_g1_env_cfg_CubeStack_ParallelGripper.py (Dex1-Parallelgreifer +
  Gelenkwinkel-Steuerung).

GRUNDLAGE (per LeRobot-Dataset-Metadaten des Nutzers verifiziert, nicht geraten -- siehe
Anleitung/Masterprojekt/cube_stack_eval_env.md fuer das vollstaendige Metadaten-JSON):
Der UniFolm-VLA-Checkpoint wurde auf einem Datensatz trainiert, dessen
`observation.state`- UND `action`-Feld exakt 28-dim sind, mit expliziter Namensliste:

    14 Arm-Gelenke (7 pro Seite, IN DIESER REIHENFOLGE):
      {Left,Right}ShoulderPitch, {Left,Right}ShoulderRoll, {Left,Right}ShoulderYaw,
      {Left,Right}Elbow, {Left,Right}WristRoll, {Left,Right}WristPitch,
      {Left,Right}WristYaw
    14 Dex3-Fingergelenke (7 pro Hand, ASYMMETRISCHE Reihenfolge links/rechts!):
      links:  Thumb0, Thumb1, Thumb2, Middle0, Middle1, Index0, Index1
      rechts: Thumb0, Thumb1, Thumb2, Index0,  Index1,  Middle0, Middle1

KEINE Taille im Datensatz (nur 28 = 14+14, keine 3 Taille-Gelenke) und KEIN
EE-Pose-Format -- der Datensatz bestaetigt damit die Nutzer-Aussage: dieser
UniFolm-VLA-Checkpoint ist echtes rohes Joint-Space (nicht das generische
`EE_R6_G1`-Pose-Encoding, das an anderer Stelle im unifolm-vla-Repo als Default fuer den
"g1_stack_block"-Datensatznamen registriert ist -- dieser konkrete Checkpoint/Datensatz
verwendet offenbar eine andere/eigene Konfiguration).

WICHTIG -- Reihenfolge ist sicherheitsrelevant, nicht nur kosmetisch: `JointPositionActionCfg`
ordnet das Aktions-/Beobachtungs-Array standardmaessig (`preserve_order=False`) nach der
INTERNEN Gelenkreihenfolge der Artikulation, NICHT nach der Reihenfolge der uebergebenen
`joint_names`-Liste (siehe isaaclab.utils.string.resolve_matching_names). Deshalb hier
`preserve_order=True` gesetzt und `DEX3_ARM_HAND_JOINT_NAMES_ORDERED` als exakte,
literale Gelenknamen-Liste (keine gruppierenden Regexe wie ".*_shoulder_pitch_joint")
in GENAU der oben dokumentierten Datensatz-Reihenfolge definiert. Fuer die
Beobachtungsseite (`observation.state`-Aequivalent) sorgt die eigene Funktion
`get_ordered_joint_pos` (per Namens-Index-Lookup, nicht ueber find_joints) fuer dieselbe
garantierte Reihenfolge.

Kein Hand-Tracking-/Motion-Controller-Teleop verdrahtet (wie bei der
ParallelGripper-Variante): ohne Pink-IK gibt es keinen eingebauten Retargeter, der
Hand-Tracking-Posen auf rohe Gelenkwinkel abbildet. Diese Env ist fuer skriptgesteuerte
Policy-Evaluation gedacht, nicht fuer Teleoperation.

Kamera-/Szenen-Setup 1:1 von fixed_base_upper_body_ik_g1_env_cfg_CubeStack.py uebernommen
(gleicher Roboter G1_29DOF_CFG/Dex3-USD, dort bereits per Teleop erfolgreich getestet).
Kamera-Namen (cam_left_high/cam_right_high/cam_left_wrist/cam_right_wrist) stimmen 1:1
mit den `observation.images.*`-Keys des Trainings-Datensatzes ueberein -- keine Anpassung
noetig.

Vorbereitet, NOCH NICHT ausgefuehrt/getestet.
'''

import os
import sys

import numpy as np
import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

##
# Action/Observation-Gelenkreihenfolge -- EXAKT wie im LeRobot-Trainingsdatensatz des
# UniFolm-VLA-Checkpoints (observation.state/action, beide 28-dim, siehe Docstring oben).
# Literale Gelenknamen (keine gruppierenden Regexe) + preserve_order=True in ActionsCfg,
# damit die Reihenfolge garantiert der Datensatz-Reihenfolge entspricht und nicht der
# internen Artikulations-Gelenkreihenfolge (Isaac Lab Default ohne preserve_order).
##

DEX3_ARM_HAND_JOINT_NAMES_ORDERED: list[str] = [
    # Arme (14) -- {Left,Right}ShoulderPitch/Roll/Yaw, Elbow, WristRoll/Pitch/Yaw
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
    # Dex3-Finger (14) -- ACHTUNG asymmetrisch: links Thumb->Middle->Index,
    # rechts Thumb->Index->Middle (exakt wie im Datensatz-Schema, nicht vereinheitlichen!)
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]


def get_ordered_joint_pos(env, joint_names_ordered: list[str]) -> torch.Tensor:
    """Gibt robot.joint_pos in EXAKT der uebergebenen Reihenfolge zurueck (per
    Namens-Index-Lookup) -- im Gegensatz zu ``find_joints()``/``get_robot_joint_state()``,
    die ohne ``preserve_order=True`` nach der internen Artikulations-Gelenkreihenfolge
    sortieren, nicht nach der Reihenfolge der uebergebenen Liste."""
    robot = env.scene["robot"]
    joint_ids = [robot.data.joint_names.index(name) for name in joint_names_ordered]
    return robot.data.joint_pos[:, joint_ids]

##
# Konstanten - Wuerfel (identisch zur Pink-IK-CubeStack-Variante, siehe dortigen Kommentar)
##

CUBE_SIZE: float = 0.045
PACKING_TABLE_POS: list[float] = [0.3, 0.55, -0.3]
CUBE_TABLE_HEIGHT: float = 0.6996

CUBE_RED_INIT_POS: list[float] = [-0.25, 0.23, CUBE_TABLE_HEIGHT]
CUBE_YELLOW_INIT_POS: list[float] = [-0.25, 0.35, CUBE_TABLE_HEIGHT]
CUBE_GREEN_INIT_POS: list[float] = [-0.25, 0.47, CUBE_TABLE_HEIGHT]

TASK_DESCRIPTION: str = "Stack the red cube on the yellow cube, then stack the green cube on top."

##
# Kamera-Hilfsfunktionen (siehe fixed_base_upper_body_ik_g1_env_cfg_CubeStack.py fuer
# Details/Herkunft -- unveraendert uebernommen, gleicher Roboter/gleiche Kameramontage)
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
# Reset-/Erfolgs-Funktionen (identisch zur Pink-IK-CubeStack-Variante)
##

def reset_cubes_disjoint(env, env_ids: torch.Tensor) -> None:
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
        state[:, 7:13] = 0.0

        cube.write_root_state_to_sim(state, env_ids)


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
    eine Wuerfelhoehe versetzt gestapelt (ohne Greifer-Offen-Check, siehe Pink-IK-Variante
    fuer die Begruendung)."""
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
    """Scene configuration fuer die Cube-Stacking-Eval-Umgebung mit G1 (Dex3, Joint-Space)."""

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

    # Unitree G1 Humanoid mit Dex3-Hand - fixed base
    robot: ArticulationCfg = G1_29DOF_CFG

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Kopf-Stereokameras (identische Kalibrierung wie in der Pink-IK-CubeStack-Variante)
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

    # Top-Down-Kamera -- NUR zur Visualisierung/Video-Aufzeichnung (z.B.
    # cubestack_jointspace_unifolm_vla_bridge.py), bewusst NICHT Teil von ObservationsCfg,
    # damit sich das 4-Kamera-Beobachtungsschema fuer die Policy nicht aendert. Blickt von
    # oben (Weltrichtung -Z) auf den Arbeitsbereich (Tisch + Wuerfel + Roboterarme).
    cam_top_down = CameraCfg(
        prim_path="/World/envs/env_.*/CamTopDown",
        update_period=1.0 / 30.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=8.0, clipping_range=(0.1, 10.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.1, 0.35, 2.0),
            rot=(0.0, 1.0, 0.0, 0.0),  # 180 Grad um X: blickt von oben nach unten (-Z)
            convention="ros",
        ),
    )

    def __post_init__(self):
        """Post initialization."""
        self.robot.spawn.articulation_props.fix_root_link = True

        # Greifkraft/Reibung der Hand erhoehen (identisch zur Pink-IK-CubeStack-Variante),
        # damit die Wuerfel sicher gehalten werden koennen.
        hands_actuator = self.robot.actuators["hands"]
        hands_actuator.stiffness = 100.0  # standard = 20.0 (5x)
        hands_actuator.damping = 4.47     # standard = 2.0 (~sqrt(5)x)


@configclass
class ActionsCfg:
    """Action specifications for the MDP -- direkte Gelenkwinkel-Steuerung (kein Pink-IK,
    siehe Docstring am Dateianfang)."""

    upper_body_joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=DEX3_ARM_HAND_JOINT_NAMES_ORDERED,
        preserve_order=True,  # sonst Artikulations-Reihenfolge statt Datensatz-Reihenfolge
        scale=1.0,
        use_default_offset=True,
    )


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

        # 28-dim, EXAKT in Datensatz-Reihenfolge (siehe Docstring/DEX3_ARM_HAND_JOINT_NAMES_ORDERED
        # oben) -- entspricht 1:1 dem "observation.state"-Feld des UniFolm-VLA-Trainingsdatensatzes.
        arm_hand_joint_pos_ordered = ObsTerm(
            func=get_ordered_joint_pos,
            params={"joint_names_ordered": DEX3_ARM_HAND_JOINT_NAMES_ORDERED},
        )

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

        # (hand_joint_state entfaellt hier -- bereits vollstaendig in
        # arm_hand_joint_pos_ordered enthalten, in der korrekten Datensatz-Reihenfolge)
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
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)

    cube_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("cube_red")},
    )

    success = DoneTerm(func=cubes_stacked_trihand)


@configclass
class FixedBaseUpperBodyIKG1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 Dex3 cube-stacking eval environment with direct
    joint-space control (no Pink IK), used to evaluate the pretrained UniFolm-VLA policy."""

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

    # Kein XR-Anker/teleop_devices: ohne Pink-IK gibt es keinen eingebauten Retargeter,
    # der Hand-Tracking-Posen auf rohe Gelenkwinkel abbildet (siehe Docstring oben). Diese
    # Env ist fuer skriptgesteuerte UniFolm-VLA-Evaluation gedacht, nicht Teleoperation.

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 200  # 200 Hz
        self.sim.render_interval = 2
