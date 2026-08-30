# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


'''Cube-Stacking Eval-Umgebung mit G1 + Parallelgreifer (Dex1), fuer den Test des
vortrainierten UniVLA (fixed base, direkte Gelenkwinkel-Steuerung).

Zweite Variante von fixed_base_upper_body_ik_g1_env_cfg_CubeStack.py (dort: G1 mit
Dex3-Dreifingerhand + Pink-IK, fuer GR00T). Diese Datei portiert dieselbe
Cube-Stacking-Szene auf den Unitree-G1-Parallelgreifer ("dex1", 2 Gelenke pro Hand), wie
er bereits in unitree_sim_isaaclab/tasks/g1_tasks/stack_rgyblock_g1_29dof_dex1/ fuer
denselben Task verwendet wird -- siehe Anleitung/Masterprojekt/cube_stack_eval_env.md fuer
den Vergleich beider Varianten (Dex3/GR00T vs. Parallelgreifer/UniVLA) sowie die komplette
Debug-Historie dieser Datei.

Aenderungen gegenueber der Dex3-Variante (siehe dortigen Docstring fuer das dort
verwendete Pink-IK-basierte Beobachtungs-/Aktionsformat):

- Robot-Asset: G1_29DOF_DEX1_CFG (unten definiert) statt isaaclab_assets.G1_29DOF_CFG --
  USD ist der 1:1 aus unitree_sim_isaaclab uebernommene Parallelgreifer-Roboter
  (assets/robots/g1-29dof-dex1-base-fix-usd/g1_29dof_with_dex1_base_fix1.usd dort),
  nach source/isaaclab_assets/custom_assets/G1_Dex1_ParallelGripper_usd/ kopiert, damit
  der Pfad auch im IsaacLab-Docker-Container (kein Mount von unitree_sim_isaaclab)
  auflösbar ist.
- Action: direkte Gelenkwinkel-Steuerung (JointPositionActionCfg) statt Pink-IK -- siehe
  Begruendung beim UPPER_BODY_JOINT_NAMES-Block weiter unten. Pink-IK wurde verworfen,
  weil es eine passende Kinematik-URDF fuer den Parallelgreifer vorausgesetzt haette, die
  nirgends existiert; unitree_sim_isaaclab loest dasselbe Problem fuer diesen Roboter auf
  dieselbe Weise (JointPositionActionCfg statt Pink-IK).
- Kein Hand-Tracking-/Motion-Controller-Teleop mehr verdrahtet (siehe EnvCfg-Kommentar
  unten): der bisherige Retargeter setzte eine 6D-Handgelenkpose-Aktion voraus, die es
  ohne Pink-IK nicht mehr gibt.

Kamera-Mount-Links (WICHTIG, per Teleop-Testlauf gefunden -- Dex1-USD hat eine andere
Link-Topologie als die Dex3-USD, urspruenglich 1:1 von dort kopiert und dadurch zuerst mit
"Unable to find source prim path: '.../Robot/torso_link/d435_link'" fehlgeschlagen):
- Kopfkamera: "Robot/d435_link/..." (DIREKT unter Robot, kein "torso_link" dazwischen wie
  bei Dex3). Referenz: unitree_sim_isaaclab/tasks/common_config/camera_configs.py
  ::g1_front_camera / CameraBaseCfg.get_camera_config (Default-Mount-Punkt + rot-Offset
  (0.5,-0.5,0.5,-0.5) fuer diese USD).
- Handgelenkkameras: "Robot/left_hand_base_link/..." bzw. "Robot/right_hand_base_link/..."
  (Parallelgreifer-spezifische Links, NICHT die Dex3-Wrist-Links) mit Offsets aus
  camera_configs.py::left/right_gripper_wrist_camera.
- Der stereoskopische Links/Rechts-Split der Kopfkamera (±0.03 m in X) ist eine eigene
  Ergaenzung dieser Datei -- unitree's g1_front_camera ist eine einzelne Kamera, keine
  Stereo-Konfiguration; visuell in Isaac Sim gegenpruefen, ob der Split sinnvoll sitzt.

Debug-Historie (chronologisch, alle per Teleop-Testlauf gefunden):
1. Kamera-Pfad-Fehler ("torso_link/d435_link" existiert nicht) -- behoben, siehe oben.
2. Pink-IK-Fehler ("'left_hand_index_0_joint' is not in list", aus
   isaaclab/controllers/pink_ik/pink_ik.py::_setup_joint_ordering_mappings) -- Pink-IK
   baut eine vollstaendige Namens-Abbildung zwischen ALLEN Gelenken der Kinematik-URDF und
   der tatsaechlichen Roboter-USD auf, nicht nur den von Pink geloesten Arm-/Taille-
   Gelenken. Die URDF wurde fuer die Dex3-Hand generiert (14 Fingergelenke); die Dex1-USD
   hat andere Gelenknamen -- daher auf JointPositionActionCfg umgestellt (siehe oben).

Noch zu verifizieren (siehe Anleitung/Masterprojekt/cube_stack_eval_env.md):
- Der Wechsel auf JointPositionActionCfg ist noch NICHT per Teleop/Env-Erstellung
  gegengetestet.
- `left_wrist_yaw_link`/`right_wrist_yaw_link` (fuer die eef-Observations) sowie
  `left/right_hand_base_link` (fuer die Handgelenkkameras) werden als auf der Dex1-USD
  vorhanden angenommen (gleiche Arm-Kette wie Dex3), aber noch nicht einzeln bestaetigt.
'''

import os
import sys

import numpy as np
import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
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

##
# Robot: G1 29-DOF mit Parallelgreifer ("dex1"), fixed base -- 1:1 portiert aus
# unitree_sim_isaaclab/robots/unitree.py (G129_CFG_WITH_DEX1_BASE_FIX), USD-Pfad auf die
# nach isaaclab_assets/custom_assets kopierte Kopie umgebogen.
##

_G1_DEX1_USD_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../../isaaclab_assets/custom_assets/G1_Dex1_ParallelGripper_usd/g1_29dof_with_dex1_base_fix1.usd",
    )
)

G1_29DOF_DEX1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_G1_DEX1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.05,
            "left_knee_joint": 0.2,
            "left_ankle_pitch_joint": -0.15,
            "left_ankle_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.05,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": -0.15,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            # Parallelgreifer-Gelenke (2 pro Hand, statt 7 Dex3-Fingergelenken pro Hand)
            "left_hand_Joint1_1": 0.0,
            "left_hand_Joint2_1": 0.0,
            "right_hand_Joint1_1": 0.0,
            "right_hand_Joint2_1": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
            armature=None,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            effort_limit=1000.0,
            velocity_limit=0.0,
            stiffness={"waist_yaw_joint": 10000.0, "waist_roll_joint": 10000.0, "waist_pitch_joint": 10000.0},
            damping={"waist_yaw_joint": 10000.0, "waist_roll_joint": 10000.0, "waist_pitch_joint": 10000.0},
            armature=None,
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=None,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=None,
            damping=None,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
            effort_limit=None,
            velocity_limit=None,
            stiffness={".*_shoulder_.*_joint": 25.0, ".*_elbow_joint": 50.0, ".*_wrist_.*_joint": 40.0},
            damping={".*_shoulder_.*_joint": 2.0, ".*_elbow_joint": 2.0, ".*_wrist_.*_joint": 2.0},
            armature=None,
        ),
        # Stiffness/Damping/Friction bereits fuer Greif-Aufgaben getunt (uebernommen aus
        # unitree_sim_isaaclab, dort bereits gegen den Parallelgreifer kalibriert).
        "hands": ImplicitActuatorCfg(
            joint_names_expr=["left_hand_Joint1_1", "left_hand_Joint2_1", "right_hand_Joint1_1", "right_hand_Joint2_1"],
            effort_limit=None,
            velocity_limit=None,
            stiffness=800.0,
            damping=3.0,
            friction=200.0,
            armature=None,
        ),
    },
    # Erforderlich (kein Default in ArticulationCfg) -- fehlte in der 1:1 aus
    # unitree_sim_isaaclab uebernommenen Definition, dort wird prim_path offenbar an
    # anderer Stelle (Preset-Factory) ergaenzt. Ohne dieses Feld schlaegt
    # cfg.validate() beim Env-Erstellen mit "Missing values ... scene.robot.prim_path"
    # fehl (siehe isaaclab_assets G1_29DOF_CFG, das dasselbe Feld genauso setzt).
    prim_path="/World/envs/env_.*/Robot",
)

##
# Action: direkte Gelenkwinkel-Steuerung (Joint Position) statt Pink-IK.
#
# Grund (siehe Docstring oben, Abschnitt "Pink-IK verworfen"): Pink-IK baut intern eine
# vollstaendige Namens-Abbildung zwischen der Kinematik-URDF und den tatsaechlichen
# Gelenknamen der Roboter-USD auf (isaaclab/controllers/pink_ik/pink_ik.py,
# _setup_joint_ordering_mappings: "isaac_lab_joint_names.index(pink_joint) for pink_joint
# in pink_joint_names") -- und verlangt dabei Namensgleichheit fuer ALLE URDF-Gelenke,
# nicht nur die von Pink tatsaechlich geloesten Arm-/Taille-Gelenke. Die vorhandene
# Kinematik-URDF (g1_29dof_with_hand_only_kinematics.urdf) wurde fuer die Dex3-Hand
# generiert und enthaelt deren 14 Fingergelenke (z.B. "left_hand_index_0_joint"); die
# Dex1-USD hat aber nur "left_hand_Joint1_1"/"Joint2_1" -- das fuehrt beim Scene-Aufbau zu
# "'left_hand_index_0_joint' is not in list". Eine passende Dex1-Kinematik-URDF existiert
# nirgends (weder in IsaacLab noch in unitree_sim_isaaclab, das fuer denselben Roboter
# ebenfalls nur Joint-Position-Steuerung verwendet, siehe
# stack_rgyblock_g1_29dof_dex1_joint_env_cfg.py::ActionsCfg). Statt eine eigene URDF zu
# erzeugen (Tooling-Aufwand, hier nicht ausfuehrbar), wird -- wie im unitree-Referenz-Task
# -- direkte Gelenkwinkel-Steuerung verwendet. Das aendert das Aktionsformat gegenueber der
# Dex3-CubeStack-Env (dort: 28-dim 6D-Handgelenkpose+Hand ueber Pink-IK) auf 21-dim rohe
# Gelenkwinkel (7 Arm-/Handgelenke pro Seite + 3 Taille + 2 Greifer-Gelenke pro Seite).
# Fuer die UniVLA-Evaluation muss das Modell also im Joint-Space auswerten, nicht im
# Pose-Space wie GR00T bei der Dex3-Env.
##

UPPER_BODY_JOINT_NAMES: list[str] = [
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_roll_joint",
    ".*_wrist_yaw_joint",
    "waist_.*_joint",
    "left_hand_Joint1_1",
    "left_hand_Joint2_1",
    "right_hand_Joint1_1",
    "right_hand_Joint2_1",
]

##
# Konstanten - Wuerfel (identisch zur Dex3-Cube-Stack-Variante, siehe dortigen Kommentar)
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
# Details/Herkunft -- unveraendert uebernommen)
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
# Reset-/Erfolgs-Funktionen (identisch zur Dex3-Cube-Stack-Variante)
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


def cubes_stacked_parallel_gripper(
    env,
    cube_bottom_cfg: SceneEntityCfg = SceneEntityCfg("cube_red"),
    cube_middle_cfg: SceneEntityCfg = SceneEntityCfg("cube_yellow"),
    cube_top_cfg: SceneEntityCfg = SceneEntityCfg("cube_green"),
    xy_threshold: float = 0.03,
    height_threshold: float = 0.01,
    height_diff: float = CUBE_SIZE,
) -> torch.Tensor:
    """Bool-Tensor (N,): rot-gelb UND gelb-gruen jeweils xy-ausgerichtet und um genau
    eine Wuerfelhoehe versetzt gestapelt (kein Greifer-Offen-Check, siehe Dex3-Variante
    fuer die Begruendung -- hier zusaetzlich der Einfachheit halber ebenfalls
    weggelassen, obwohl ein Parallelgreifer im Gegensatz zur Trihand einen einfachen
    "offen"-Zustand haette; kann spaeter ueber SceneEntityCfg("robot") + Joint1_1/
    Joint2_1-Position ergaenzt werden, falls noetig)."""
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
    """Scene configuration fuer die Cube-Stacking-Eval-Umgebung mit G1 + Parallelgreifer."""

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

    # G1 mit Parallelgreifer (dex1) -- fixed base
    robot: ArticulationCfg = G1_29DOF_DEX1_CFG

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Kopf-Stereokameras -- Mount-Link korrigiert: die Dex1-USD hat "d435_link" DIREKT
    # unter "Robot" (kein "torso_link" dazwischen wie bei der Dex3-USD), siehe
    # unitree_sim_isaaclab/tasks/common_config/camera_configs.py::g1_front_camera
    # (prim_path=".../Robot/d435_link/front_cam"). rot-Offset (0.5,-0.5,0.5,-0.5) ist
    # dort ebenfalls der Default fuer diesen Mount-Punkt, siehe CameraBaseCfg.get_camera_config.
    cam_left_high = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/d435_link/cam_left_high",
        update_period=1.0 / 30.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12, clipping_range=(0.1, 1.0e5)),
        offset=CameraCfg.OffsetCfg(pos=(-0.03, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    cam_right_high = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/d435_link/cam_right_high",
        update_period=1.0 / 30.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12, clipping_range=(0.1, 1.0e5)),
        offset=CameraCfg.OffsetCfg(pos=(0.03, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    # Handgelenkkameras -- Mount-Link + Offsets korrigiert auf
    # left/right_hand_base_link (Parallelgreifer-spezifisch, siehe
    # camera_configs.py::left/right_gripper_wrist_camera) statt der Dex3-Wrist-Links.
    cam_left_wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_base_link/cam_left_wrist",
        update_period=1.0 / 30.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.1, 1.0e5)),
        offset=CameraCfg.OffsetCfg(
            pos=(0.02541028, 0.045, 0.135),
            rot=(-0.34202, 0.93969, 0.0, 0.0),
            convention="ros",
        ),
    )

    cam_right_wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_base_link/cam_right_wrist",
        update_period=1.0 / 30.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.1, 1.0e5)),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.02541028, 0.045, 0.135),
            rot=(-0.34202, 0.93969, 0.0, 0.0),
            convention="ros",
        ),
    )

    def __post_init__(self):
        """Post initialization."""
        self.robot.spawn.articulation_props.fix_root_link = True


@configclass
class ActionsCfg:
    """Action specifications for the MDP -- direkte Gelenkwinkel-Steuerung (kein Pink-IK,
    siehe Begruendung beim UPPER_BODY_JOINT_NAMES-Block oben)."""

    upper_body_joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=UPPER_BODY_JOINT_NAMES,
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
            "asset_cfg": SceneEntityCfg("robot", body_names=r"(left|right)_hand_.*"),
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

        # 2 Gelenke pro Hand (Joint1_1/Joint2_1) statt 7 Dex3-Fingergelenken pro Hand.
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
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)

    cube_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("cube_red")},
    )

    success = DoneTerm(func=cubes_stacked_parallel_gripper)


@configclass
class FixedBaseUpperBodyIKG1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 fixed base upper body IK cube-stacking eval environment
    with parallel gripper (dex1), used to evaluate the pretrained UniVLA policy."""

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

    # Kein XR-Anker/teleop_devices mehr noetig: ohne Pink-IK gibt es keinen eingebauten
    # Retargeter, der Hand-Tracking/Motion-Controller-Eingaben direkt in rohe
    # Gelenkwinkel-Ziele uebersetzt (das war zuvor die Aufgabe der Pink-IK-Action). Diese
    # Env ist fuer skriptgesteuerte Policy-Evaluation (UniVLA) gedacht, nicht fuer
    # Hand-Tracking-Teleoperation. Fuer eine reine Struktur-/Szenen-Validierung ohne echte
    # Steuerung eignet sich `--teleop_device keyboard` weiterhin (nutzt dann den generischen
    # Se3Keyboard-Fallback, auch wenn die Achsen nicht sinnvoll auf 21 Gelenkwinkel mappen).

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 200  # 200 Hz
        self.sim.render_interval = 2
