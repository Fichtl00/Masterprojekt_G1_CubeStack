# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Articulation-Konfiguration für G1 + Dex3-Hand -- Task-Gruppe ``1_cube_stack``.

Portiert aus https://github.com/Docboter/projektarbeit_humanoider_roboter
(Branch training-luca-IKR-IS6.0, Simulation/g1_dex3_sim/g1_dex3_cfg.py) -- auf Wunsch NUR
die Articulation-/Actuator-Config, nicht die dortige RL-Env-Logik (Reward/Success/DR aus
g1_dex3_blockstack_env.py). Siehe fixed_base_upper_body_ik_g1_env_cfg_1_cube_stack.py für
die Env drumherum (Pink-IK + OpenXR-Teleop, wie die bestehende ...CubeStack.py-Env) und die
Masterprojekt-Doku für die vollständige Liste der Unterschiede zum Original.

ASSET-PFAD GEÄNDERT ggü. Original: das Original zeigt auf ein eigens kombiniertes,
schwarzhändig eingefärbtes USD (``/data/assets/g1_dex3_blackhands.usd``, per eigener
URDF->USD-Konvertierung + Recolor-Skript erzeugt). Dieses Asset existiert hier nicht --
verwendet wird stattdessen dasselbe Nucleus-Asset wie ``G1_29DOF_CFG``
(isaaclab_assets.robots.unitree), das den Dex3-Hand-Mesh bereits enthält (nur eben in Weiß
statt Schwarz -- kein Hand-Recolor Teil dieses Ports).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# ---------------------------------------------------------------------------
# Joint-Namen (unverändert aus dem Original -- deckt sich mit unserem eigenen
# g1_dex3_blockstacking-Datensatz-Schema, siehe
# Anleitung/Masterprojekt/unifolm_vla_scratch_finetuning_g1_dex3_blockstacking.md)
# ---------------------------------------------------------------------------

LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# Achtung: rechts Index VOR Middle (≠ links) -- Quelle: g1_dex3_config.py + dataset info.json.
LEFT_DEX3_JOINTS = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
]
RIGHT_DEX3_JOINTS = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]
ALL_JOINTS_ORDERED = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + LEFT_DEX3_JOINTS + RIGHT_DEX3_JOINTS

# Startpose = observation.state aus Frame 0 / Episode 0 von unitreerobotics/G1_Dex3_BlockStacking_Dataset,
# unverändert aus dem Original übernommen. Reihenfolge = ALL_JOINTS_ORDERED.
DATASET_INIT_STATE = [
    -0.05662,  0.31445,  0.12487, -0.28736,  0.20897,  0.13095, -0.09767,  # left_arm
    -0.40259, -0.34180, -0.01584,  0.17989,  0.00290, -0.09826,  0.29865,  # right_arm
    -0.59676,  1.01160,  0.05485, -0.169,   -0.01227, -0.163,   -0.01201,  # left_dex3
    -0.70991, -1.01463, -0.21031,  0.171,    0.01391,  0.142,    0.03354,  # right_dex3
]
# middle_0/index_0 beider Hände sind im Original mit invertierter USD-Achse dokumentiert
# (Original-Kommentar in g1_dex3_cfg.py). Ob dieser Flip auch auf UNSEREM Nucleus-Asset
# nötig ist, ist NICHT verifiziert (das Original ist auf seinem eigenen USD kalibriert) --
# vor produktiver Nutzung gegen ein Teleop-Live-Bild prüfen. Hier bewusst NICHT automatisch
# angewendet (anders als im Original), da diese Env nur die Config+Kameras übernimmt.
SIGN_FLIP_JOINT_NAMES = [
    "left_hand_middle_0_joint", "left_hand_index_0_joint",
    "right_hand_index_0_joint", "right_hand_middle_0_joint",
]

# ---------------------------------------------------------------------------
# Articulation-Konfiguration
# ---------------------------------------------------------------------------

G1_DEX3_CFG_1_CUBE_STACK = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # GEÄNDERT ggü. Original (s. Docstring): Nucleus-Asset statt eigenem schwarzhändigem USD.
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos=dict(zip(ALL_JOINTS_ORDERED, DATASET_INIT_STATE)),
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Arme: erst auf G1_29DOF_CFG-Werte angehoben (stiffness=3000, velocity_limit=100),
        # laut Nutzer-Feedback danach zu schnell/ruckartig -- auf ca. 1/4 davon reduziert.
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=LEFT_ARM_JOINTS,
            effort_limit=300.0, velocity_limit=25.0, stiffness=750.0, damping=10.0, armature=0.01,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=RIGHT_ARM_JOINTS,
            effort_limit=300.0, velocity_limit=25.0, stiffness=750.0, damping=10.0, armature=0.01,
        ),
        # Hände: stiffness/damping unveraendert (empirisch fuer Dex3-Fingerkraft gegen ~50g-
        # Wuerfel kalibriert, siehe Original-Kommentar). velocity_limit/effort_limit waren mit
        # 3.0/20.0 deutlich niedriger als G1_29DOF_CFG's Hand-Gains (velocity_limit=100,
        # effort_limit=300) -- das bremste das Schliessen der Finger beim Teleop unnoetig aus.
        # Angehoben, Stiffness/Damping fuer den Griff aber unangetastet gelassen.
        "left_hand": ImplicitActuatorCfg(
            joint_names_expr=LEFT_DEX3_JOINTS,
            effort_limit=60.0, velocity_limit=20.0, stiffness=60.0, damping=4.0, armature=0.001,
        ),
        "right_hand": ImplicitActuatorCfg(
            joint_names_expr=RIGHT_DEX3_JOINTS,
            effort_limit=60.0, velocity_limit=20.0, stiffness=60.0, damping=4.0, armature=0.001,
        ),
    },
)
