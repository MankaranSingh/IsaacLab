# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Mujoco Humanoid AMP robot."""

from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

STIFFNESS = 100.0
DAMPING = 2.0

HUMANOID_AMP_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/mankaran/orbit/source/extensions/omni.isaac.lab_assets/omni/isaac/lab_assets/humanoid_amp/amp_humanoid.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={".*": 0.0},
    ),
    actuators = {
    "body": ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        stiffness={
            "abdomen_.*": STIFFNESS,            # Matches abdomen_x, abdomen_y, abdomen_z
            "neck_.*": STIFFNESS,
            ".*_shoulder_.*": STIFFNESS,        # Matches left_shoulder_x, left_shoulder_y, etc.
            ".*_elbow*": STIFFNESS,               # Matches left_elbow, right_elbow
            ".*_hip_.*": STIFFNESS,          # Matches left_hip_x, left_hip_y, right_hip_x, etc.
            ".*_knee": STIFFNESS,                # Matches left_knee, right_knee
            ".*_ankle_.*": STIFFNESS,            # Matches left_ankle_x, left_ankle_y, etc.
            },
        damping={
            "abdomen_.*": DAMPING,             # Matches abdomen_x, abdomen_y, abdomen_z
            "neck_.*": DAMPING,
            ".*_shoulder_.*": DAMPING,         # Matches left_shoulder_x, left_shoulder_y, etc.
            ".*_elbow": DAMPING,               # Matches left_elbow, right_elbow
            ".*_hip_.*": DAMPING,           # Matches left_hip_x, left_hip_y, right_hip_x, etc.
            ".*_knee": DAMPING,                # Matches left_knee, right_knee
            ".*_ankle_.*": DAMPING,            # Matches left_ankle_x, left_ankle_y, etc.
            },
        ),
    },
)
"""Configuration for the Mujoco AMP Humanoid robot."""
