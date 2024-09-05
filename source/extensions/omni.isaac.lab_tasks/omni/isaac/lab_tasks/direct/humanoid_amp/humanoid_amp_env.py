# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab_assets import HUMANOID_AMP_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.direct.locomotion.locomotion_amp_env import LocomotionAMPEnv


@configclass
class HumanoidAMPEnvCfg(DirectRLEnvCfg):
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = HUMANOID_AMP_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list = [
        67.5000,  # abdomen_x
        67.5000,  # abdomen_y
        67.5000,  # abdomen_z

        135.0000,  # right_hip_x
        135.0000,  # right_hip_y
        135.0000,  # right_hip_z

        135.0000,  # left_hip_x
        135.0000,  # left_hip_y
        135.0000,  # left_hip_z

        22.5,  # neck_x
        22.5,  # neck_y
        22.5,  # neck_z

        67.5000,  # left_shoulder_x
        67.5000,  # left_shoulder_y
        67.5000,  # left_shoulder_z
        
        67.5000,  # right_shoulder_x
        67.5000,  # right_shoulder_y
        67.5000,  # right_shoulder_z

        90.0000,  # right_knee
        90.0000,  # left_knee

        45.0000,  # left_elbow
        45.0000,  # right_elbow

        22.5,  # right_ankle_x
        22.5,  # right_ankle_y
        22.5,  # right_ankle_z
        
        22.5,  # left_ankle_x
        22.5,  # left_ankle_y
        22.5,  # left_ankle_z
    ]

    joint_gears: list = [50.0]*28

    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    num_actions = 28
    num_observations = 96
    num_states = 0

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.0
    actions_cost_scale: float = 0.0
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.01

    death_cost: float = -1.0
    termination_height: float = 0.7

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01


class HumanoidAMPEnv(LocomotionAMPEnv):
    cfg: HumanoidAMPEnvCfg

    def __init__(self, cfg: HumanoidAMPEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
