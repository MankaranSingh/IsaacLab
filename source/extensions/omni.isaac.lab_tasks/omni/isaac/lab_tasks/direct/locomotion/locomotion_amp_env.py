# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import torch
import numpy as np
import gymnasium as gym

import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.core.utils.torch.torch_jit_utils import quat_mul, calc_heading_quat_inv, \
                                                        exp_map_to_quat, quat_to_tan_norm, my_quat_rotate, calc_heading_quat_inv
from omni.isaac.lab.utils.poselib.motion_lib_lab import MotionLib

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg


NUM_DOFS = 28

BODY_NAMES_ISAAC_GYM = ['pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 'right_hand', 'left_upper_arm', 'left_lower_arm', 'left_hand', 
                        'right_thigh', 'right_shin', 'right_foot', 'left_thigh', 'left_shin', 'left_foot']

BODY_NAMES_LAB = ['pelvis', 'torso', 'right_thigh', 'left_thigh', 'head', 'left_upper_arm', 'right_upper_arm', 'right_shin', 'left_shin',
               'left_lower_arm', 'right_lower_arm', 'right_foot', 'left_foot', 'left_hand', 'right_hand']

DOF_NAMES_LAB = ['abdomen_x', 'abdomen_y', 'abdomen_z', 'right_hip_x', 'right_hip_y', 'right_hip_z', 'left_hip_x', 'left_hip_y', 'left_hip_z', 'neck_x', 'neck_y', 'neck_z',
                'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z', 'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z', 'right_knee', 'left_knee', 
                'left_elbow', 'right_elbow', 'right_ankle_x', 'right_ankle_y', 'right_ankle_z', 'left_ankle_x', 'left_ankle_y', 'left_ankle_z']

DOF_BODY_IDS_LAB = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
DOF_OFFSETS_LAB = [0, 3, 6, 9, 12, 15, 18, 19, 20, 21, 22, 25, 28]

KEY_BODY_NAMES = ["right_hand", "left_hand", "right_foot", "left_foot"]

KEY_BODY_IDS = np.array([BODY_NAMES_LAB.index(body) for body in KEY_BODY_NAMES])
LAB_IDS = np.array([BODY_NAMES_LAB.index(body) for body in BODY_NAMES_ISAAC_GYM])

NUM_AMP_OBS_PER_STEP = 13 + 52 + 28 + 12 # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class LocomotionAMPEnv(DirectRLEnv):
    cfg: DirectRLEnvCfg

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)
        self._joint_dof_idx, _ = self.robot.find_joints(".*", preserve_order=True)

        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.targets += self.scene.env_origins
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        # AMP specific
        motion_file = "amp_humanoid_run.npy"
        motion_file_path = os.path.join("/home/mankaran/orbit/source/extensions/omni.isaac.lab_assets/omni/isaac/lab_assets/humanoid_amp/motions", motion_file)
        self._motion_lib = MotionLib(motion_file=motion_file_path, 
                                     num_dofs=NUM_DOFS,
                                     device=self.device)

        self._build_pd_action_offset_scale(self.robot.data.default_joint_limits[0])
        self._num_amp_obs_steps = 2
        assert(self._num_amp_obs_steps >= 2)
        self._amp_batch_size = 512
        self._amp_obs_demo_buf = torch.zeros((self._amp_batch_size, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP), device=self.device, dtype=torch.float)
        self.num_amp_obs = self._num_amp_obs_steps * NUM_AMP_OBS_PER_STEP
        self.task_reward = torch.ones(self.num_envs, device=self.device)
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=[self.num_amp_obs, ])

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add articultion to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos
        #self._processed_actions = self._pd_action_offset + self._pd_action_scale * self.actions

    def _apply_action(self):
        # pd_targets = self._pd_action_offset + self._pd_action_scale * self.actions
        self.robot.set_joint_position_target(self._processed_actions, joint_ids=self._joint_dof_idx)
    
    def fetch_amp_obs_demo(self, num_samples):
        dt = self.sim.cfg.dt
        motion_ids = self._motion_lib.sample_motions(num_samples)
            
        truncate_time = dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time
        
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps])
        motion_times = np.expand_dims(motion_times0, axis=-1)
        time_steps = -dt * np.arange(0, self._num_amp_obs_steps)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)

        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.num_amp_obs)

        return amp_obs_demo_flat
    
    def _build_pd_action_offset_scale(self, joint_limits):
        num_joints = len(DOF_OFFSETS_LAB) - 1

        lim_low = joint_limits[:, 0].cpu().numpy()
        lim_high = joint_limits[:, 1].cpu().numpy()

        for j in range(num_joints):
            dof_offset = DOF_OFFSETS_LAB[j]
            dof_size = DOF_OFFSETS_LAB[j + 1] - DOF_OFFSETS_LAB[j]

            if (dof_size == 3):
                lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

            elif (dof_size == 1):
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)
                
                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] =  curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = torch.tensor(self._pd_action_offset, device=self.device)
        self._pd_action_scale = torch.tensor(self._pd_action_scale, device=self.device)

        return

    def _compute_intermediate_values(self):
        self.torso_position, self.torso_rotation = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        self.velocity, self.ang_velocity = self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w
        self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel

        # print("### root", self.torso_position.shape, self.torso_rotation.shape, self.velocity.shape, self.ang_velocity.shape)
        # print("### dofs", self.dof_pos.shape, self.dof_vel.shape, self.robot.data.body_names)
        # print("### body", self.robot.data.body_names, self.robot.data.body_pos_w)

        (
            self.up_proj,
            self.heading_proj,
            self.up_vec,
            self.heading_vec,
            self.vel_loc,
            self.angvel_loc,
            self.roll,
            self.pitch,
            self.yaw,
            self.angle_to_target,
            self.dof_pos_scaled,
            self.prev_potentials,
            self.potentials,
        ) = compute_intermediate_values(
            self.targets,
            self.torso_position,
            self.torso_rotation,
            self.velocity,
            self.ang_velocity,
            self.dof_pos,
            self.robot.data.soft_joint_pos_limits[0, :, 0],
            self.robot.data.soft_joint_pos_limits[0, :, 1],
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.cfg.sim.dt,
        )  
    
    def get_observations(self):
        return self._get_observations()

    def _get_observations(self) -> dict:
        amp_obs = build_amp_observations(self.robot.data.root_pos_w, self.robot.data.root_quat_w, self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w, \
                                          self.dof_pos, self.dof_vel, self.robot.data.body_pos_w[:, KEY_BODY_IDS])
        observations = {"policy": amp_obs}

        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.num_amp_obs)
        self.extras["amp_obs"] = amp_obs_flat

        return observations

    def _get_rewards(self) -> torch.Tensor:
        return self.task_reward
    
    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return

    def _compute_amp_observations(self, env_ids=None):
        if (env_ids is None):
            self._curr_amp_obs_buf[:] = build_amp_observations(self.robot.data.root_pos_w, self.robot.data.root_quat_w, self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w, \
                                          self.dof_pos, self.dof_vel, self.robot.data.body_pos_w[:, KEY_BODY_IDS])
        else:            
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self.robot.data.root_pos_w[env_ids], self.robot.data.root_quat_w[env_ids], self.robot.data.root_lin_vel_w[env_ids], self.robot.data.root_ang_vel_w[env_ids], \
                                          self.dof_pos[env_ids], self.dof_vel[env_ids], self.robot.data.body_pos_w[:, KEY_BODY_IDS][env_ids])
        return

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self.torso_position[:, 2] < self.cfg.termination_height
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        motion_times = self._motion_lib.sample_time(motion_ids)
        #motion_times = np.zeros(num_envs)

        # set half elements to start of the clip
        indices = np.random.choice(num_envs, num_envs//2, replace=False)    
        motion_times[indices] = 0

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state(motion_ids, motion_times)

        root_rot = math_utils.convert_quat(root_rot, to="wxyz")

        joint_pos = dof_pos
        joint_vel = dof_vel
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] = root_pos
        default_root_state[:, 3:7] = root_rot
        default_root_state[:, 7:10] = root_vel
        default_root_state[:, 10:13] = root_ang_vel
        
        default_root_state[:, :3] += self.scene.env_origins[env_ids]


        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

        self._compute_intermediate_values()

        self._compute_amp_observations(env_ids)
        self._init_amp_obs_default(env_ids)

        self.extras["terminate"] = self.reset_terminated

# @torch.jit.script
# def compute_rewards(
#     actions: torch.Tensor,
#     reset_terminated: torch.Tensor,
#     up_weight: float,
#     heading_weight: float,
#     heading_proj: torch.Tensor,
#     up_proj: torch.Tensor,
#     dof_vel: torch.Tensor,
#     dof_pos_scaled: torch.Tensor,
#     potentials: torch.Tensor,
#     prev_potentials: torch.Tensor,
#     actions_cost_scale: float,
#     energy_cost_scale: float,
#     dof_vel_scale: float,
#     death_cost: float,
#     alive_reward_scale: float,
#     motor_effort_ratio: torch.Tensor,
# ):
#     heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
#     heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, heading_weight * heading_proj / 0.8)

#     # aligning up axis of robot and environment
#     up_reward = torch.zeros_like(heading_reward)
#     up_reward = torch.where(up_proj > 0.93, up_reward + up_weight, up_reward)

#     # energy penalty for movement
#     actions_cost = torch.sum(actions**2, dim=-1)
#     electricity_cost = torch.sum(
#         torch.abs(actions * dof_vel * dof_vel_scale) * motor_effort_ratio.unsqueeze(0),
#         dim=-1,
#     )

#     # dof at limit cost
#     dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98, dim=-1)

#     # reward for duration of staying alive
#     alive_reward = torch.ones_like(potentials) * alive_reward_scale
#     progress_reward = potentials - prev_potentials

#     total_reward = (
#         progress_reward
#         + alive_reward
#         + up_reward
#         + heading_reward
#         - actions_cost_scale * actions_cost
#         - energy_cost_scale * electricity_cost
#         - dof_at_limit_cost
#     )
#     # adjust reward for fallen agents
#     total_reward = torch.where(reset_terminated, torch.ones_like(total_reward) * death_cost, total_reward)
#     return total_reward

@torch.jit.script
def dof_to_obs(pose):
    # type: (Tensor) -> Tensor
    dof_obs_size = 52
    dof_offsets = [0, 3, 6, 9, 12, 15, 18, 19, 20, 21, 22, 25, 28]
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = exp_map_to_quat(joint_pose)
            joint_dof_obs = quat_to_tan_norm(joint_pose_q)
            dof_obs_size = 6
        else:
            joint_dof_obs = joint_pose
            dof_obs_size = 1

        dof_obs[:, dof_obs_offset:(dof_obs_offset + dof_obs_size)] = joint_dof_obs
        dof_obs_offset += dof_obs_size

    return dof_obs

@torch.jit.script
def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, local_root_obs=True):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos)

    obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs

@torch.jit.script
def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
):
    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
    )
