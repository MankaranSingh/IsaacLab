# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
from datetime import datetime

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.amp import AMP, AMP_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.utils import set_seed
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, shared_model
from skrl.utils.model_instantiators.torch import Policy, Value, Discriminator
from skrl.resources.preprocessors.torch import RunningStandardScaler

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlSequentialLogTrainer, SkrlVecEnvWrapper, process_skrl_cfg


def main():
    """Train with skrl agent."""
    # read the seed from command line
    args_cli_seed = args_cli.seed

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if experiment_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{experiment_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    experiment_cfg["agent"]["experiment"]["directory"] = log_root_path
    experiment_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # max iterations for training
    if args_cli.max_iterations:
        experiment_cfg["trainer"]["timesteps"] = args_cli.max_iterations * experiment_cfg["agent"]["rollouts"]

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), experiment_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), experiment_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env)  # same as: `wrap_env(env, wrapper="isaac-orbit")`

    # set seed for the experiment (override from command line)
    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

    # instantiate models using skrl model instantiator utility
    # https://skrl.readthedocs.io/en/latest/api/utils/model_instantiators.html
    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, env.device)
    models["value"] = Value(env.observation_space, env.action_space, env.device)
    models["discriminator"] = Discriminator(env.amp_observation_space, env.action_space, env.device)
    

    # instantiate a RandomMemory as rollout buffer (any memory can be used for this)
    # https://skrl.readthedocs.io/en/latest/api/memories/random.html
    memory_size = experiment_cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
    memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=env.device)

    # configure and instantiate AMP agent
    # https://skrl.readthedocs.io/en/latest/api/agents/amp.html
    agent_cfg = AMP_DEFAULT_CONFIG.copy()
    agent_cfg["rollouts"] = 16  # memory_size
    agent_cfg["learning_epochs"] = 6
    agent_cfg["mini_batches"] = 4  # 16 * 4096 / 32768
    agent_cfg["discount_factor"] = 0.99
    agent_cfg["lambda"] = 0.95
    agent_cfg["learning_rate"] = 5e-5
    agent_cfg["random_timesteps"] = 0
    agent_cfg["learning_starts"] = 0
    agent_cfg["grad_norm_clip"] = 0
    agent_cfg["ratio_clip"] = 0.2
    agent_cfg["value_clip"] = 0.2
    agent_cfg["clip_predicted_values"] = False
    agent_cfg["entropy_loss_scale"] = 0.0
    agent_cfg["value_loss_scale"] = 2.5
    agent_cfg["discriminator_loss_scale"] = 5.0
    agent_cfg["amp_batch_size"] = 512
    agent_cfg["task_reward_weight"] = 0.0
    agent_cfg["style_reward_weight"] = 1.0
    agent_cfg["discriminator_batch_size"] = 4096
    agent_cfg["discriminator_reward_scale"] = 2
    agent_cfg["discriminator_logit_regularization_scale"] = 0.05
    agent_cfg["discriminator_gradient_penalty_scale"] = 5
    agent_cfg["discriminator_weight_decay_scale"] = 0.0001
    agent_cfg["state_preprocessor"] = RunningStandardScaler
    agent_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": env.device}
    agent_cfg["value_preprocessor"] = RunningStandardScaler
    agent_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}
    agent_cfg["amp_state_preprocessor"] = RunningStandardScaler
    agent_cfg["amp_state_preprocessor_kwargs"] = {"size": env.amp_observation_space, "device": env.device}
    # logging to TensorBoard and write checkpoints (in timesteps)
    agent_cfg["experiment"]["write_interval"] = 160
    agent_cfg["experiment"]["checkpoint_interval"] = 4000
    agent_cfg["experiment"]["directory"] = "runs/torch/HumanoidAMP"

    experiment_cfg["agent"]["rewards_shaper"] = None  # avoid 'dictionary changed size during iteration'

    agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})

    agent = AMP(models=models,
            memory=memory,
            cfg=agent_cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            amp_observation_space=env.amp_observation_space,
            motion_dataset=RandomMemory(memory_size=200000, device=env.device),
            reply_buffer=RandomMemory(memory_size=1000000, device=env.device),
            collect_reference_motions=lambda num_samples: env.fetch_amp_obs_demo(num_samples),
            collect_observation=lambda: env.get_observations()["policy"])

    # configure and instantiate a custom RL trainer for logging episode events
    # https://skrl.readthedocs.io/en/latest/api/trainers.html
    trainer_cfg = experiment_cfg["trainer"]
    trainer = SkrlSequentialLogTrainer(cfg=trainer_cfg, env=env, agents=agent)

    # train the agent
    trainer.train()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
