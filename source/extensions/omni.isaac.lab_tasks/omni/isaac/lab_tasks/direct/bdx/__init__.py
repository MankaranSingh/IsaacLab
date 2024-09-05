# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents
from .bdx_env import BDXEnv, BDXFlatEnvCfg, BDXRoughEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-BDX-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.bdx:BDXEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BDXFlatEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.BDXFlatPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-BDX-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.bdx:BDXEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BDXRoughEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.BDXRoughPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)
