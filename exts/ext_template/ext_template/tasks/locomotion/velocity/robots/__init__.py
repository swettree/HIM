import gymnasium as gym

from ext_template.tasks.locomotion.velocity.config.go1.agents.rsl_rl_ppo_cfg import UnitreeGo1RoughPPORunnerCfg

from . import go1_him_env_cfg

##
# Create PPO runners for RSL-RL
##

go1_blind_rough_runner_cfg_v0 = UnitreeGo1RoughPPORunnerCfg()
go1_blind_rough_runner_cfg_v0.experiment_name = "unitree_go1_rough_v0"
go1_blind_rough_runner_cfg_v0.run_name = "v0"

##
# Register Gym environments
##

#############################
# UNITREE GO1 Blind Rough Environment HIM v0
#############################
gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go1-HIM-v0",
    entry_point="ext_template.envs:HIMManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_him_env_cfg.UnitreeGo1RoughEnvCfg_v0,
        "rsl_rl_cfg_entry_point": go1_blind_rough_runner_cfg_v0,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go1-Play-HIM-v0",
    entry_point="ext_template.envs:HIMManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_him_env_cfg.UnitreeGo1RoughEnvCfg_PLAY_v0,
        "rsl_rl_cfg_entry_point": go1_blind_rough_runner_cfg_v0,
    },
)