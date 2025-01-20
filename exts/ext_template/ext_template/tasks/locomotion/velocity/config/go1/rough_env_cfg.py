# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from ext_template.tasks.locomotion.velocity.velocity_env_cfg  import LocomotionVelocityRoughEnvCfg 

##
# Pre-defined configs
##
from ext_template.assets.unitree import UNITREE_GO1_CFG  # isort: skip



@configclass
class UnitreeGo1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()


        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # print(self.scene.robot.joint_names)
        # 高度扫描仪
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        # self.events.push_robot = None
        self.events.physics_material.params["static_friction_range"] = (0.05, 4.5)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 0.9)      
        self.events.physics_material.params["restitution_range"] = (0.0, 1.0)   
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),   
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.randomize_joint_parameters = None

        # rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.lin_vel_z_l2.weight=-2.0
        self.rewards.ang_vel_xy_l2.weight=-0.05
        self.rewards.dof_torques_l2.weight=-0.0002
        self.rewards.dof_acc_l2.weight=-2.5e-7
        self.rewards.action_rate_l2.weight=-0.01

        # -- optional penalties
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [".*_foot"]

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_thigh", ".*_calf"]
 
        self.rewards.flat_orientation_l2.weight = -0.5
        
        self.rewards.joint_pos_limits.weight = -5 
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = -2e-5
        
        self.rewards.stand_still_when_zero_command.weight = -0.5

        self.rewards.base_height_l2.weight = -2.0
        self.rewards.base_height_l2.params["target_height"] = 0.3
        self.rewards.base_height_l2.params["asset_cfg"].body_names = ["trunk"]

        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [".*_foot"]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [".*_foot"]

        self.rewards.feet_clearance.weight = -0.01
        self.rewards.feet_clearance.params["target_feet_height"] = -0.2

        self.rewards.action_smoothness_penalty.weight = -0.01

        if self.__class__.__name__ == "UnitreeGo1RoughEnvCfg":
            self.disable_zero_weight_rewards()

        
        # huangding
        # self.observations.policy.base_lin_vel = None

        self.observations.policy.height_scan= None
        # HIM
        self.observations.policy.history_length = 5
        self.observations.critic.history_length = 1
        self.observations.before_step_Critic.history_length = 1

        self.observations.policy.flatten_history_dim = False
        self.observations.policy.concatenate_terms = False
        
        self.observations.critic.flatten_history_dim = False
        self.observations.critic.concatenate_terms = False

        self.observations.before_step_Critic.flatten_history_dim = False
        self.observations.before_step_Critic.concatenate_terms = False
        # 非对称AC
        # self.observations.policy.history_length = 1
        # self.observations.critic.history_length = 5

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"

        

@configclass
class UnitreeGo1RoughEnvCfg_PLAY(UnitreeGo1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # self.observations.critic = None
        # remove random pushing event
        self.events.base_external_force_torque = None
        # self.events.push_robot = None
        
