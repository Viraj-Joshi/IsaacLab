# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR
from omni.isaac.lab.utils.math import sample_uniform

from omni.isaac.lab_tasks.reward_utils import _gripper_caging_reward, tolerance, hamacher_product, to_torch
from gym import spaces
import numpy as np


@configclass
class FrankaPushBackEnvCfg(DirectRLEnvCfg):

    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    num_actions = 9
    num_observations = 39
    num_states = 0

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=3.0, replicate_physics=False)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}\\franka_usd\\franka_panda_gripper_for_usd\\franka_panda_gripper_for_usd.usd",
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": .564,
                "panda_joint3": 0,
                "panda_joint4": -2,
                "panda_joint5": -.28,
                "panda_joint6": 2.55,
                "panda_joint7": -.695,
                "panda_finger_joint.*": 0.035,
            },
            pos=(.55, 0.0, 1.027),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # Table
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=[1.2, 1.2, .054],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )

    # button_press = ArticulationCfg(
    #     prim_path="/World/envs/env_.*/ButtonPress",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}\\unified_objects\\buttonbox\\buttonbox.usd",
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=False,
    #             max_depenetration_velocity=5.0,
    #             kinematic_enabled=True
    #         ),
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         joint_pos={
    #             "btnbox_joint": 0,
    #         },
    #         pos=(0.0, 0.0, 1.12),
    #         rot=(0, 0, -0.7068252, 0.7073883),
    #     ),
    #     actuators={
    #         "btnbox_slide": ImplicitActuatorCfg(
    #             joint_names_expr=["btnbox_joint"],
    #             effort_limit=87.0,
    #             velocity_limit=2.175,
    #             stiffness=80.0,
    #             damping=4.0,
    #         ),
    #     }  
    # )

    box = RigidObjectCfg(
        prim_path="/World/envs/env_.*/obj",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}\\unified_objects\\push_back_box\\push_back_box.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0,0,0),
        )
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )


class FrankaPushBackEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaPushBackEnvCfg

    def __init__(self, cfg: FrankaPushBackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.target_pos = torch.zeros((self.num_envs,3), device=self.device)
        self.obj_init_pos = torch.zeros_like(self.target_pos)

        self.obj_pos = torch.zeros_like(self.target_pos)
        self.obj_quat = torch.zeros((self.num_envs,4)).to(self.device)
        
        # position and quaternion of 'second object' are zeros
        self.pos_2 = torch.zeros_like(self.obj_pos)
        self.quat_2 = torch.zeros_like(self.obj_quat)

        self.tcp_init = None
        self.prev_obs = torch.zeros((self.num_envs,18),device=self.device)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )

        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.eef_link_idx = self._robot.find_bodies("panda_grip_site")[0][0]
        self.obj_link_idx = self._box.find_bodies("obj")[0][0]

        self._table_surface_pos = [0, 0, 1.027]

        # define goals
        obj_low = (.2, -.1, self._table_surface_pos[2]+.02)
        obj_high = (.25, .1, self._table_surface_pos[2]+.02)

        goal_low = (0, -.1, self._table_surface_pos[2]+.0199)
        goal_high = (.1, .1, self._table_surface_pos[2]+.0201)

        goal_space = spaces.Box(np.array(goal_low),np.array(goal_high))
        self._random_reset_space = spaces.Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        
        # get random parameters
        self._random_reset_params = self._random_reset_space.sample()
        self.obj_init_pos[:,:3] = to_torch(self._random_reset_params[:3],device=self.device)
        self.target_pos[:,:3] = to_torch(self._random_reset_params[3:],device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._table = RigidObject(self.cfg.table)
        # self._button_press = Articulation(self.cfg.button_press)
        self._box = RigidObject(self.cfg.box)
        self.scene.articulations["robot"] = self._robot
        
        # print(self._table.body_names)
        # print(self._robot.body_names)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(truncated, dtype=torch.bool)  # never terminate
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]


        return self._compute_rewards(
            self.actions,
            robot_left_finger_pos,
            robot_right_finger_pos,
            self.tcp_init,
            self.target_pos,
            self.obj_pos,
            self.obj_init_pos
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # object state
       
        root_state = self._box.data.default_root_state[env_ids].clone()

        root_state[env_ids,:3] += self.scene.env_origins[env_ids] + self.obj_init_pos[env_ids]
        root_state[env_ids,3:7] = to_torch([1, 0, 0, 0], device=self.device)
        root_state[env_ids,7:13] = to_torch([0, 0, 0, 0, 0, 0], device=self.device)

        self._box.write_root_state_to_sim(root_state, env_ids)
        self._box.reset(env_ids)

    def _get_observations(self) -> dict:
        # get init tcp
        if self.tcp_init is None:
            self.tcp_init = ((self._robot.data.body_pos_w[:, self.left_finger_link_idx] + self._robot.data.body_pos_w[:, self.right_finger_link_idx]) / 2).clone()
        eef_pos = self._robot.data.body_pos_w[:, self.eef_link_idx]
        eef_quat = self._robot.data.body_quat_w[:, self.eef_link_idx]

        self.franka_lfinger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        self.franka_rfinger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        gripper_distance_apart = torch.norm(self.franka_rfinger_pos-self.franka_lfinger_pos,dim=-1)
        normalized_openess = torch.clip(gripper_distance_apart/.095,0.0,1.0).unsqueeze(-1)

        obj_pos = self._box.data.body_pos_w[:, self.obj_link_idx]
        obj_quat = self._box.data.body_quat_w[:, self.obj_link_idx]

        # print("IN OBSERVATION")
        # print(obj_pos)

        cur_obs = torch.cat(
            (
                eef_pos,
                normalized_openess,
                obj_pos,
                obj_quat,
                torch.zeros_like(obj_pos),
                torch.zeros_like(obj_quat),
            ),
            dim=-1,
        )
        
        obs = torch.cat(
            (
                cur_obs,
                self.prev_obs,
                self.target_pos
            ),
            dim=-1
        )

        self.prev_obs = cur_obs.clone()

        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods

    def _compute_rewards(
        self, actions: torch.Tensor,
        franka_lfinger_pos: torch.Tensor, franka_rfinger_pos: torch.Tensor, 
        tcp_init: torch.Tensor, target_pos: torch.Tensor, obj_pos: torch.Tensor, obj_init_pos: torch.Tensor
    ):
        
        TARGET_RADIUS = 0.05    
    
        tcp = (franka_lfinger_pos + franka_rfinger_pos) / 2
        tcp_to_obj = torch.norm(obj_pos - tcp,dim=-1)
        target_to_obj = torch.norm(obj_pos - target_pos,dim=-1)
        target_to_obj_init = torch.norm(obj_init_pos - target_pos,dim=-1)

        in_place = tolerance(
            target_to_obj,
            bounds=(0.0, TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid="long_tail",
        )
        
        # create a normalized 0 to 1 measurement of how open the gripper is, where 1 is fully open and 0 is fully closed
        gripper_distance_apart = torch.norm(franka_rfinger_pos-franka_lfinger_pos,dim=-1)
        tcp_opened = torch.clip(gripper_distance_apart/.095,0.0,1.0)

        object_grasped = _gripper_caging_reward(
            obj_pos,
            franka_lfinger_pos,
            franka_rfinger_pos,
            tcp,
            tcp_init,
            actions,
            obj_init_pos,
            object_reach_radius=0.01,
            obj_radius=0.015,
            pad_success_thresh=0.05,
            xz_thresh=0.005,
            high_density=True,
        )
        rewards = hamacher_product(object_grasped, in_place)

        rewards = torch.where((tcp_to_obj < .02) & (tcp_opened > 0) , rewards + 1.0 + 5.0 * in_place, rewards)
        
        success = (target_to_obj < TARGET_RADIUS)

        rewards = torch.where(success, 10, rewards)

        return rewards
