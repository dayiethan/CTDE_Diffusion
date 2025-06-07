import argparse
import json
import os
import yaml
from pathlib import Path
import time
import copy

import hydra
import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as R

import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"
from robosuite.controllers import load_composite_controller_config

from custom_env.two_arm_handover_role import TwoArmHandoverRole

class DecentralizedPolicy:
    def __init__(self, agent_path, model_name):
        with open(Path(agent_path, "agent_config.yaml"), "r") as f:
            config_yaml = f.read()
            agent_config = yaml.safe_load(config_yaml)
        with open(Path(agent_path, "obs_config.yaml"), "r") as f:
            config_yaml = f.read()
            obs_config = yaml.safe_load(config_yaml)
        with open(Path(agent_path, "ac_norm.json"), "r") as f:
            ac_norm_dict = json.load(f)
            loc, scale = ac_norm_dict["loc"], ac_norm_dict["scale"]
            self.loc = np.array(loc).astype(np.float32)
            self.scale = np.array(scale).astype(np.float32)

        agent = hydra.utils.instantiate(agent_config)
        save_dict = torch.load(Path(agent_path, model_name), map_location="cpu", weights_only=False)
        agent.load_state_dict(save_dict["model"])
        self.agent = agent.eval().cuda()

        self.agents = ['robot0', 'robot1']
        self.transform = hydra.utils.instantiate(obs_config["transform"])
        
        self.img_keys, self.state_keys = dict(), dict()
        for agent_name in self.agents:
            self.img_keys[agent_name] = obs_config[agent_name]["imgs"]
            self.state_keys[agent_name] = obs_config[agent_name]["states"]
        
        print(f"loaded agent from {agent_path}, at step: {save_dict['global_step']}")
        self._last_time = None

    def _proc_image(self, agent, obs_dict, size=(256, 256)):
        torch_imgs = dict()
        for i, k in enumerate(self.img_keys[agent]):
            rgb_img = obs_dict[k][:, :, :3]
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            bgr_img = cv2.resize(bgr_img, size, interpolation=cv2.INTER_AREA)
            rgb_img = bgr_img[:, :, ::-1].copy()
            rgb_img = torch.from_numpy(rgb_img).float().permute((2, 0, 1)) / 255
            rgb_img = self.transform(rgb_img)[None].cuda()

            torch_imgs[f"cam{i}"] = rgb_img
        
        return torch_imgs
    
    def _proc_state(self, agent, obs_dict):
        state = []
        for i, k in enumerate(self.state_keys[agent]):
            state.append(obs_dict[k])
        state = np.concatenate([
                x if isinstance(x, np.ndarray) else np.array([x])
                for x in state
            ])
        state = np.array(state).astype(np.float32)
        return torch.from_numpy(state)[None].cuda()
    
    def forward(self, obs):
        action = {}
        for agent in self.agents:
            img = self._proc_image(agent, obs)
            state = self._proc_state(agent, obs)

            with torch.no_grad():
                ac = self.agent.get_actions(img, state)
            
            ac = ac[0, :].cpu().numpy().astype(np.float32)
            ac = np.clip(ac * self.scale + self.loc, -1, 1)  # denormalize the actions
            action[agent] = ac
        ac = np.concatenate([action[agent] for agent in self.agents], axis=1)
        print("Action shape:", ac.shape)

        cur_time = time.time()
        if self._last_time is not None:
            print("Effective HZ:", 1.0 / (cur_time - self._last_time))
        self._last_time = cur_time
        return ac
    

class EvalEnv:
    def __init__(self, env_config):
        controller_config = load_composite_controller_config(robot=env_config['robots'][0], controller=env_config['controller'])
        env = TwoArmHandoverRole(
            robots=env_config['robots'],
            controller_configs=controller_config,
            gripper_types="default",
            prehensile=env_config['prehensile'],
            has_renderer=True,
            render_camera=None,
            has_offscreen_renderer=True,                       
            horizon=env_config['env_horizon'], 
            use_camera_obs=True,
            camera_names=env_config['camera_names'],
            camera_heights=env_config['camera_heights'],
            camera_widths = env_config['camera_widths'],
            camera_depths = env_config['camera_depths'],
            control_freq=env_config['control_freq']
        )
        self.env = env
        robot0_base_body_id = self.env.sim.model.body_name2id("robot0_base")
        self.robot0_base_pos = self.env.sim.data.body_xpos[robot0_base_body_id]
        self.robot0_base_ori_rotm = self.env.sim.data.body_xmat[robot0_base_body_id].reshape((3,3))

        robot1_base_body_id = self.env.sim.model.body_name2id("robot1_base")
        self.robot1_base_pos = self.env.sim.data.body_xpos[robot1_base_body_id]
        self.robot1_base_ori_rotm = self.env.sim.data.body_xmat[robot1_base_body_id].reshape((3,3))

    def reset(self):
        np.random.seed(seed=2)
        obs = self.env.reset()
        
        jnt_id_0 = self.env.sim.model.joint_name2id("gripper0_right_finger_joint") #gripper0_right_finger_joint, gripper0_right_right_outer_knuckle_joint
        self.qpos_index_0 = self.env.sim.model.jnt_qposadr[jnt_id_0]
        jnt_id_1 = self.env.sim.model.joint_name2id("gripper1_right_finger_joint") #gripper0_right_finger_joint, gripper0_right_right_outer_knuckle_joint
        self.qpos_index_1 = self.env.sim.model.jnt_qposadr[jnt_id_1]

        return self._process_obs(obs)
    
    def step(self, action):
        action = self._process_action(action)
        obs, reward, done, info = self.env.step(action)
        return self._process_obs(obs), reward, done, info
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()

    def _get_poses(self, obs):
        robot0_pos_world = obs['robot0_eef_pos']
        robot0_rotm_world = R.from_quat(obs['robot0_eef_quat_site']).as_matrix()

        robot1_pos_world = obs['robot1_eef_pos']
        robot1_rotm_world = R.from_quat(obs['robot1_eef_quat_site']).as_matrix()

        robot0_pos = self.robot0_base_ori_rotm.T @ (robot0_pos_world - self.robot0_base_pos)
        robot0_rotm = self.robot0_base_ori_rotm.T @ robot0_rotm_world

        robot1_pos = self.robot1_base_ori_rotm.T @ (robot1_pos_world - self.robot1_base_pos)
        robot1_rotm = self.robot1_base_ori_rotm.T @ robot1_rotm_world
        
        return robot0_pos, robot0_rotm, robot1_pos, robot1_rotm
    
    def _process_obs(self, obs):

        processed_obs = copy.deepcopy(obs)
        robot0_pos, robot0_rotm, robot1_pos, robot1_rotm = self._get_poses(obs)

        processed_obs['robot0_eef_pos'] = robot0_pos
        processed_obs['robot0_eef_quat_site'] = R.from_matrix(robot0_rotm).as_quat()
        processed_obs['robot0_eef_rotvec'] = R.from_matrix(robot0_rotm).as_rotvec()
        processed_obs['robot0_eef_rot6d'] = robot0_rotm[:3, :2].T.flatten()

        processed_obs['robot1_eef_pos'] = robot1_pos
        processed_obs['robot1_eef_quat_site'] = R.from_matrix(robot1_rotm).as_quat()
        processed_obs['robot1_eef_rotvec'] = R.from_matrix(robot1_rotm).as_rotvec()
        processed_obs['robot1_eef_rot6d'] = robot1_rotm[:3, :2].T.flatten()

        processed_obs['robot0_gripper_pos'] = self.env.sim.data.qpos[self.qpos_index_0]
        processed_obs['robot1_gripper_pos'] = self.env.sim.data.qpos[self.qpos_index_1]

        return processed_obs
    
    def _process_action(self, action):
        env_action = []
        env_action.append(action[0:3])
        rotvec = _rot6d_to_rotvec(action[3:9])
        env_action.append(rotvec)
        env_action.append(action[9:13])  # 1(gripper) + 3(position)
        rotvec = _rot6d_to_rotvec(action[13:19])
        env_action.append(rotvec)
        env_action.append(action[19])  # gripper action
        env_action = np.concatenate([
            x if isinstance(x, np.ndarray) else np.array([x])
            for x in env_action
        ])
        # print("Processed action:", env_action.shape)

        return env_action

def _rot6d_to_rotvec(rot6d):
    x_raw = rot6d[:3]
    y_raw = rot6d[3:]
    x = x_raw / np.linalg.norm(x_raw)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    rotvec = R.from_matrix(np.column_stack((x, y, z))).as_rotvec()
    
    return rotvec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default='logs/decentralized/wandb_decentralized_combined_bimanual_decentralized_resnet_gn_2025-05-31_03-55-46/decentralized_combined.ckpt')
    parser.add_argument("--config_file", default="config/train/DiT_decentralized.yaml")
    args = parser.parse_args()

    # Set up policy
    agent_path = os.path.expanduser(os.path.dirname(args.checkpoint))
    model_name = args.checkpoint.split("/")[-1]
    policy = DecentralizedPolicy(agent_path, model_name)

    # Set up environment
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    env_config = config['env_parameters']
    env = EvalEnv(env_config)

    # Rollout
    for _ in range(5):
        obs = env.reset()
        print("Initial observation:", obs.keys())

        for _ in range(env_config['env_horizon']):
            actions = policy.forward(obs)
            for i in range(actions.shape[0]):
                obs, reward, done, info = env.step(actions[i])
                env.render()
                if done:
                    break
            if done:
                break
            # print(f"Step {i+1}, Action: {action}")

    env.close()

if __name__ == "__main__":
    main()