import time
import torch

import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle as pkl
import copy

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from conditional_Action_DiT import Conditional_ODE

from env import TwoArmLiftRole

from scipy.spatial.transform import Rotation as R
from transform_utils import SE3_log_map, SE3_exp_map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TwoArmLift():
    def __init__(self, state_size=7, action_size=7):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoArmLift"

class PolicyPlayer:
    def __init__ (self, env, render = False):
        self.env = env

        self.control_freq = env.control_freq
        self.dt = 1.0 / self.control_freq
        self.max_time = 10
        self.max_steps = int(self.max_time / self.dt)

        self.render = render

        # robot0_base_body_id = env.sim.model.body_name2id("robot0:base")
        # possible: 'robot0_base', 'robot0_fixed_base_link', 'robot0_shoulder_link'

        # Extract the base position and orientation (quaternion) from the simulation data
        robot0_base_body_id = self.env.sim.model.body_name2id("robot0_base")
        self.robot0_base_pos = self.env.sim.data.body_xpos[robot0_base_body_id]
        self.robot0_base_ori_rotm = self.env.sim.data.body_xmat[robot0_base_body_id].reshape((3,3))

        robot1_base_body_id = self.env.sim.model.body_name2id("robot1_base")
        self.robot1_base_pos = self.env.sim.data.body_xpos[robot1_base_body_id]
        self.robot1_base_ori_rotm = self.env.sim.data.body_xmat[robot1_base_body_id].reshape((3,3))

        # Rotation matrix of robots for the home position, both in their own base frame
        self.R_be_home = np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, -1]])

        # robot0_init_rotm_world = R.from_quat(obs['robot0_eef_quat_site'], scalar_first = False).as_matrix()
        # robot1_init_rotm_world = R.from_quat(obs['robot1_eef_quat_site'], scalar_first = False).as_matrix()

        self.n_action = self.env.action_spec[0].shape[0]

        # Setting up constants
        self.pot_handle_offset_z = 0.012
        self.pot_handle_offset_x = 0.015
        self.pot_handle_offset = np.array([self.pot_handle_offset_x, 0, self.pot_handle_offset_z])
        self.pot_handle0_pos = self.robot0_base_ori_rotm.T @ (self.env._handle0_xpos - self.robot0_base_pos) + self.pot_handle_offset
        self.pot_handle1_pos = self.robot1_base_ori_rotm.T @ (self.env._handle1_xpos - self.robot1_base_pos) + self.pot_handle_offset

        self.setup_waypoints()

    def reset(self, seed = 0, mode = 1):
        """
        Resetting environment. Re-initializing the waypoints too.
        """
        np.random.seed(seed)
        obs = self.env.reset()

        # Setting up constants
        self.pot_handle_offset_z = 0.012
        self.pot_handle_offset_x = 0.015
        self.pot_handle_offset = np.array([self.pot_handle_offset_x, 0, self.pot_handle_offset_z])
        self.pot_handle0_pos = self.robot0_base_ori_rotm.T @ (self.env._handle0_xpos - self.robot0_base_pos) + self.pot_handle_offset
        self.pot_handle1_pos = self.robot1_base_ori_rotm.T @ (self.env._handle1_xpos - self.robot1_base_pos) + self.pot_handle_offset
        jnt_id_0 = self.env.sim.model.joint_name2id("gripper0_right_finger_joint") #gripper0_right_finger_joint, gripper0_right_right_outer_knuckle_joint
        self.qpos_index_0 = self.env.sim.model.jnt_qposadr[jnt_id_0]
        jnt_id_1 = self.env.sim.model.joint_name2id("gripper1_right_finger_joint") #gripper0_right_finger_joint, gripper0_right_right_outer_knuckle_joint
        self.qpos_index_1 = self.env.sim.model.jnt_qposadr[jnt_id_1]

        self.setup_waypoints(mode = mode)

        self.rollout = {}
        self.rollout["observations"] = []
        self.rollout["actions"] = []

        return obs

    def setup_waypoints(self, mode = 1):
        self.waypoints_robot0 = []
        self.waypoints_robot1 = []

        robot0_x_init = self.pot_handle0_pos[0]
        robot0_y_init = self.pot_handle0_pos[1]
        robot0_z_init = self.pot_handle0_pos[2]
        robot1_x_init = self.pot_handle1_pos[0]
        robot1_y_init = self.pot_handle1_pos[1]
        robot1_z_init = self.pot_handle1_pos[2]

        if mode == 1:
            rotm0 = self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', np.pi/2).as_matrix()
            rotm1 = self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', np.pi/2).as_matrix()
            robot0_x_pass = robot0_x_init
            robot0_y_pass = robot0_y_init
            robot0_z_pass = 0.45
            robot1_x_pass = robot1_x_init
            robot1_y_pass = robot1_y_init
            robot1_z_pass = 0.45
        elif mode == 2:
            rotm0 = self.R_be_home @ R.from_euler('z', -np.pi/2).as_matrix()
            rotm0 = self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', np.pi/2).as_matrix()
            rotm1 = self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', np.pi/2).as_matrix()
            robot0_x_pass = 0.815
            robot0_y_pass = robot0_y_init
            robot0_z_pass = 0.4
            robot1_x_pass = 0.426
            robot1_y_pass = robot1_y_init
            robot1_z_pass = 0.4
        elif mode == 3:
            rotm0 = self.R_be_home @ R.from_euler('z', -np.pi/2).as_matrix()
            rotm0 = self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', np.pi/2).as_matrix()
            rotm1 = self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', np.pi/2).as_matrix()
            robot0_x_pass = 0.414
            robot0_y_pass = robot0_y_init
            robot0_z_pass = 0.4
            robot1_x_pass = 0.826
            robot1_y_pass = robot1_y_init
            robot1_z_pass = 0.4
        else:
            raise ValueError("Invalid mode. Please choose a valid mode (1, 2, or 3).")

        """
        Robot 0 Waypoints
        """

        #wp0
        waypoint = {"goal_pos": np.array([robot0_x_init, robot0_y_pass, robot0_z_init]),
                     "goal_rotm": rotm0,
                     "gripper": -1}
        self.waypoints_robot0.append(waypoint)

        #wp1
        waypoint = {"goal_pos": np.array([robot0_x_init, robot0_y_pass, robot0_z_init]),
                     "goal_rotm": rotm0,
                     "gripper": 1}
        self.waypoints_robot0.append(waypoint)

        #wp2
        waypoint = {"goal_pos": np.array([robot0_x_pass, robot0_y_pass, robot0_z_pass]),
                     "goal_rotm": rotm0,
                     "gripper": 1}
        self.waypoints_robot0.append(waypoint)
        
        #wp3
        waypoint = {"goal_pos": np.array([robot0_x_pass, -robot0_y_pass, robot0_z_pass]),
                     "goal_rotm": rotm0,
                     "gripper": 1}
        self.waypoints_robot0.append(waypoint)

        #wp4
        waypoint = {"goal_pos": np.array([robot0_x_init, -robot0_y_pass, robot0_z_init]),
                     "goal_rotm": rotm0,
                     "gripper": 1}
        self.waypoints_robot0.append(waypoint)

        #wp5
        waypoint = {"goal_pos": np.array([robot0_x_init, -robot0_y_pass, robot0_z_init]),
                     "goal_rotm": rotm0,
                     "gripper": -1}
        self.waypoints_robot0.append(waypoint)

        """
        Robot 1 Waypoints
        """

        #wp0
        waypoint = {"goal_pos": np.array([robot1_x_init, robot1_y_pass, robot1_z_init]),
                     "goal_rotm": rotm1,
                     "gripper": -1}
        self.waypoints_robot1.append(waypoint)

        #wp1
        waypoint = {"goal_pos": np.array([robot1_x_init, robot1_y_pass, robot1_z_init]),
                     "goal_rotm": rotm1,
                     "gripper": 1}
        self.waypoints_robot1.append(waypoint)

        #wp2
        waypoint = {"goal_pos": np.array([robot1_x_pass, robot1_y_pass, robot1_z_pass]),
                     "goal_rotm": rotm1,
                     "gripper": 1}
        self.waypoints_robot1.append(waypoint)

        #wp3
        waypoint = {"goal_pos": np.array([robot1_x_pass, -robot1_y_pass, robot1_z_pass]),
                     "goal_rotm": rotm1,
                     "gripper": 1}
        self.waypoints_robot1.append(waypoint)

        #wp4
        waypoint = {"goal_pos": np.array([robot1_x_init, -robot1_y_pass, robot1_z_init]),
                     "goal_rotm": rotm1,
                     "gripper": 1}
        self.waypoints_robot1.append(waypoint)

        #wp5
        waypoint = {"goal_pos": np.array([robot1_x_init, -robot1_y_pass, robot1_z_init]),
                     "goal_rotm": rotm1,
                     "gripper": -1}
        self.waypoints_robot1.append(waypoint)

        
    def convert_action_robot(self, robot_pos, robot_rotm, robot_goal_pos, robot_goal_rotm, robot_gripper, alpha = 0.5):
        action = np.zeros(int(self.n_action/2))

        g = np.eye(4)
        g[0:3, 0:3] = robot_rotm
        g[0:3, 3] = robot_pos

        gd = np.eye(4)
        gd[0:3, 0:3] = robot_goal_rotm
        gd[0:3, 3] = robot_goal_pos

        xi = SE3_log_map(np.linalg.inv(g) @ gd)

        gd_modified = g @ SE3_exp_map(alpha * xi)

        action[0:3] = gd_modified[:3,3]
        action[3:6] = R.from_matrix(gd_modified[:3,:3]).as_rotvec()
        action[6] = robot_gripper

        return action
    
    
    def get_poses(self, obs):
        robot0_pos_world = obs['robot0_eef_pos']
        robot0_rotm_world = R.from_quat(obs['robot0_eef_quat_site']).as_matrix()

        robot1_pos_world = obs['robot1_eef_pos']
        robot1_rotm_world = R.from_quat(obs['robot1_eef_quat_site']).as_matrix()

        robot0_pos = self.robot0_base_ori_rotm.T @ (robot0_pos_world - self.robot0_base_pos)
        robot0_rotm = self.robot0_base_ori_rotm.T @ robot0_rotm_world

        robot1_pos = self.robot1_base_ori_rotm.T @ (robot1_pos_world - self.robot1_base_pos)
        robot1_rotm = self.robot1_base_ori_rotm.T @ robot1_rotm_world
        
        return robot0_pos, robot0_rotm, robot1_pos, robot1_rotm
    
    def check_arrived(self, pos1, rotm1, pos2, rotm2, threshold = 0.05):
        pos_diff = pos1 - pos2
        rotm_diff = rotm2.T @ rotm1


        distance = np.sqrt(0.5 * np.linalg.norm(pos_diff)**2 + np.trace(np.eye(3) - rotm_diff))

        if distance < threshold:
            return True
        else:
            return False
        
    
    def load_model(self, type = "rotvec", state_dim = 7, action_dim = 7):
        n_gradient_steps = 100_000
        batch_size = 64
        model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
        H = 250 # horizon, length of each trajectory

        expert_data = np.load("data/expert_actions_"+type+".npy")
        expert_data1 = expert_data[:, :, :action_dim]
        expert_data2 = expert_data[:, :, action_dim:action_dim*2]

        states = np.load("data/expert_states_"+type+".npy")
        states1 = states[:, :, :state_dim]
        states2 = states[:, :, state_dim:state_dim*2]


        # Compute mean and standard deviation
        combined_data = np.concatenate((expert_data1, expert_data2), axis=0)
        mean = np.mean(combined_data, axis=(0,1))
        std = np.std(combined_data, axis=(0,1))

        # Normalize data
        expert_data1 = (expert_data1 - mean) / std
        expert_data2 = (expert_data2 - mean) / std

        # Prepare Data for Training
        X_train1 = []
        Y_train1 = []
        for traj in expert_data1:
            for i in range(len(traj) - 1):
                X_train1.append(traj[i])  # Current state + goal
                Y_train1.append(traj[i + 1])  # Next state
        X_train1 = torch.tensor(np.array(X_train1), dtype=torch.float32) # Shape: (N, 7)
        Y_train1 = torch.tensor(np.array(Y_train1), dtype=torch.float32) # Shape: (N, 7)

        X_train2 = []
        Y_train2 = []
        for traj in expert_data2:
            for i in range(len(traj) - 1):
                X_train2.append(traj[i])  # Current state + goal
                Y_train2.append(traj[i + 1])  # Next state
        X_train2 = torch.tensor(np.array(X_train2), dtype=torch.float32)  # Shape: (N, 7)
        Y_train2 = torch.tensor(np.array(Y_train2), dtype=torch.float32)  # Shape: (N, 7)

        env = TwoArmLift()

        obs_init1 = states1[:, 0, :]
        obs_init2 = states2[:, 0, :]
        obs_final1 = states1[:, -1, :]
        obs_final2 = states2[:, -1, :]
        obs1 = np.hstack([obs_init1, obs_final1])
        obs2 = np.hstack([obs_init2, obs_final2])
        actions1 = expert_data1[:, :H-1, :]
        actions2 = expert_data2[:, :H-1, :]
        obs1 = torch.FloatTensor(obs1).to(device)
        obs2 = torch.FloatTensor(obs2).to(device)

        attr1 = obs1
        attr2 = obs2
        attr_dim1 = attr1.shape[1]
        attr_dim2 = attr2.shape[1]
        assert attr_dim1 == env.state_size * 2
        assert attr_dim2 == env.state_size * 2

        actions1 = torch.FloatTensor(actions1).to(device)
        actions2 = torch.FloatTensor(actions2).to(device)
        sigma_data1 = actions1.std().item()
        sigma_data2 = actions2.std().item()

        action_cond_ode = Conditional_ODE(env, [attr_dim1, attr_dim2], [sigma_data1, sigma_data2], device=device, N=100, n_models = 2, **model_size)
        action_cond_ode.load(extra="_T250_"+type)

        return action_cond_ode

    
    def get_demo(self, seed, mode):
        """
        Main file to get the demonstration data
        """
        obs = self.reset(seed, mode)

        max_step_move = int(15 * self.control_freq) # 15 seconds
        max_step_grip = int(0.9 * self.control_freq)

        robot0_arrived = False
        robot1_arrived = False

        model = self.load_model(type = "rotvec", state_dim = 7, action_dim = 7)

        # stage 1: robot0 move to wp0 robot1 moves to wp0
        for i in range(max_step_move):
            robot0_pos, robot0_rotm, robot1_pos, robot1_rotm = self.get_poses(obs)

            if not robot0_arrived:
                goal_pos0 = self.waypoints_robot0[0]["goal_pos"]
                goal_rotm0 = self.waypoints_robot0[0]["goal_rotm"]
                action0 = self.convert_action_robot(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, self.waypoints_robot0[0]["gripper"])
                robot0_arrived = self.check_arrived(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, threshold = 0.05)

            if not robot1_arrived:
                goal_pos1 = self.waypoints_robot1[0]["goal_pos"]
                goal_rotm1 = self.waypoints_robot1[0]["goal_rotm"]
                action1 = self.convert_action_robot(robot1_pos, robot1_rotm,goal_pos1, goal_rotm1, self.waypoints_robot1[0]["gripper"])
                robot1_arrived = self.check_arrived(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, threshold = 0.05)


            action = np.hstack([action0, action1])
            obs, reward, done, info = self.env.step(action)
            self.rollout["observations"].append(self.process_obs(obs))
            self.rollout["actions"].append(action)

            if self.render:
                self.env.render()

            if robot0_arrived and robot1_arrived:
                break

        # stage 2: robot0 closes gripper (wp1) robot1 closes gripper (wp2)
        robot0_arrived = False
        robot1_arrived = False
        for i in range(max_step_grip):
            robot0_pos, robot0_rotm, robot1_pos, robot1_rotm = self.get_poses(obs)

            if not robot0_arrived:
                goal_pos0 = self.waypoints_robot0[1]["goal_pos"]
                goal_rotm0 = self.waypoints_robot0[1]["goal_rotm"]
                action0 = self.convert_action_robot(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, self.waypoints_robot0[1]["gripper"])
                robot0_arrived = self.check_arrived(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, threshold = 0.05)

            if not robot1_arrived:
                goal_pos1 = self.waypoints_robot1[1]["goal_pos"]
                goal_rotm1 = self.waypoints_robot1[1]["goal_rotm"]
                action1 = self.convert_action_robot(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, self.waypoints_robot1[1]["gripper"])
                robot1_arrived = self.check_arrived(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, threshold = 0.05)

            action = np.hstack([action0, action1])
            obs, reward, done, info = self.env.step(action)
            self.rollout["observations"].append(self.process_obs(obs))
            self.rollout["actions"].append(action)

            if self.render:
                self.env.render()

        # stage 3: robot0 move to wp2, robot1 moves to wp2
        robot0_arrived = False
        robot1_arrived = False
        for i in range(max_step_move):
            robot0_pos, robot0_rotm, robot1_pos, robot1_rotm = self.get_poses(obs)

            if not robot0_arrived:
                goal_pos0 = self.waypoints_robot0[2]["goal_pos"]
                goal_rotm0 = self.waypoints_robot0[2]["goal_rotm"]
                action0 = self.convert_action_robot(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, self.waypoints_robot0[2]["gripper"])
                robot0_arrived = self.check_arrived(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, threshold = 0.05)

            if not robot1_arrived:
                goal_pos1 = self.waypoints_robot1[2]["goal_pos"]
                goal_rotm1 = self.waypoints_robot1[2]["goal_rotm"]
                action1 = self.convert_action_robot(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, self.waypoints_robot1[2]["gripper"])
                robot1_arrived = self.check_arrived(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, threshold = 0.05)


            action = np.hstack([action0, action1])
            obs, reward, done, info = self.env.step(action)
            self.rollout["observations"].append(self.process_obs(obs))
            self.rollout["actions"].append(action)

            if self.render:
                self.env.render()

            if robot0_arrived and robot1_arrived:
                break

        # stage 4: robot0 move to wp3 robot1 moves to wp3
        robot0_arrived = False
        robot1_arrived = False
        for i in range(max_step_move):
            robot0_pos, robot0_rotm, robot1_pos, robot1_rotm = self.get_poses(obs)

            if not robot0_arrived:
                goal_pos0 = self.waypoints_robot0[3]["goal_pos"]
                goal_rotm0 = self.waypoints_robot0[3]["goal_rotm"]
                action0 = self.convert_action_robot(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, self.waypoints_robot0[3]["gripper"])
                robot0_arrived = self.check_arrived(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, threshold = 0.05)

            if not robot1_arrived:
                goal_pos1 = self.waypoints_robot1[3]["goal_pos"]
                goal_rotm1 = self.waypoints_robot1[3]["goal_rotm"]
                action1 = self.convert_action_robot(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, self.waypoints_robot1[3]["gripper"])
                robot1_arrived = self.check_arrived(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, threshold = 0.05)


            action = np.hstack([action0, action1])
            obs, reward, done, info = self.env.step(action)
            self.rollout["observations"].append(self.process_obs(obs))
            self.rollout["actions"].append(action)

            if self.render:
                self.env.render()

            if robot0_arrived and robot1_arrived:
                break

        # stage 5: robot0 move to wp4 robot1 move to wp4
        robot0_arrived = False
        robot1_arrived = False
        for i in range(max_step_move):
            robot0_pos, robot0_rotm, robot1_pos, robot1_rotm = self.get_poses(obs)

            if not robot0_arrived:
                goal_pos0 = self.waypoints_robot0[4]["goal_pos"]
                goal_rotm0 = self.waypoints_robot0[4]["goal_rotm"]
                action0 = self.convert_action_robot(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, self.waypoints_robot0[4]["gripper"])
                robot0_arrived = self.check_arrived(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, threshold = 0.05)

            if not robot1_arrived:
                goal_pos1 = self.waypoints_robot1[4]["goal_pos"]
                goal_rotm1 = self.waypoints_robot1[4]["goal_rotm"]
                action1 = self.convert_action_robot(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, self.waypoints_robot1[4]["gripper"])
                robot1_arrived = self.check_arrived(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, threshold = 0.05)


            action = np.hstack([action0, action1])
            obs, reward, done, info = self.env.step(action)
            self.rollout["observations"].append(self.process_obs(obs))
            self.rollout["actions"].append(action)

            if self.render:
                self.env.render()

            if robot0_arrived and robot1_arrived:
                break

        # stage 6: robot0 closes gripper (wp5) robot1 closes gripper (wp5)
        robot0_arrived = False
        robot1_arrived = False
        for i in range(max_step_grip):
            robot0_pos, robot0_rotm, robot1_pos, robot1_rotm = self.get_poses(obs)

            if not robot0_arrived:
                goal_pos0 = self.waypoints_robot0[5]["goal_pos"]
                goal_rotm0 = self.waypoints_robot0[5]["goal_rotm"]
                action0 = self.convert_action_robot(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, self.waypoints_robot0[5]["gripper"])
                robot0_arrived = self.check_arrived(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, threshold = 0.05)

            if not robot1_arrived:
                goal_pos1 = self.waypoints_robot1[5]["goal_pos"]
                goal_rotm1 = self.waypoints_robot1[5]["goal_rotm"]
                action1 = self.convert_action_robot(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, self.waypoints_robot1[5]["gripper"])
                robot1_arrived = self.check_arrived(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, threshold = 0.05)

            action = np.hstack([action0, action1])
            obs, reward, done, info = self.env.step(action)
            self.rollout["observations"].append(self.process_obs(obs))
            self.rollout["actions"].append(action)

            if self.render:
                self.env.render()
        
        return self.rollout


    def process_obs(self, obs):

        processed_obs = copy.deepcopy(obs)
        robot0_pos, robot0_rotm, robot1_pos, robot1_rotm = self.get_poses(obs)

        processed_obs['robot0_eef_pos'] = robot0_pos
        processed_obs['robot0_eef_quat_site'] = R.from_matrix(robot0_rotm).as_quat()

        processed_obs['robot1_eef_pos'] = robot1_pos
        processed_obs['robot1_eef_quat_site'] = R.from_matrix(robot1_rotm).as_quat()

        processed_obs['robot0_gripper_pos'] = self.env.sim.data.qpos[self.qpos_index_0]
        processed_obs['robot1_gripper_pos'] = self.env.sim.data.qpos[self.qpos_index_1]

        return processed_obs
            

    
        
if __name__ == "__main__":
    CAMERA_NAMES = ['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand', 'robot1_robotview', 'robot1_eye_in_hand']
    controller_config = load_composite_controller_config(robot="Kinova3", controller="kinova.json")

    env = TwoArmLiftRole(
    robots=["Kinova3", "Kinova3"],
    gripper_types="default",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    render_camera=None,
    )

    player = PolicyPlayer(env, render = False)
    rollout = player.get_demo(seed = 100, mode = 2)
    # for i in range(100):   
    #     rollout = player.get_demo(seed = i*10, mode = 2)
    #     with open("rollouts/rollout_seed%s_mode2.pkl" % (i*10), "wb") as f:
    #         pkl.dump(rollout, f)
    #     rollout = player.get_demo(seed = i*10, mode = 3)
    #     with open("rollouts/rollout_seed%s_mode3.pkl" % (i*10), "wb") as f:
    #         pkl.dump(rollout, f)
