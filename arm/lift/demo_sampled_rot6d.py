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
from transform_utils import SE3_log_map, SE3_exp_map, quat_to_rot6d, rotvec_to_rot6d, rot6d_to_quat, rot6d_to_rotvec


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
            rotm0 = self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', np.pi/2).as_matrix()
            rotm1 = self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', np.pi/2).as_matrix()
            robot0_x_pass = 0.815
            robot0_y_pass = robot0_y_init
            robot0_z_pass = 0.4
            robot1_x_pass = 0.426
            robot1_y_pass = robot1_y_init
            robot1_z_pass = 0.4
        elif mode == 3:
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

        env = TwoArmLift(state_size=state_dim, action_size=action_dim)

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

    def parse_data(self, rollout):
        obs = rollout["observations"]
        actions = np.array(rollout["actions"])

        pos0 = actions[:,:3]
        rotvec0 = actions[:,3:6]
        gripper0 = actions[:,6]
        pos1 = actions[:,7:10]
        rotvec1 = actions[:,10:13]
        gripper1 = actions[:,13]

        rot6d_list0 = []
        for rv in rotvec0:
            rot6d_list0.append(rotvec_to_rot6d(rv))
        rot6d0 = np.array(rot6d_list0)

        rot6d_list1 = []
        for rv in rotvec1:
            rot6d_list1.append(rotvec_to_rot6d(rv))
        rot6d1 = np.array(rot6d_list1)

        actions = np.concatenate((pos0, rot6d0, gripper0.reshape(-1, 1), pos1, rot6d1, gripper1.reshape(-1, 1)), axis=1)

        robot0_eef_pos = np.array([o["robot0_eef_pos"] for o in obs])
        robot0_eef_quat = np.array([o["robot0_eef_quat"] for o in obs])
        robot0_gripper_pos = np.array([o["robot0_gripper_pos"] for o in obs])
        robot1_eef_pos = np.array([o["robot1_eef_pos"] for o in obs])
        robot1_eef_quat = np.array([o["robot1_eef_quat"] for o in obs])
        robot1_gripper_pos = np.array([o["robot1_gripper_pos"] for o in obs])

        repeats_needed = 250 - actions.shape[0]

        repeated_last = np.tile(actions[-1], (repeats_needed, 1))
        actions = np.vstack([actions, repeated_last])

        repeated_last = np.tile(robot0_eef_pos[-1], (repeats_needed, 1))
        robot0_eef_pos = np.vstack([robot0_eef_pos, repeated_last])
        state = robot0_eef_pos

        repeated_last = np.tile(robot0_eef_quat[-1], (repeats_needed, 1))
        robot0_eef_quat = np.vstack([robot0_eef_quat, repeated_last])
        eef_rot6d0 = []
        for q in robot0_eef_quat:
            eef_rot6d0.append(quat_to_rot6d(q))
        robot0_eef_rot6d = np.array(eef_rot6d0)
        state = np.hstack([state, robot0_eef_rot6d])


        repeated_last = np.tile(robot0_gripper_pos[-1], (repeats_needed, 1))
        robot0_gripper_pos = robot0_gripper_pos.reshape(-1, 1)
        robot0_gripper_pos = np.vstack([robot0_gripper_pos, repeated_last])
        state = np.hstack([state, robot0_gripper_pos])

        repeated_last = np.tile(robot1_eef_pos[-1], (repeats_needed, 1))
        robot1_eef_pos = np.vstack([robot1_eef_pos, repeated_last])
        state = np.hstack([state, robot1_eef_pos])

        repeated_last = np.tile(robot1_eef_quat[-1], (repeats_needed, 1))
        robot1_eef_quat = np.vstack([robot1_eef_quat, repeated_last])
        eef_rot6d1 = []
        for q in robot1_eef_quat:
            eef_rot6d1.append(quat_to_rot6d(q))
        robot1_eef_rot6d = np.array(eef_rot6d1)
        state = np.hstack([state, robot1_eef_rot6d])

        repeated_last = np.tile(robot1_gripper_pos[-1], (repeats_needed, 1))
        robot1_gripper_pos = robot1_gripper_pos.reshape(-1, 1)
        robot1_gripper_pos = np.vstack([robot1_gripper_pos, repeated_last])
        state = np.hstack([state, robot1_gripper_pos])

        return state, actions

    
    def get_demo(self, seed, mode, file_name = "rollouts_pot/rollout_seed0_mode2.pkl"):
        """
        Main file to get the demonstration data
        """
        obs = self.reset(seed, mode)
        obs = self.process_obs(obs)

        max_step_move = int(15 * self.control_freq)
        max_step_grip = int(0.9 * self.control_freq)

        expert_data = np.load("data/expert_actions_rot6d.npy")
        expert_data1 = expert_data[:, :, :10]
        expert_data2 = expert_data[:, :, 10:20]
        combined_data = np.concatenate((expert_data1, expert_data2), axis=0)
        mean = np.mean(combined_data, axis=(0,1))
        std = np.std(combined_data, axis=(0,1))

        model = self.load_model(type = "rot6d", state_dim = 10, action_dim = 10)
        with open(file_name, "rb") as f:
            rollout = pkl.load(f)
        state, actions = self.parse_data(rollout)

        obs1_init = state[0][:10]
        obs2_init = state[0][10:]
        obs1_final = state[-1][:10]
        obs2_final = state[-1][10:]
        obs1 = np.hstack([obs1_init, obs1_final])
        obs2 = np.hstack([obs2_init, obs2_final])
        obs1 = torch.FloatTensor(obs1).to(device).unsqueeze(0)
        obs2 = torch.FloatTensor(obs2).to(device).unsqueeze(0)

        traj_len = 250
        n_samples = 1

        sampled1 = model.sample(obs1, traj_len, n_samples, w=1., model_index=0)
        sampled2 = model.sample(obs2, traj_len, n_samples, w=1., model_index=1)
        sampled1 = sampled1.cpu().detach().numpy()[0]
        sampled2 = sampled2.cpu().detach().numpy()[0]
        sampled1 = sampled1 * std + mean
        sampled2 = sampled2 * std + mean

        pos1 = sampled1[:, :3]   # shape: (250, 3)
        rot6d1 = sampled1[:, 3:9]   # shape: (250, 6)
        grip1 = sampled1[:, 9]       # shape: (250,)
        pos2 = sampled2[:, :3]   # shape: (250, 3)
        rot6d2 = sampled2[:, 3:9]   # shape: (250, 6)
        grip2 = sampled2[:, 9]       # shape: (250,)

        rotvecs1 = np.array([rot6d_to_rotvec(row) for row in rot6d1])
        rotvecs2 = np.array([rot6d_to_rotvec(row) for row in rot6d2])

        combined1 = np.hstack([pos1, rotvecs1, grip1.reshape(-1,1)])
        combined2 = np.hstack([pos2, rotvecs2, grip2.reshape(-1,1)])

        for i in range(len(combined1)):
            action = np.hstack([combined1[i], combined2[i]])
            obs, reward, done, info = self.env.step(action)

            if self.render:
                self.env.render()


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
    player.get_demo(seed = 0, mode = 2)
    # for i in range(100):   
    #     rollout = player.get_demo(seed = i*10, mode = 2)
    #     with open("rollouts/rollout_seed%s_mode2.pkl" % (i*10), "wb") as f:
    #         pkl.dump(rollout, f)
    #     rollout = player.get_demo(seed = i*10, mode = 3)
    #     with open("rollouts/rollout_seed%s_mode3.pkl" % (i*10), "wb") as f:
    #         pkl.dump(rollout, f)
