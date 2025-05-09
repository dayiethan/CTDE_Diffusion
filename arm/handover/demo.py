# Generates demonstrations for the Two Arm Handover task

import time
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import copy
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from transform_utils import *
import pdb
from env import TwoArmHandoverRole
from scipy.spatial.transform import Rotation as R

class PolicyPlayer:
    def __init__ (self, env, render= True):
        
        
        self.env = env

        self.control_freq = env.control_freq
        self.dt = 1.0 / self.control_freq
        self.max_time = 10
        self.max_steps = int(self.max_time / self.dt)

        self.render = render

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

        obs = self.reset()
        self.n_action = self.env.action_spec[0].shape[0]

    def reset(self, seed = 0, mode = 1):
        """
        Resetting environment. Re-initializing the waypoints too.
        """
        np.random.seed(seed)
        obs = self.env.reset()

        self.handle_length = self.env.hammer.handle_length
        self.hammer_headsize = 2*self.env.hammer.head_halfsize

        self.hammer_pos0 = self.robot0_base_ori_rotm.T @ (self.env._hammer_pos - self.robot0_base_pos)
        self.hammer_pos1 = self.robot1_base_ori_rotm.T @ (self.env._hammer_pos - self.robot1_base_pos)
        self.hammer_rotm = quat_to_rotm(self.env._hammer_quat)
        jnt_id_0 = self.env.sim.model.joint_name2id("gripper0_right_finger_joint") #gripper0_right_finger_joint, gripper0_right_right_outer_knuckle_joint
        self.qpos_index_0 = self.env.sim.model.jnt_qposadr[jnt_id_0]
        jnt_id_1 = self.env.sim.model.joint_name2id("gripper1_right_finger_joint") #gripper0_right_finger_joint, gripper0_right_right_outer_knuckle_joint
        self.qpos_index_1 = self.env.sim.model.jnt_qposadr[jnt_id_1]

        self.rollout = {}
        self.rollout["observations"] = []
        self.rollout["actions"] = []
        self.setup_waypoints(mode = mode)

        return obs

    def setup_waypoints(self, mode = 1):
        self.waypoints_robot0 = []
        self.waypoints_robot1 = []

        pass_x = 0.55   # position of where the hammer is passed
        pass_y = 0      # position of where the hammer is passed
        pass_z = 0.1    # position of where the hammer is passed
        giver_x_offset = 0.05       # offset for the giver robot at handover
        receiver_x_offset = 0.24    # offset for the receiver robot at handover

        if mode == 1:
            """
            Robot 0 Waypoints
            """
            self.pickup_pos = self.hammer_pos0 + np.array([0, -self.handle_length/2 + 0.03, 0.02])
            #wp0
            waypoint = {"goal_pos": self.hammer_pos0 + np.array([0, -self.handle_length/2 + 0.03, 0.02]),
                        "goal_rotm": self.hammer_rotm @ R.from_euler('y', -np.pi/2).as_matrix() @ R.from_euler('z', -np.pi/2).as_matrix(),
                        "gripper": -1}
            self.waypoints_robot0.append(waypoint)

            #wp1
            waypoint = {"goal_pos": self.hammer_pos0 + np.array([0, -self.handle_length/2 + 0.03, -0.02]),
                        "goal_rotm": self.hammer_rotm @ R.from_euler('y', -np.pi/2).as_matrix() @ R.from_euler('z', -np.pi/2).as_matrix(),
                        "gripper": 1}
            self.waypoints_robot0.append(waypoint)

            #wp2
            waypoint = {"goal_pos": np.array([pass_x, pass_y, pass_z]),
                        "goal_rotm": self.hammer_rotm @ R.from_euler('y', -np.pi/2).as_matrix() @ R.from_euler('z', -np.pi/2).as_matrix(),
                        "gripper": 1}
            self.waypoints_robot0.append(waypoint)

            #wp3
            waypoint = {"goal_pos": np.array([pass_x + giver_x_offset, pass_y, pass_z]),
                        "goal_rotm": self.hammer_rotm @ R.from_euler('y', -np.pi/2).as_matrix() @ R.from_euler('z', -np.pi).as_matrix(),
                        "gripper": 1}
            self.waypoints_robot0.append(waypoint)

            #wp4
            waypoint = {"goal_pos": np.array([pass_x + giver_x_offset, pass_y, pass_z]),
                        "goal_rotm": self.hammer_rotm @ R.from_euler('y', -np.pi/2).as_matrix() @ R.from_euler('z', -np.pi).as_matrix(),
                        "gripper": 1}
            self.waypoints_robot0.append(waypoint)

            #wp5
            waypoint = {"goal_pos": np.array([pass_x + giver_x_offset, pass_y, pass_z]),
                        "goal_rotm": self.hammer_rotm @ R.from_euler('y', -np.pi/2).as_matrix() @ R.from_euler('z', -np.pi).as_matrix(),
                        "gripper": -1}
            self.waypoints_robot0.append(waypoint)

            #wp6
            waypoint = {"goal_pos": np.array([pass_x, pass_y, pass_z]),
                        "goal_rotm": self.hammer_rotm @ R.from_euler('y', -np.pi/2).as_matrix() @ R.from_euler('z', -np.pi).as_matrix(),
                        "gripper": -1}
            self.waypoints_robot0.append(waypoint)


            """
            Robot 1 Waypoints
            """
            #wp0
            waypoint = {"goal_pos": np.array([pass_x, pass_y, pass_z]),
                    "goal_rotm": self.R_be_home,
                    "gripper": -1}
            self.waypoints_robot1.append(waypoint)

            #wp1
            waypoint = {"goal_pos": np.array([pass_x, pass_y, pass_z]),
                    "goal_rotm": self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix(),
                    "gripper": -1}
            self.waypoints_robot1.append(waypoint)

            #wp2
            waypoint = {"goal_pos": np.array([pass_x, pass_y, pass_z]),
                    "goal_rotm": self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix(),
                    "gripper": -1}
            self.waypoints_robot1.append(waypoint)

            #wp3
            waypoint = {"goal_pos": np.array([pass_x + receiver_x_offset, pass_y, pass_z]),
                    "goal_rotm": self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', -np.pi/2).as_matrix(),
                    "gripper": -1}
            self.waypoints_robot1.append(waypoint)

            #wp4
            waypoint = {"goal_pos": np.array([pass_x + receiver_x_offset, pass_y, pass_z]),
                    "goal_rotm": self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', -np.pi/2).as_matrix(),
                    "gripper": 1}
            self.waypoints_robot1.append(waypoint)

            #wp5
            waypoint = {"goal_pos": np.array([pass_x + receiver_x_offset, pass_y, pass_z]),
                    "goal_rotm": self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', -np.pi/2).as_matrix(),
                    "gripper": 1}
            self.waypoints_robot1.append(waypoint)

            #wp6
            waypoint = {"goal_pos": np.array([pass_x, pass_y, pass_z]),
                    "goal_rotm": self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', -np.pi/2).as_matrix(),
                    "gripper": 1}
            self.waypoints_robot1.append(waypoint)

        elif mode == 2:
            """
            Robot 0 Waypoints
            """
            #wp0
            waypoint = {"goal_pos": np.array([pass_x, 0, pass_z]),
                    "goal_rotm": self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix(),
                    "gripper": -1}
            self.waypoints_robot0.append(waypoint)

            #wp1
            waypoint = {"goal_pos": np.array([pass_x, pass_y, pass_z]),
                    "goal_rotm": self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix(),
                    "gripper": -1}
            self.waypoints_robot0.append(waypoint)

            #wp2
            waypoint = {"goal_pos": np.array([pass_x, pass_y, pass_z]),
                    "goal_rotm": self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix(),
                    "gripper": -1}
            self.waypoints_robot0.append(waypoint)

            #wp3
            waypoint = {"goal_pos": np.array([pass_x + receiver_x_offset, pass_y, pass_z]),
                    "goal_rotm": self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', -np.pi/2).as_matrix(),
                    "gripper": -1}
            self.waypoints_robot0.append(waypoint)

            #wp4
            waypoint = {"goal_pos": np.array([pass_x + receiver_x_offset, pass_y, pass_z]),
                    "goal_rotm": self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', -np.pi/2).as_matrix(),
                    "gripper": 1}
            self.waypoints_robot0.append(waypoint)

            #wp5
            waypoint = {"goal_pos": np.array([pass_x + receiver_x_offset, pass_y, pass_z]),
                    "goal_rotm": self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', -np.pi/2).as_matrix(),
                    "gripper": 1}
            self.waypoints_robot0.append(waypoint)

            #wp6
            waypoint = {"goal_pos": np.array([pass_x, pass_y, pass_z]),
                    "goal_rotm": self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', -np.pi/2).as_matrix(),
                    "gripper": 1}
            self.waypoints_robot0.append(waypoint)

            """
            Robot 1 Waypoints
            """
            self.pickup_pos = self.hammer_pos1 + np.array([0, self.handle_length/2 - 0.02, 0.02])
            #wp0
            waypoint = {"goal_pos": self.hammer_pos1 + np.array([0, self.handle_length/2 - 0.02, 0.02]),
                        "goal_rotm": self.hammer_rotm @ R.from_euler('y', -np.pi/2).as_matrix() @ R.from_euler('z', -np.pi/2).as_matrix(),
                        "gripper": -1}
            self.waypoints_robot1.append(waypoint)

            #wp1
            waypoint = {"goal_pos": self.hammer_pos1 + np.array([0, self.handle_length/2 - 0.02, -0.02]),
                        "goal_rotm": self.hammer_rotm @ R.from_euler('y', -np.pi/2).as_matrix() @ R.from_euler('z', -np.pi/2).as_matrix(),
                        "gripper": 1}
            self.waypoints_robot1.append(waypoint)

            #wp2
            waypoint = {"goal_pos": np.array([pass_x, pass_y, pass_z]),
                        "goal_rotm": self.hammer_rotm @ R.from_euler('y', -np.pi/2).as_matrix() @ R.from_euler('z', -np.pi/2).as_matrix(),
                        "gripper": 1}
            self.waypoints_robot1.append(waypoint)

            #wp3
            waypoint = {"goal_pos": np.array([pass_x + giver_x_offset, pass_y, pass_z]),
                        "goal_rotm": self.hammer_rotm @ R.from_euler('y', -np.pi/2).as_matrix(),
                        "gripper": 1}
            self.waypoints_robot1.append(waypoint)

            #wp4
            waypoint = {"goal_pos": np.array([pass_x + giver_x_offset, pass_y, pass_z]),
                        "goal_rotm": self.hammer_rotm @ R.from_euler('y', -np.pi/2).as_matrix(),
                        "gripper": 1}
            self.waypoints_robot1.append(waypoint)

            #wp5
            waypoint = {"goal_pos": np.array([pass_x + giver_x_offset, pass_y, pass_z]),
                        "goal_rotm": self.hammer_rotm @ R.from_euler('y', -np.pi/2).as_matrix(),
                        "gripper": -1}
            self.waypoints_robot1.append(waypoint)

            #wp6
            waypoint = {"goal_pos": np.array([pass_x, pass_y, pass_z]),
                        "goal_rotm": self.hammer_rotm @ R.from_euler('y', -np.pi/2).as_matrix(),
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

    
    def get_demo(self, seed, mode = 1):
        """
        Main file to get the demonstration data
        """
        obs = self.reset(seed, mode)

        max_step_move = int(3 * self.control_freq) # 15 seconds
        max_step_grip = int(1. * self.control_freq) # 0.5 seconds

        # stage 0: robot0 move to wp0 robot1 moves to wp0
        robot0_arrived = False
        robot1_arrived = False
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
                action1 = self.convert_action_robot(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, self.waypoints_robot1[0]["gripper"])
                robot1_arrived = self.check_arrived(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, threshold = 0.05)

            action = np.hstack([action0, action1])
            obs, reward, done, info = self.env.step(action)
            self.rollout["observations"].append(self.process_obs(obs))
            self.rollout["actions"].append(action)

            if self.render:
                self.env.render()

            # if robot0_arrived and robot1_arrived:
            #     break
        
        # stage 1: robot0 move to wp1 robot1 moves to wp1
        robot0_arrived = False
        robot1_arrived = False
        for i in range(max_step_move):
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

            # if robot0_arrived and robot1_arrived:
            #     break
        
        # stage 2: robot0 move to wp2 robot1 moves to wp2
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

            # if robot0_arrived and robot1_arrived:
            #     break

        # stage 3: robot0 move to wp3 robot1 moves to wp3
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

            # if robot0_arrived and robot1_arrived:
            #     break

        # stage 4: robot0 move to wp4 robot1 moves to wp4
        robot0_arrived = False
        robot1_arrived = False
        for i in range(max_step_grip):
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

        # stage 5: robot0 move to wp5 robot1 moves to wp5
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

        # stage 6: robot0 move to wp3 robot1 moves to wp3
        robot0_arrived = False
        robot1_arrived = False
        for i in range(max_step_move):
            robot0_pos, robot0_rotm, robot1_pos, robot1_rotm = self.get_poses(obs)

            if not robot0_arrived:
                goal_pos0 = self.waypoints_robot0[6]["goal_pos"]
                goal_rotm0 = self.waypoints_robot0[6]["goal_rotm"]
                action0 = self.convert_action_robot(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, self.waypoints_robot0[6]["gripper"])
                robot0_arrived = self.check_arrived(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, threshold = 0.05)

            if not robot1_arrived:
                goal_pos1 = self.waypoints_robot1[6]["goal_pos"]
                goal_rotm1 = self.waypoints_robot1[6]["goal_rotm"]
                action1 = self.convert_action_robot(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, self.waypoints_robot1[6]["gripper"])
                robot1_arrived = self.check_arrived(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, threshold = 0.05)

            action = np.hstack([action0, action1])
            obs, reward, done, info = self.env.step(action)
            self.rollout["observations"].append(self.process_obs(obs))
            self.rollout["actions"].append(action)

            if self.render:
                self.env.render()

            # if robot0_arrived and robot1_arrived:
            #     break

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
    controller_config = load_composite_controller_config(robot="Kinova3", controller="kinova.json")

    env = TwoArmHandoverRole(
    robots=["Kinova3", "Kinova3"],
    gripper_types="default",
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    prehensile=True,
    render_camera=None,
    )

    player = PolicyPlayer(env, render = False)
    # rollout = player.get_demo(seed = 11111, mode = 2)
    for i in range(200):   
        rollout = player.get_demo(seed = i*10, mode = 1)
        rollout['hammer_pos'] = player.pickup_pos
        with open("rollouts_pickup_pos/rollout_seed%s_mode1.pkl" % (i*10), "wb") as f:
            pkl.dump(rollout, f)
        rollout = player.get_demo(seed = i*10, mode = 2)
        rollout['hammer_pos'] = player.pickup_pos
        with open("rollouts_pickup_pos/rollout_seed%s_mode2.pkl" % (i*10), "wb") as f:
            pkl.dump(rollout, f)
