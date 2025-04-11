import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import robosuite as suite
import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"

import copy
from utils.transform_utils import SE3_log_map, SE3_exp_map

class PolicyPlayer:
    def __init__ (self, env, render= True):
        '''
        Playing of scripted policy for the two-arm handover role environment
        env: TwoArmHandoverRole environment
        render: bool, whether to render the environment

        NOTE: The waypoints are hardcoded in the setup_waypoints function
        NOTE: Observations are modified to be the "local"
        For example, obs['robot0_eef_pos'] is the position of the robot0 end-effector in the robot0 base frame
        obs['robot0_eef_quat_site'] is the orientation of the robot0 end-effector in the robot0 base frame

        NOTE outputs only rollout, composed of 
        rollout["observations"] = [obs0, obs1, obs2, ...]
        obs0: dictionary with observation keys
        rollout["actions"] = [action0, action1, action2, ...] 
        action0: numpy array of shape (14,) with the action for robot0 and robot1

        Now the preprocessing on the observation is not needed.
        '''

        #TODO(JS) Update the Policy player so that the actions are in the P-control action towards the waypoints
        
        
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

        obs = self.reset()

        robot0_init_rotm_world = R.from_quat(obs['robot0_eef_quat_site'], scalar_first = False).as_matrix()
        robot1_init_rotm_world = R.from_quat(obs['robot1_eef_quat_site'], scalar_first = False).as_matrix()

        self.n_action = self.env.action_spec[0].shape[0]

    def reset(self, seed = 0):
        """
        Resetting environment. Re-initializing the waypoints too.
        """
        np.random.seed(seed)
        obs = self.env.reset()

        self.handle_length = self.env.hammer.handle_length
        self.hammer_headsize = 2*self.env.hammer.head_halfsize

        jnt_id_0 = self.env.sim.model.joint_name2id("gripper0_right_finger_joint") #gripper0_right_finger_joint, gripper0_right_right_outer_knuckle_joint
        self.qpos_index_0 = self.env.sim.model.jnt_qposadr[jnt_id_0]
        jnt_id_1 = self.env.sim.model.joint_name2id("gripper1_right_finger_joint") #gripper0_right_finger_joint, gripper0_right_right_outer_knuckle_joint
        self.qpos_index_1 = self.env.sim.model.jnt_qposadr[jnt_id_1]

        self.rollout = {}
        self.rollout["observations"] = []
        self.rollout["actions"] = []

        # self.gripper_action = [] # contains {"robot0_gripper": action0[-1], "robot0_gripper": action1[-1]}
        self.setup_waypoints()

        return obs

    def setup_waypoints(self):
        self.waypoints_robot0 = []
        self.waypoints_robot1 = []

        """
        Robot 0 Waypoints
        """

        if self.handle_length < 0.12:
            robot0_pos_xd = 0.5 + 0.5 * self.handle_length
        else:
            robot0_pos_xd = 0.5 + 0.5 * self.handle_length + np.random.uniform(-0.02, 0.02)
        pos_yd = np.random.uniform(-0.2, 0.2)
        pos_zd = np.random.uniform(0.15, 0.35)

        #wp0
        waypoint = {"goal_pos": np.array([0.4, 0, 0.2]),
                     "goal_rotm": self.R_be_home,
                     "gripper": 1}
        self.waypoints_robot0.append(waypoint)

        #wp1
        waypoint = {"goal_pos": np.array([robot0_pos_xd, pos_yd, pos_zd]),
                     "goal_rotm": self.R_be_home,
                     "gripper": 1}
        self.waypoints_robot0.append(waypoint)
        
        #wp2
        waypoint = {"goal_pos": np.array([robot0_pos_xd, pos_yd, pos_zd]),
                    "goal_rotm": self.R_be_home,
                    "gripper": -1}
        self.waypoints_robot0.append(waypoint)

        #wp3
        waypoint = {"goal_pos": np.array([0.4, 0, pos_zd]),
                    "goal_rotm": self.R_be_home,
                    "gripper": -1}
        self.waypoints_robot0.append(waypoint)

        """
        Robot 1 Waypoints
        """
        if self.handle_length < 0.12:
            robot1_pos_xd = 1.45 - robot0_pos_xd - self.handle_length * 0.5
        else:
            robot1_pos_xd = 1.47 - robot0_pos_xd - self.handle_length * 0.5 + np.random.uniform(-0.02, 0.00)
        #wp0
        waypoint = {"goal_pos": np.array([0.5, 0, 0.4]),
                    "goal_rotm": self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix(),
                    "gripper": -1}
        self.waypoints_robot1.append(waypoint)

        #wp1
        waypoint = {"goal_pos": np.array([0.6, -pos_yd, pos_zd]),
                    "goal_rotm": self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix(),
                    "gripper": -1}
        self.waypoints_robot1.append(waypoint)

        #wp2
        waypoint = {"goal_pos": np.array([robot1_pos_xd, -pos_yd, pos_zd]),
                    "goal_rotm": self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', np.pi/2).as_matrix(),
                    "gripper": -1}
        self.waypoints_robot1.append(waypoint)

        #wp3
        waypoint = {"goal_pos": np.array([robot1_pos_xd, -pos_yd, pos_zd]),
                    "goal_rotm": self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', np.pi/2).as_matrix(),
                    "gripper": 1}
        self.waypoints_robot1.append(waypoint)

        #wp4
        waypoint = {"goal_pos": np.array([0.4, 0, 0.2]),
                    "goal_rotm": self.R_be_home,
                    "gripper": 1}
        self.waypoints_robot1.append(waypoint)
        
    #alpha = 0.75 was working
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

    
    def get_demo(self, seed):
        """
        Main file to get the demonstration data
        """
        obs = self.reset(seed)
        print("hammer length:", self.env.hammer.handle_length)
        # print("distance between tables:", self.robot0_base_pos - self.robot1_base_pos)

        max_step_move = int(15 * self.control_freq) # 15 seconds
        max_step_grip = int(0.5 * self.control_freq)

        robot0_arrived = False
        robot1_arrived_0 = False
        robot1_arrived_1 = False

        # stage 1: robot0 move to wp0 robot1 moves to wp0 and wp1
        for i in range(max_step_move):
            robot0_pos, robot0_rotm, robot1_pos, robot1_rotm = self.get_poses(obs)

            if not robot0_arrived:
                goal_pos0 = self.waypoints_robot0[0]["goal_pos"]
                goal_rotm0 = self.waypoints_robot0[0]["goal_rotm"]
                action0 = self.convert_action_robot(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, self.waypoints_robot0[0]["gripper"])
                robot0_arrived = self.check_arrived(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, threshold = 0.1)

            if not robot1_arrived_0:
                goal_pos1 = self.waypoints_robot1[0]["goal_pos"]
                goal_rotm1 = self.waypoints_robot1[0]["goal_rotm"]
                action1 = self.convert_action_robot(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, self.waypoints_robot1[0]["gripper"])
                robot1_arrived_0 = self.check_arrived(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, threshold = 0.2)

            elif robot1_arrived_0 and not robot1_arrived_1:
                # print(" I am here")
                goal_pos1 = self.waypoints_robot1[1]["goal_pos"]
                goal_rotm1 = self.waypoints_robot1[1]["goal_rotm"]
                action1 = self.convert_action_robot(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, self.waypoints_robot1[1]["gripper"])
                robot1_arrived_1 = self.check_arrived(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, threshold = 0.01)

            action = np.hstack([action0, action1])
            obs, reward, done, info = self.env.step(action)

            self.rollout["observations"].append(self.process_obs(obs))
            self.rollout["actions"].append(action)
            
            if self.render:
                self.env.render()

            if robot0_arrived and robot1_arrived_1:
                break

        # stage 2: robot0 move to wp1 robot1 moves to wp2
        robot0_arrived = False
        robot1_arrived = False
        
        for i in range(max_step_move):
            robot0_pos, robot0_rotm, robot1_pos, robot1_rotm = self.get_poses(obs)

            if not robot0_arrived:
                goal_pos0 = self.waypoints_robot0[1]["goal_pos"]
                goal_rotm0 = self.waypoints_robot0[1]["goal_rotm"]
                action0 = self.convert_action_robot(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, self.waypoints_robot0[1]["gripper"])
                robot0_arrived = self.check_arrived(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, threshold = 0.01)

            if not robot1_arrived:
                goal_pos1 = self.waypoints_robot1[2]["goal_pos"]
                goal_rotm1 = self.waypoints_robot1[2]["goal_rotm"]
                action1 = self.convert_action_robot(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, self.waypoints_robot1[2]["gripper"])
                robot1_arrived = self.check_arrived(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, threshold = 0.01)

            action = np.hstack([action0, action1])
            obs, reward, done, info = self.env.step(action)
            self.rollout["observations"].append(self.process_obs(obs))
            self.rollout["actions"].append(action)
            
            if self.render:
                self.env.render()

            if robot0_arrived and robot1_arrived:
                break

        # stage 3: robot0 stay in wp1 (close gripper) robot1 moves to wp3 (close gripper)
        for i in range(max_step_grip):
            robot0_pos, robot0_rotm, robot1_pos, robot1_rotm = self.get_poses(obs)

            goal_pos0 = self.waypoints_robot0[1]["goal_pos"]
            goal_rotm0 = self.waypoints_robot0[1]["goal_rotm"]
            action0 = self.convert_action_robot(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, self.waypoints_robot0[1]["gripper"])

            goal_pos1 = self.waypoints_robot1[3]["goal_pos"]
            goal_rotm1 = self.waypoints_robot1[3]["goal_rotm"]
            action1 = self.convert_action_robot(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, self.waypoints_robot1[3]["gripper"])

            action = np.hstack([action0, action1])
            obs, reward, done, info = self.env.step(action)
            self.rollout["observations"].append(self.process_obs(obs))
            self.rollout["actions"].append(action)
            
            if self.render:
                self.env.render()

        # stage 4: robot0 move to wp2 (open gripper) robot1 stay in wp3 (close gripper)
        for i in range(max_step_grip):
            robot0_pos, robot0_rotm, robot1_pos, robot1_rotm = self.get_poses(obs)

            goal_pos0 = self.waypoints_robot0[2]["goal_pos"]
            goal_rotm0 = self.waypoints_robot0[2]["goal_rotm"]
            action0 = self.convert_action_robot(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, self.waypoints_robot0[2]["gripper"])

            goal_pos1 = self.waypoints_robot1[3]["goal_pos"]
            goal_rotm1 = self.waypoints_robot1[3]["goal_rotm"]
            action1 = self.convert_action_robot(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, self.waypoints_robot1[3]["gripper"])

            action = np.hstack([action0, action1])
            obs, reward, done, info = self.env.step(action)
            self.rollout["observations"].append(self.process_obs(obs))
            self.rollout["actions"].append(action)
            
            if self.render:
                self.env.render()

        # stage 5: robot0 move to wp3 robot1 move to wp4
        robot0_arrived = False
        robot1_arrived = False
        for i in range(max_step_move):
            robot0_pos, robot0_rotm, robot1_pos, robot1_rotm = self.get_poses(obs)

            if not robot0_arrived:
                goal_pos0 = self.waypoints_robot0[3]["goal_pos"]
                goal_rotm0 = self.waypoints_robot0[3]["goal_rotm"]
                action0 = self.convert_action_robot(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, self.waypoints_robot0[3]["gripper"])
                robot0_arrived = self.check_arrived(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, threshold = 0.1)

            if not robot1_arrived:
                goal_pos1 = self.waypoints_robot1[4]["goal_pos"]
                goal_rotm1 = self.waypoints_robot1[4]["goal_rotm"]
                action1 = self.convert_action_robot(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, self.waypoints_robot1[4]["gripper"])
                robot1_arrived = self.check_arrived(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, threshold = 0.1)

            action = np.hstack([action0, action1])
            obs, reward, done, info = self.env.step(action)
            self.rollout["observations"].append(self.process_obs(obs))
            self.rollout["actions"].append(action)
            
            if self.render:
                self.env.render()

            if robot0_arrived and robot1_arrived:
                break
        
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

    def test(self):
        """
        Testing the environment during the development of the scripted policy
        """
        goal_pos0 = np.array([0.6, 0, 0.2])
        goal_pos1 = np.array([0.6, 0, 0.2])
        goal_rotm0 = self.R_be_home
        goal_rotm1 = self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', np.pi/2).as_matrix()

        print("hammer length:", self.env.hammer.handle_length)
        print("hammer headsize:", 2*self.env.hammer.head_halfsize)


        for i in range(self.max_steps):
            action0 = self.convert_action_robot0(goal_pos0, goal_rotm0, 1) #(7,)
            action1 = self.convert_action_robot1(goal_pos1, goal_rotm1, 1) #(7,)
            action = np.hstack([action0, action1]) #(14,)

            obs, reward, done, info = self.env.step(action)
            self.env.render()

            print(obs['robot1_gripper_qpos'])
            