import time

import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle as pkl

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

from two_arm_lift_env import TwoArmLiftRole

from scipy.spatial.transform import Rotation as R


class PolicyPlayer:
    def __init__ (self, env, render= True):
        
        
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

        self.setup_waypoints(mode = mode)

        self.rollout = []

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
            robot0_z_pass = 0.4
            robot1_x_pass = robot1_x_init
            robot1_y_pass = robot1_y_init
            robot1_z_pass = 0.4
        elif mode == 2:
            rotm0 = self.R_be_home @ R.from_euler('z', -np.pi/2).as_matrix()
            rotm0 = self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', np.pi/2).as_matrix()
            rotm1 = self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', np.pi/2).as_matrix()
            robot0_x_pass = robot0_x_init + 0.15
            robot0_y_pass = robot0_y_init
            robot0_z_pass = 0.4
            robot1_x_pass = robot1_x_init - 0.15
            robot1_y_pass = robot1_y_init
            robot1_z_pass = 0.4
        elif mode == 3:
            rotm0 = self.R_be_home @ R.from_euler('z', -np.pi/2).as_matrix()
            rotm0 = self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', np.pi/2).as_matrix()
            rotm1 = self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', np.pi/2).as_matrix()
            robot0_x_pass = robot0_x_init - 0.15
            robot0_y_pass = robot0_y_init
            robot0_z_pass = 0.4
            robot1_x_pass = robot1_x_init + 0.15
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

        

    def convert_action_robot(self, robot_goal_pos, robot_goal_rotm, robot_gripper):
        action = np.zeros(int(self.n_action/2))
        action[0:3] = robot_goal_pos
        action[3:6] = R.from_matrix(robot_goal_rotm).as_rotvec()
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

    
    def get_demo(self, seed, mode):
        """
        Main file to get the demonstration data
        """
        obs = self.reset(seed, mode)

        max_step_move = int(15 * self.control_freq) # 15 seconds
        max_step_grip = int(0.9 * self.control_freq)

        robot0_arrived = False
        robot1_arrived = False

        # stage 1: robot0 move to wp0 robot1 moves to wp0
        for i in range(max_step_move):
            robot0_pos, robot0_rotm, robot1_pos, robot1_rotm = self.get_poses(obs)

            if not robot0_arrived:
                goal_pos0 = self.waypoints_robot0[0]["goal_pos"]
                goal_rotm0 = self.waypoints_robot0[0]["goal_rotm"]
                action0 = self.convert_action_robot(goal_pos0, goal_rotm0, self.waypoints_robot0[0]["gripper"])
                robot0_arrived = self.check_arrived(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, threshold = 0.05)

            if not robot1_arrived:
                goal_pos1 = self.waypoints_robot1[0]["goal_pos"]
                goal_rotm1 = self.waypoints_robot1[0]["goal_rotm"]
                action1 = self.convert_action_robot(goal_pos1, goal_rotm1, self.waypoints_robot1[0]["gripper"])
                robot1_arrived = self.check_arrived(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, threshold = 0.05)


            action = np.hstack([action0, action1])
            obs, reward, done, info = self.env.step(action)
            self.rollout.append(obs)
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
                action0 = self.convert_action_robot(goal_pos0, goal_rotm0, self.waypoints_robot0[1]["gripper"])
                robot0_arrived = self.check_arrived(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, threshold = 0.05)

            if not robot1_arrived:
                goal_pos1 = self.waypoints_robot1[1]["goal_pos"]
                goal_rotm1 = self.waypoints_robot1[1]["goal_rotm"]
                action1 = self.convert_action_robot(goal_pos1, goal_rotm1, self.waypoints_robot1[1]["gripper"])
                robot1_arrived = self.check_arrived(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, threshold = 0.05)

            action = np.hstack([action0, action1])
            obs, reward, done, info = self.env.step(action)
            self.rollout.append(obs)

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
                action0 = self.convert_action_robot(goal_pos0, goal_rotm0, self.waypoints_robot0[2]["gripper"])
                robot0_arrived = self.check_arrived(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, threshold = 0.05)

            if not robot1_arrived:
                goal_pos1 = self.waypoints_robot1[2]["goal_pos"]
                goal_rotm1 = self.waypoints_robot1[2]["goal_rotm"]
                action1 = self.convert_action_robot(goal_pos1, goal_rotm1, self.waypoints_robot1[2]["gripper"])
                robot1_arrived = self.check_arrived(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, threshold = 0.05)


            action = np.hstack([action0, action1])
            obs, reward, done, info = self.env.step(action)
            self.rollout.append(obs)
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
                action0 = self.convert_action_robot(goal_pos0, goal_rotm0, self.waypoints_robot0[3]["gripper"])
                robot0_arrived = self.check_arrived(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, threshold = 0.05)

            if not robot1_arrived:
                goal_pos1 = self.waypoints_robot1[3]["goal_pos"]
                goal_rotm1 = self.waypoints_robot1[3]["goal_rotm"]
                action1 = self.convert_action_robot(goal_pos1, goal_rotm1, self.waypoints_robot1[3]["gripper"])
                robot1_arrived = self.check_arrived(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, threshold = 0.05)


            action = np.hstack([action0, action1])
            obs, reward, done, info = self.env.step(action)
            self.rollout.append(obs)
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
                action0 = self.convert_action_robot(goal_pos0, goal_rotm0, self.waypoints_robot0[4]["gripper"])
                robot0_arrived = self.check_arrived(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, threshold = 0.05)

            if not robot1_arrived:
                goal_pos1 = self.waypoints_robot1[4]["goal_pos"]
                goal_rotm1 = self.waypoints_robot1[4]["goal_rotm"]
                action1 = self.convert_action_robot(goal_pos1, goal_rotm1, self.waypoints_robot1[4]["gripper"])
                robot1_arrived = self.check_arrived(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, threshold = 0.05)


            action = np.hstack([action0, action1])
            obs, reward, done, info = self.env.step(action)
            self.rollout.append(obs)
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
                action0 = self.convert_action_robot(goal_pos0, goal_rotm0, self.waypoints_robot0[5]["gripper"])
                robot0_arrived = self.check_arrived(robot0_pos, robot0_rotm, goal_pos0, goal_rotm0, threshold = 0.05)

            if not robot1_arrived:
                goal_pos1 = self.waypoints_robot1[5]["goal_pos"]
                goal_rotm1 = self.waypoints_robot1[5]["goal_rotm"]
                action1 = self.convert_action_robot(goal_pos1, goal_rotm1, self.waypoints_robot1[5]["gripper"])
                robot1_arrived = self.check_arrived(robot1_pos, robot1_rotm, goal_pos1, goal_rotm1, threshold = 0.05)

            action = np.hstack([action0, action1])
            obs, reward, done, info = self.env.step(action)
            self.rollout.append(obs)

            if self.render:
                self.env.render()
        
        return self.rollout


    def test(self):
        """
        TTesting the environment during the development of the scripted policy
        """
        goal_pos0 = np.array([0.6, 0, 0.2])
        goal_pos1 = np.array([0.6, 0, 0.2])
        goal_rotm0 = self.R_be_home
        goal_rotm1 = self.R_be_home @ R.from_euler('x', -np.pi/2).as_matrix() @ R.from_euler('z', np.pi/2).as_matrix()


        for i in range(self.max_steps):
            action0 = self.convert_action_robot0(goal_pos0, goal_rotm0, 1) #(7,)
            action1 = self.convert_action_robot1(goal_pos1, goal_rotm1, 1) #(7,)
            action = np.hstack([action0, action1]) #(14,)

            obs, reward, done, info = self.env.step(action)
            self.env.render()

            print(obs['robot1_gripper_qpos'])
            

    
        
if __name__ == "__main__":
    CAMERA_NAMES = ['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand', 'robot1_robotview', 'robot1_eye_in_hand']
    controller_config = load_composite_controller_config(robot="Kinova3", controller="kinova.json")

    env = TwoArmLiftRole(
    robots=["Kinova3", "Kinova3"],
    gripper_types="default",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    render_camera=None,
    camera_names=CAMERA_NAMES,
    camera_heights=256,
    camera_widths=256,
    camera_depths=True,
    camera_segmentations='instance',
    )

    player = PolicyPlayer(env)
    rollout = player.get_demo(seed = 100, mode = 2)
    with open("rollout.pkl", "wb") as f:
        pkl.dump(rollout, f)
