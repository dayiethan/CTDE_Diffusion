from collections import OrderedDict

import numpy as np

from scipy.spatial.transform import Rotation

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import HammerObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler


class TwoArmHandoverRole(TwoArmEnv):
    """
    This class corresponds to the handover task for two robot arms.

    Args:
        robots (str or list of str): Specification for specific robot(s)
            Note: Must be either 2 robots or 1 bimanual robot!

        env_configuration (str): Specifies how to position the robots within the environment if two robots inputted. Can be either:

            :`'parallel'`: Sets up the two robots next to each other on the -x side of the table
            :`'opposed'`: Sets up the two robots opposed from each others on the opposite +/-y sides of the table.

        Note that "default" "opposed" if two robots are used.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        prehensile (bool): If true, handover object starts on the table. Else, the object starts in Arm0's gripper

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        ValueError: [Invalid number of robots specified]
        ValueError: [Invalid env configuration]
        ValueError: [Invalid robots for specified env configuration]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        prehensile=True,
        table_full_size=(0.8, 1.2, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
    ):
        # Task settings
        self.prehensile = prehensile

        # settings for table top
        self.table_full_size = table_full_size
        self.table_true_size = list(table_full_size)
        self.table_true_size[1] *= 0.25  # true size will only be partially wide
        self.table_friction = table_friction
        self.table_offset = [0, self.table_full_size[1] * (-3 / 8), 0.8]

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.height_threshold = 0.1  # threshold above the table surface which the hammer is considered lifted

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.0 is provided when only Arm 1 is gripping the handle and has the handle
              lifted above a certain threshold

        Un-normalized max-wise components if using reward shaping:

            - Arm0 Reaching: (1) in [0, 0.25] proportional to the distance between Arm 0 and the handle
            - Arm0 Grasping: (2) in {0, 0.5}, nonzero if Arm 0 is gripping the hammer (any part).
            - Arm0 Lifting: (3) in {0, 1.0}, nonzero if Arm 0 lifts the handle from the table past a certain threshold
            - Arm0 Hovering: (4) in {0, [1.0, 1.25]}, nonzero only if Arm0 is actively lifting the hammer, and is
              proportional to the distance between the handle and Arm 1
              conditioned on the handle being lifted from the table and being grasped by Arm 0
            - Mutual Grasping: (5) in {0, 1.5}, nonzero if both Arm 0 and Arm 1 are gripping the hammer (Arm 1 must be
              gripping the handle) while lifted above the table
            - Handover: (6) in {0, 2.0}, nonzero when only Arm 1 is gripping the handle and has the handle
              lifted above the table

        Note that the final reward is normalized and scaled by reward_scale / 2.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        # Initialize reward
        reward = 0

        # use a shaping reward if specified
        if self.reward_shaping:
            # Grab relevant parameters
            arm0_grasp_any, arm1_grasp_handle, hammer_height, table_height = self._get_task_info()
            # First, we'll consider the cases if the hammer is lifted above the threshold (step 3 - 6)
            if hammer_height - table_height > self.height_threshold:
                # Split cases depending on whether arm1 is currently grasping the handle or not
                if arm1_grasp_handle:
                    # Check if arm0 is grasping
                    if arm0_grasp_any:
                        # This is step 5
                        reward = 1.5
                    else:
                        # This is step 6 (completed task!)
                        reward = 2.0
                # This is the case where only arm0 is grasping (step 2-3)
                else:
                    reward = 1.0
                    # Add in up to 0.25 based on distance between handle and arm1
                    dist = np.linalg.norm(self._gripper_1_to_handle)
                    reaching_reward = 0.25 * (1 - np.tanh(1.0 * dist))
                    reward += reaching_reward
            # Else, the hammer is still on the ground ):
            else:
                # Split cases depending on whether arm0 is currently grasping the handle or not
                if arm0_grasp_any:
                    # This is step 2
                    reward = 0.5
                else:
                    # This is step 1, we want to encourage arm0 to reach for the handle
                    dist = np.linalg.norm(self._gripper_0_to_handle)
                    reaching_reward = 0.25 * (1 - np.tanh(1.0 * dist))
                    reward = reaching_reward

        # Else this is the sparse reward setting
        else:
            # Provide reward if only Arm 1 is grasping the hammer and the handle lifted above the pre-defined threshold
            if self._check_success():
                reward = 2.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.0

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose(s) accordingly
        if self.env_configuration == "single-robot":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation, offset in zip(self.robots, (np.pi / 2, -np.pi / 2), (-0.15, 0.15)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    xpos += np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:  # "parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                for robot, offset in zip(self.robots, (-0.6, 0.6)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_true_size, table_friction=self.table_friction, table_offset=self.table_offset
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify camera
        agent0_cam_pos = [0.8894354364730311, -3.481824231498976e-08, 1.7383813133506494]
        agent0_cam_quat = [0.6530981063842773, 0.2710406184196472, 0.27104079723358154, 0.6530979871749878]

        agent1_cam_pos = [-agent0_cam_pos[0], -agent0_cam_pos[1], agent0_cam_pos[2]]
        # rotm = Rotation.from_quat(agent0_cam_quat, scalar_first=True).as_matrix()
        quat0 = np.array(agent0_cam_quat)
        quat_scalar_last0 = np.array([quat0[1], quat0[2], quat0[3], quat0[0]])
        rotm = Rotation.from_quat(quat_scalar_last0).as_matrix()
        rot_z = Rotation.from_euler('z', 180, degrees=True).as_matrix()
        rotm_rotated = rot_z @ rotm
        # agent1_cam_quat = Rotation.from_matrix(rotm_rotated).as_quat(scalar_first = True)
        agent1_cam_quat_scalar_last = Rotation.from_matrix(rotm_rotated).as_quat()
        agent1_cam_quat = np.array([
            agent1_cam_quat_scalar_last[3],
            agent1_cam_quat_scalar_last[0],
            agent1_cam_quat_scalar_last[1],
            agent1_cam_quat_scalar_last[2]
        ])


        mujoco_arena.set_camera(
            camera_name="robot0_leftview",
            pos=agent0_cam_pos,
            quat=agent0_cam_quat,
        )
        mujoco_arena.set_camera(
            camera_name="robot1_leftview",
            pos= agent1_cam_pos,
            quat=agent1_cam_quat,
        )
        mujoco_arena.set_camera(
            camera_name="birdview",
            pos=[0, 0, 2.2],
            quat=[0.7071, 0, 0, 0.7071],
        )

        # initialize objects of interest
        self.hammer = HammerObject(name="hammer")

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.hammer)
        else:
            # Set rotation about y-axis if hammer starts on table else rotate about z if it starts in gripper
            rotation_axis = "z" if self.prehensile else "y"

            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.hammer,
                x_range=[-0.1, 0.1],
                y_range=[-0.05, 0.05],
                rotation=np.pi/2,
                rotation_axis=rotation_axis,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.hammer,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Hammer object references from this env
        self.hammer_body_id = self.sim.model.body_name2id(self.hammer.root_body)
        self.hammer_handle_geom_id = self.sim.model.geom_name2id(self.hammer.handle_geoms[0])

        # General env references
        self.table_top_id = self.sim.model.site_name2id("table_top")

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            modality = "object"

            # position and rotation of hammer
            @sensor(modality=modality)
            def hammer_pos(obs_cache):
                return np.array(self._hammer_pos)

            @sensor(modality=modality)
            def hammer_quat(obs_cache):
                return np.array(self._hammer_quat)

            @sensor(modality=modality)
            def handle_xpos(obs_cache):
                return np.array(self._handle_xpos)

            sensors = [hammer_pos, hammer_quat, handle_xpos]
            names = [s.__name__ for s in sensors]
            arm_sensor_fns = []
            if self.env_configuration == "single-robot":
                # If single-robot, we only have one robot. gripper 0 is always right and gripper 1 is always left
                pf0 = self.robots[0].robot_model.naming_prefix + "right_"
                pf1 = self.robots[0].robot_model.naming_prefix + "left_"
                prefixes = [pf0, pf1]
                arm_sensor_fns = [
                    self._get_obj_eef_sensor(full_pf, f"handle_xpos", f"gripper{i}_to_handle", modality)
                    for i, full_pf in enumerate(prefixes)
                ]
            else:
                # If not single-robot, we have two robots. gripper 0 is always the first robot's gripper and
                # gripper 1 is always the second robot's gripper. However, must account for the fact that
                # each robot may have multiple arms/grippers
                robot_arm_prefixes = [self._get_arm_prefixes(robot, include_robot_name=False) for robot in self.robots]
                robot_full_prefixes = [self._get_arm_prefixes(robot, include_robot_name=True) for robot in self.robots]
                for i, (arm_prefixes, full_prefixes) in enumerate(zip(robot_arm_prefixes, robot_full_prefixes)):
                    arm_sensor_fns += [
                        self._get_obj_eef_sensor(full_pf, f"handle_xpos", f"{arm_pf}gripper{i}_to_handle", modality)
                        for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
                    ]

            sensors += arm_sensor_fns
            names += [s.__name__ for s in arm_sensor_fns]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                # If prehensile, set the object normally
                if self.prehensile:
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                # Else, set the object in the hand of the robot and loop a few steps to guarantee the robot is grasping
                #   the object initially
                else:
                    eef_rot_quat = T.mat2quat(T.euler2mat([np.pi - T.mat2euler(self._eef0_xmat)[2], 0, 0]))
                    obj_quat = T.quat_multiply(obj_quat, eef_rot_quat)
                    for j in range(100):
                        # Set object in hand always placing it in the right gripper
                        obj_target_position = self._eef0_xpos + np.array([0, 0, 0.04])
                        self.sim.data.set_joint_qpos(
                            obj.joints[0], np.concatenate([obj_target_position, np.array(obj_quat)])
                        )
                        # Close gripper (action = 1) and prevent arm from moving
                        if self.env_configuration == "single-robot":
                            # Execute no-op action with gravity compensation
                            torques = np.concatenate(
                                [
                                    self.robots[0].part_controllers["right"].torque_compensation,
                                    self.robots[0].part_controllers["left"].torque_compensation,
                                ]
                            )
                            self.sim.data.ctrl[self.robots[0]._ref_arm_joint_actuator_indexes] = torques
                        else:
                            # for robot in self.robots:
                            #     print(f"Robot: {robot}")
                            #     print(f"arm: {robot.arms}")
                            #     print(f"controller: {robot.part_controllers}")
                            robot_noops = [
                                np.concatenate([robot.part_controllers[arm].torque_compensation for arm in robot.arms])
                                for robot in self.robots
                            ]

                            # Execute no-op action with gravity compensation
                            self.sim.data.ctrl[self.robots[0]._ref_arm_joint_actuator_indexes] = robot_noops[0]
                            self.sim.data.ctrl[self.robots[1]._ref_arm_joint_actuator_indexes] = robot_noops[1]
                        # Execute gripper action
                        gripper_ac = [1] * self.robots[0].gripper["right"].dof
                        gripper_ac = self.robots[0].gripper["right"].format_action(gripper_ac)
                        self.robots[0].part_controllers["right_gripper"].set_goal(gripper_ac)
                        # Take forward step
                        self.sim.step()

    def _get_task_info(self):
        """
        Helper function that grabs the current relevant locations of objects of interest within the environment

        Returns:
            4-tuple:

                - (bool) True if Arm0 is grasping any part of the hammer
                - (bool) True if Arm1 is grasping the hammer handle
                - (float) Height of the hammer body
                - (float) Height of the table surface
        """
        # Get height of hammer and table and define height threshold
        hammer_angle_offset = (self.hammer.handle_length / 2 + 2 * self.hammer.head_halfsize) * np.sin(
            self._hammer_angle
        )
        hammer_height = (
            self.sim.data.geom_xpos[self.hammer_handle_geom_id][2] - self.hammer.top_offset[2] - hammer_angle_offset
        )
        table_height = self.sim.data.site_xpos[self.table_top_id][2]

        # Check if any Arm's gripper is grasping the hammer handle
        (g0, g1) = (
            (self.robots[0].gripper["right"], self.robots[0].gripper["left"])
            if self.env_configuration == "single-robot"
            else (self.robots[0].gripper, self.robots[1].gripper)
        )
        arm0_grasp_any = self._check_grasp(gripper=g0, object_geoms=self.hammer)
        arm1_grasp_handle = self._check_grasp(gripper=g1, object_geoms=self.hammer.handle_geoms)

        # Return all relevant values
        return arm0_grasp_any, arm1_grasp_handle, hammer_height, table_height

    def _check_success(self):
        """
        Check if hammer is successfully handed off

        Returns:
            bool: True if handover has been completed
        """
        # Grab relevant params
        arm0_grasp_any, arm1_grasp_handle, hammer_height, table_height = self._get_task_info()
        return (
            True
            if arm1_grasp_handle and not arm0_grasp_any and hammer_height - table_height > self.height_threshold
            else False
        )

    @property
    def _handle_xpos(self):
        """
        Grab the position of the hammer handle.

        Returns:
            np.array: (x,y,z) position of handle
        """
        return self.sim.data.geom_xpos[self.hammer_handle_geom_id]

    @property
    def _hammer_pos(self):
        """
        Grab the position of the hammer body.

        Returns:
            np.array: (x,y,z) position of body
        """
        return np.array(self.sim.data.body_xpos[self.hammer_body_id])

    @property
    def _hammer_quat(self):
        """
        Grab the orientation of the hammer body.

        Returns:
            np.array: (x,y,z,w) quaternion of the hammer body
        """
        return T.convert_quat(self.sim.data.body_xquat[self.hammer_body_id], to="xyzw")

    @property
    def _hammer_angle(self):
        """
        Calculate the angle of hammer with the ground, relative to it resting horizontally

        Returns:
            float: angle in radians
        """
        mat = T.quat2mat(self._hammer_quat)
        z_unit = [0, 0, 1]
        z_rotated = np.matmul(mat, z_unit)
        return np.pi / 2 - np.arccos(np.dot(z_unit, z_rotated))

    @property
    def _gripper_0_to_handle(self):
        """
        Calculate vector from gripper0 to the hammer handle.

        Returns:
            np.array: (dx,dy,dz) distance vector between handle and EEF0
        """
        return self._gripper0_to_target(self.hammer.handle_geoms[0], target_type="geom")

    @property
    def _gripper_1_to_handle(self):
        """
        Calculate vector from gripper1 to the hammer handle.

        Returns:
            np.array: (dx,dy,dz) distance vector between handle and EEF1
        """
        return self._gripper1_to_target(self.hammer.handle_geoms[0], target_type="geom")
