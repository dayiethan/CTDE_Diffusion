# Sets up the Two Arm Lift environment

from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import PotWithHandlesObject
from robosuite.models.objects.primitive.box import BoxObject as Box
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler


class TwoArmLiftRole(TwoArmEnv):
    """
    This class corresponds to the lifting task for two robot arms.

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
        table_full_size=(1.2, 0.6, 0.05),
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
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

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

            - a discrete reward of 3.0 is provided if the pot is lifted and is parallel within 30 deg to the table

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 0.5], per-arm component that is proportional to the distance between each arm and its
              respective pot handle, and exactly 0.5 when grasping the handle
              - Note that the agent only gets the lifting reward when flipping no more than 30 degrees.
            - Grasping: in {0, 0.25}, binary per-arm component awarded if the gripper is grasping its correct handle
            - Lifting: in [0, 1.5], proportional to the pot's height above the table, and capped at a certain threshold

        Note that the final reward is normalized and scaled by reward_scale / 3.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0

        # check if the pot is tilted more than 30 degrees
        mat = T.quat2mat(self._pot_quat)
        z_unit = [0, 0, 1]
        z_rotated = np.matmul(mat, z_unit)
        cos_z = np.dot(z_unit, z_rotated)
        cos_30 = np.cos(np.pi / 6)
        direction_coef = 1 if cos_z >= cos_30 else 0

        # check for goal completion: cube is higher than the table top above a margin
        if self._check_success():
            reward = 3.0 * direction_coef

        # use a shaping reward
        elif self.reward_shaping:
            # lifting reward
            pot_bottom_height = self.sim.data.site_xpos[self.pot_center_id][2] - self.pot.top_offset[2]
            table_height = self.sim.data.site_xpos[self.table_top_id][2]
            elevation = pot_bottom_height - table_height
            r_lift = min(max(elevation - 0.05, 0), 0.15)
            reward += 10.0 * direction_coef * r_lift

            _gripper0_to_handle0 = self._gripper0_to_handle0
            _gripper1_to_handle1 = self._gripper1_to_handle1

            # gh stands for gripper-handle
            # When grippers are far away, tell them to be closer

            # Get contacts
            (g0, g1) = (
                (self.robots[0].gripper["right"], self.robots[0].gripper["left"])
                if self.env_configuration == "single-robot"
                else (self.robots[0].gripper, self.robots[1].gripper)
            )

            _g0h_dist = np.linalg.norm(_gripper0_to_handle0)
            _g1h_dist = np.linalg.norm(_gripper1_to_handle1)

            # Grasping reward
            if self._check_grasp(gripper=g0, object_geoms=self.pot.handle0_geoms):
                reward += 0.25
            # Reaching reward
            reward += 0.5 * (1 - np.tanh(10.0 * _g0h_dist))

            # Grasping reward
            if self._check_grasp(gripper=g1, object_geoms=self.pot.handle1_geoms):
                reward += 0.25
            # Reaching reward
            reward += 0.5 * (1 - np.tanh(10.0 * _g1h_dist))

        if self.reward_scale is not None:
            reward *= self.reward_scale / 3.0

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
                for robot, rotation in zip(self.robots, (np.pi / 2, -np.pi / 2)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:  # "parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                for robot, offset in zip(self.robots, (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        self.pot = PotWithHandlesObject(name="pot")
        self.obs = Box(name="obs", size=[0.07, 0.07, 0.175], rgba=[1, 0, 0, 1])

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.pot)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.pot,
                x_range=[-0.05, 0.05],
                y_range=[-0.05, 0.05],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset + np.array([0.3, 0., 0.]),
                rotation=np.pi
                # rotation=(np.pi + -np.pi / 3, np.pi + np.pi / 3),
            )
        
        # Create a separate placement initializer for the box
        self.box_placement_initializer = UniformRandomSampler(
            name="BoxSampler",
            mujoco_objects=self.obs,
            # Set x_range and y_range to [0, 0] so it always gets placed at the reference position.
            x_range=[0.0, 0.0],
            y_range=[0.0, 0.0],
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset + np.array([0., 0., .2]),
            rotation=(0, 0)
        )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.pot, self.obs],
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.pot_body_id = self.sim.model.body_name2id(self.pot.root_body)
        self.handle0_site_id = self.sim.model.site_name2id(self.pot.important_sites["handle0"])
        self.handle1_site_id = self.sim.model.site_name2id(self.pot.important_sites["handle1"])
        self.table_top_id = self.sim.model.site_name2id("table_top")
        self.pot_center_id = self.sim.model.site_name2id(self.pot.important_sites["center"])

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

            # position and rotation of object

            @sensor(modality=modality)
            def pot_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.pot_body_id])

            @sensor(modality=modality)
            def pot_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.pot_body_id], to="xyzw")

            @sensor(modality=modality)
            def handle0_xpos(obs_cache):
                return np.array(self._handle0_xpos)

            @sensor(modality=modality)
            def handle1_xpos(obs_cache):
                return np.array(self._handle1_xpos)

            sensors = [pot_pos, pot_quat, handle0_xpos, handle1_xpos]
            names = [s.__name__ for s in sensors]

            arm_sensor_fns = []
            if self.env_configuration == "single-robot":
                # If single-robot, we only have one robot. gripper 0 is always right and gripper 1 is always left
                pf0 = self.robots[0].robot_model.naming_prefix + "right_"
                pf1 = self.robots[0].robot_model.naming_prefix + "left_"
                prefixes = [pf0, pf1]
                arm_sensor_fns = [
                    self._get_obj_eef_sensor(full_pf, f"handle{i}_xpos", f"gripper{i}_to_handle{i}", modality)
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
                        self._get_obj_eef_sensor(
                            full_pf, f"handle{i}_xpos", f"{arm_pf}gripper{i}_to_handle{i}", modality
                        )
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
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

            # Now sample and set the placement for the box.
            box_placements = self.box_placement_initializer.sample()
            for obj_pos, obj_quat, obj in box_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                )


    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to each handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to each handle
        if vis_settings["grippers"]:
            handles = [self.pot.important_sites[f"handle{i}"] for i in range(2)]
            grippers = (
                [self.robots[0].gripper[arm] for arm in self.robots[0].arms]
                if self.env_configuration == "single-robot"
                else [robot.gripper for robot in self.robots]
            )
            for gripper, handle in zip(grippers, handles):
                self._visualize_gripper_to_target(gripper=gripper, target=handle, target_type="site")

    def _check_success(self):
        """
        Check if pot is successfully lifted

        Returns:
            bool: True if pot is lifted
        """
        pot_bottom_height = self.sim.data.site_xpos[self.pot_center_id][2] - self.pot.top_offset[2]
        table_height = self.sim.data.site_xpos[self.table_top_id][2]

        # cube is higher than the table top above a margin
        return pot_bottom_height > table_height + 0.10

    @property
    def _handle0_xpos(self):
        """
        Grab the position of the left (blue) hammer handle.

        Returns:
            np.array: (x,y,z) position of handle
        """
        return self.sim.data.site_xpos[self.handle0_site_id]

    @property
    def _handle1_xpos(self):
        """
        Grab the position of the right (green) hammer handle.

        Returns:
            np.array: (x,y,z) position of handle
        """
        return self.sim.data.site_xpos[self.handle1_site_id]

    @property
    def _pot_quat(self):
        """
        Grab the orientation of the pot body.

        Returns:
            np.array: (x,y,z,w) quaternion of the pot body
        """
        return T.convert_quat(self.sim.data.body_xquat[self.pot_body_id], to="xyzw")

    @property
    def _gripper0_to_handle0(self):
        """
        Calculate vector from gripper0 to the left pot handle.

        Returns:
            np.array: (dx,dy,dz) distance vector between handle and gripper0
        """
        return self._gripper0_to_target(self.pot.important_sites["handle0"], target_type="site")

    @property
    def _gripper1_to_handle1(self):
        """
        Calculate vector from gripper1 to the right pot handle.

        Returns:
            np.array: (dx,dy,dz) distance vector between handle and EEF0
        """
        return self._gripper1_to_target(self.pot.important_sites["handle1"], target_type="site")