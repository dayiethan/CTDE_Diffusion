import argparse
import yaml
import timeit
import h5py
import os

import numpy as np
import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"
from robosuite.controllers import load_composite_controller_config

from custom_env.two_arm_handover_role import TwoArmHandoverRole
from data_collection.scripted_policy.policy_player import PolicyPlayer
from utils.transform_utils import quat_to_rot6d, rotvec_to_rot6d


def format_action_observation(rollout, task_config, env_config, split_robot=True):
    """
    Supports only splited robot observations.
    Observation is constructed as obs[2:end] and action is constructed as obs[0:end-2]
    """
    if split_robot:
        # Format action
        action_dict = {"robot0": {}, "robot1": {}}
        
        # Action pose
        action_dict['robot0']['/action/position'] = np.array([action[0:3] for action in rollout['actions']])
        action_dict['robot0']['/action/rotation_ortho6'] = np.array([rotvec_to_rot6d(action[3:6]) for action in rollout['actions']])
        action_dict['robot0']['/action/gripper'] = np.array([action[6] for action in rollout['actions']])
        action_dict['robot1']['/action/position'] = np.array([action[7:10] for action in rollout['actions']])
        action_dict['robot1']['/action/rotation_ortho6'] = np.array([rotvec_to_rot6d(action[10:13]) for action in rollout['actions']])
        action_dict['robot1']['/action/gripper'] = np.array([action[13] for action in rollout['actions']])

        # Format observation
        obs_dict = {"robot0": {}, "robot1": {}}
        
        # End-Effector's Cartesian pose 
        obs_dict['robot0']['/observations/eef/position'] = np.array([obs['robot0_eef_pos'] for obs in rollout['observations']])
        obs_dict['robot0']['/observations/eef/rotation_ortho6'] = np.array([quat_to_rot6d(obs['robot0_eef_quat_site']) for obs in rollout['observations']])
        obs_dict['robot0']['/observations/gripper'] = np.array([obs['robot0_gripper_pos'] for obs in rollout['observations']])
        obs_dict['robot1']['/observations/eef/position'] = np.array([obs['robot1_eef_pos'] for obs in rollout['observations']])
        obs_dict['robot1']['/observations/eef/rotation_ortho6'] = np.array([quat_to_rot6d(obs['robot1_eef_quat_site']) for obs in rollout['observations']])
        obs_dict['robot1']['/observations/gripper'] = np.array([obs['robot1_gripper_pos'] for obs in rollout['observations']])

        # Cameras
        for cam in task_config['robot0']['camera_names']:
            obs_dict['robot0'][f'/observations/images/{cam}'] = [obs[f'{cam}_image'] for obs in rollout['observations']]
        for cam in task_config['robot1']['camera_names']:
            obs_dict['robot1'][f'/observations/images/{cam}'] = [obs[f'{cam}_image'] for obs in rollout['observations']]

        for key in obs_dict['robot0'].keys():
            print(f"Key: {key}, shape: {np.array(obs_dict['robot0'][key]).shape}")
        for key in obs_dict['robot1'].keys():
            print(f"Key: {key}, shape: {np.array(obs_dict['robot1'][key]).shape}")
        for key in action_dict['robot0'].keys():
            print(f"Key: {key}, shape: {np.array(action_dict['robot0'][key]).shape}")
        for key in action_dict['robot1'].keys():
            print(f"Key: {key}, shape: {np.array(action_dict['robot1'][key]).shape}")

    else: 
        # Format action
        action_dict = {}

        # position = [np.concatenate((action[0:3], action[7:10])) for action in rollout['actions']]
        # action_dict['/action/position'] = np.array(position)

        # rot6d = [np.concatenate((rotvec_to_rot6d(action[3:6]), rotvec_to_rot6d(action[10:13]))) 
        #          for action in rollout['actions']]
        # action_dict['/action/rotation_ortho6'] = np.array(rot6d)

        # gripper = [np.array([action[6], action[13]]) for action in rollout['actions']]
        # action_dict['/action/gripper'] = np.array(gripper)

        action_dict['/action'] = [np.concatenate((action[0:3],
                                                  rotvec_to_rot6d(action[3:6]),
                                                  action[6:10],
                                                  rotvec_to_rot6d(action[10:13]),
                                                  np.array([action[13]]))) for action in rollout['actions']]
        action_dict['/action'] = np.array(action_dict['/action'])

        # Format observation
        obs_dict = {}

        # position = [np.concatenate((obs['robot0_eef_pos'], obs['robot1_eef_pos'])) 
        #             for obs in rollout['observations']]
        # obs_dict['/observations/eef/position'] = np.array(position)

        # rot6d = [np.concatenate((quat_to_rot6d(obs['robot0_eef_quat_site']), quat_to_rot6d(obs['robot1_eef_quat_site'])))
        #           for obs in rollout['observations']]
        # obs_dict['/observations/eef/rotation_ortho6'] = np.array(rot6d)

        # gripper = [np.array([obs['robot0_gripper_pos'], obs['robot1_gripper_pos']])
        #             for obs in rollout['observations']]
        # obs_dict['/observations/gripper'] = np.array(gripper)

        obs_dict['/observations/eef_pos'] = [np.concatenate((obs['robot0_eef_pos'],
                                                            quat_to_rot6d(obs['robot0_eef_quat_site']),
                                                            np.array([obs['robot0_gripper_pos']]),
                                                            obs['robot1_eef_pos'],
                                                            quat_to_rot6d(obs['robot1_eef_quat_site']),
                                                            np.array([obs['robot1_gripper_pos']])
                                                            )) for obs in rollout['observations']]
        obs_dict['/observations/eef_pos'] = np.array(obs_dict['/observations/eef_pos'])

        for cam in env_config['camera_names']:
            obs_dict[f'/observations/images/{cam}'] = [obs[f'{cam}_image'] 
                                                      for obs in rollout['observations']]
            obs_dict[f'/observations/images/{cam}'] = np.array(obs_dict[f'/observations/images/{cam}'])
    
        for key in obs_dict.keys():
            print(f"Key: {key}, shape: {obs_dict[key].shape}")
        for key in action_dict.keys():
            print(f"Key: {key}, shape: {action_dict[key].shape}")
            

    return action_dict, obs_dict 

def save_data(dataset_dir, collected_demo_num, max_timesteps, obs_dict, action_dict, task_config, env_config):

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    with h5py.File(os.path.join(dataset_dir, f'episode_{collected_demo_num}.hdf5'), 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True
        root.attrs['max_timesteps'] = max_timesteps
        root.attrs['camera_names'] = task_config['camera_names']
        root.attrs['control_frequency'] = env_config['control_freq']
        root.attrs['action_type'] = "absolute_pose"  # delta_pose

        for k, v in obs_dict.items():
            root.create_dataset(k, shape=np.array(v).shape)

        for name, array in obs_dict.items():
            root[name][...] = array[:max_timesteps]

        for k, v in action_dict.items():
            root.create_dataset(k, shape=np.array(v).shape)

        for name, array in action_dict.items():
            root[name][...] = array[:max_timesteps]

def main(args):

    # Get config
    with open(args['config_file'], 'r') as f:
        config = yaml.safe_load(f)
    decentralized = config['decentralized']
    env_config = config['env_parameters']
    task_config = config['task_parameters']
    max_timesteps = task_config['episode_len']

    # setup the environment
    controller_config = load_composite_controller_config(robot=env_config['robots'][0], controller=env_config['controller'])
    env = TwoArmHandoverRole(
        robots=env_config['robots'],
        gripper_types="default",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        prehensile=env_config['prehensile'],
        render_camera=None,
        camera_names=env_config['camera_names'],
        camera_heights=env_config['camera_heights'],
        camera_widths=env_config['camera_widths'],
        camera_depths=env_config['camera_depths'],
        camera_segmentations=env_config['camera_segmentations'],
        control_freq=env_config['control_freq'],
    )

    # setup the scripted policy
    player = PolicyPlayer(env)
    
    # collect demo
    collected_demo_num = 0
    episode_idx = 0
    while True:
    # for episode_idx in range(task_config['num_episodes']):
        start_time = timeit.default_timer()
        
        # get the demo
        rollout = player.get_demo(seed=env_config['seed'] + episode_idx)
        episode_idx += 1

        # check if to save the demo or not
        print(f"Length of rollout: {len(rollout['observations'])}")
        print("Save the demo? (y/n)")
        save = input()
        if save == 'n':
            continue
        elif save == 'y':
            pass
        else:
            print("Invalid input. The demo is not saved.")
            continue

        action_dict, obs_dict = format_action_observation(rollout, task_config, env_config, split_robot=decentralized)
        
        computation_time = timeit.default_timer() - start_time
        print('Episode:', episode_idx, 'duration:', round(computation_time, 2), end="\r")
    
        # fill the rest of the data to make each episode have the same length
        if decentralized:
            for robot in obs_dict.keys():
                for name, array in obs_dict[robot].items():
                    try:
                        elements_to_add = max_timesteps - len(array)
                        if elements_to_add > 0:
                            last_element = np.expand_dims(array[-1], axis=0)
                            pad_array = np.repeat(last_element, elements_to_add, axis=0)
                            array = np.concatenate([array, pad_array], axis=0)
                            # last_element = [array[-1]] * elements_to_add
                            # array.extend(last_element)
                            obs_dict[robot][name] = array
                    except:
                        print("ERROR", name, array)
            for robot in action_dict.keys():
                for name, array in action_dict[robot].items():
                    try:
                        elements_to_add = max_timesteps - len(array)
                        if elements_to_add > 0:
                            last_element = np.expand_dims(array[-1], axis=0)
                            pad_array = np.repeat(last_element, elements_to_add, axis=0)
                            array = np.concatenate([array, pad_array], axis=0)
                            # last_element = [array[-1]] * elements_to_add
                            # array.extend(last_element)
                            action_dict[robot][name] = array
                    except:
                        print("ERROR", name, array)
        else:
            for name, array in obs_dict.items():
                try:
                    elements_to_add = max_timesteps - len(array)
                    if elements_to_add > 0:
                        last_element = np.expand_dims(array[-1], axis=0)
                        pad_array = np.repeat(last_element, elements_to_add, axis=0)
                        array = np.concatenate([array, pad_array], axis=0)
                        # last_element = [array[-1]] * elements_to_add
                        # array.extend(last_element)
                        obs_dict[name] = array
                except:
                    print("ERROR", name, array)
            for name, array in action_dict.items():
                    try:
                        elements_to_add = max_timesteps - len(array)
                        if elements_to_add > 0:
                            last_element = np.expand_dims(array[-1], axis=0)
                            pad_array = np.repeat(last_element, elements_to_add, axis=0)
                            array = np.concatenate([array, pad_array], axis=0)
                            # last_element = [array[-1]] * elements_to_add
                            # array.extend(last_element)
                            action_dict[name] = array
                    except:
                        print("ERROR", name, array)
                        
        # save the data
        if decentralized:
            dataset_dir = os.path.join(task_config['dataset_dir'], 'robot0') # change
            save_data(dataset_dir, collected_demo_num, max_timesteps, 
                      obs_dict['robot0'], action_dict['robot0'], 
                      task_config['robot0'], env_config)
            print(f'Saved episode {collected_demo_num} to {dataset_dir}.hdf5')

            dataset_dir = os.path.join(task_config['dataset_dir'], 'robot1')
            save_data(dataset_dir, collected_demo_num, max_timesteps, 
                      obs_dict['robot1'], action_dict['robot1'], 
                      task_config['robot1'], env_config)
            print(f'Saved episode {collected_demo_num} to {dataset_dir}.hdf5')
        else:
            dataset_dir = os.path.join(task_config['dataset_dir'])
            save_data(dataset_dir, collected_demo_num, max_timesteps, 
                      obs_dict, action_dict, task_config, env_config)
            print(f'Saved episode {collected_demo_num} to {dataset_dir}.hdf5')

        collected_demo_num += 1
        if collected_demo_num == task_config['num_episodes']:
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="CompACT")
    parser.add_argument("--controller", type=str, default="role_bimanual/config/controller/kinova3_absolute_pose.json")
    parser.add_argument("--config_file", type=str, default="role_bimanual/config/train/CompACT.yaml")

    args = parser.parse_args()


    main(vars(args))