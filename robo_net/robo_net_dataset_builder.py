from typing import Iterator, Tuple, Any

import glob, cv2, json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from robo_net.conversion_utils import MultiThreadedDatasetBuilder


_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
_inst = 'Interact with the objects in the bin'
_language_embedding = _embed([_inst])[0].numpy()


def _cam_helper(idx, folder, t):
    path = folder + f'/cam{idx}/fr{t:03d}.jpg'
    return cv2.imread(path)[:,:,::-1].copy()


def _get_robot(path):
    if 'sawyer' in path:
        return 'sawyer'
    elif 'widowx' in path:
        return 'widowx'
    elif 'kuka' in path:
        return 'kuka'
    elif 'baxter' in path:
        return 'baxter'
    elif 'franka' in path:
        return 'franka'
    raise NotImplementedError()


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    def _parse_example(episode_path):
        # load raw data --> this should change for your dataset
        traj_data = np.load(episode_path + '/traj_data.npz')
        states = traj_data['state']
        actions = traj_data['actions']

        # bounds for gripper
        low_bound = traj_data['low_bound'][0,-1]
        high_bound = traj_data['high_bound'][0,-1]
        midpoint = (high_bound + low_bound) / 2.0

        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        episode = []
        for t, (s, ns, a) in enumerate(zip(states[:-1], states[1:], actions)):
            # compute Kona language embedding
            out_action = np.zeros((5,), dtype=np.float32)
            out_action[:4] = a.astype(np.float32)
            out_action[-1] = 1.0 if ns[-1] > midpoint else -1.0

            episode.append({
                'observation': {
                    'image': _cam_helper(0, episode_path, t),
                    'image1': _cam_helper(1, episode_path, t),
                    'image2': _cam_helper(2, episode_path, t),
                    'state': s.astype(np.float32),
                },
                'action': out_action,
                'discount': 1.0,
                'reward': 0,
                'is_first': t == 0,
                'is_last': t == (len(actions) - 1),
                'is_terminal': t == (len(actions) - 1),
                'language_instruction': _inst,
                'language_embedding': _language_embedding.copy(),
            })
        # create output data sample
        out_path = episode_path.split('/')[-1] + '.hdf5'
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': out_path,
                'robot': _get_robot(episode_path)
            }
        }
        # if you want to skip an example for whatever reason, simply return None
        return out_path, sample

    # for smallish datasets, use single-thread parsing
    for sample in paths:
        yield _parse_example(sample)



class RoboNet(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 20             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 1000  # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(240, 320, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'image1': tfds.features.Image(
                            shape=(240, 320, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'image2': tfds.features.Image(
                            shape=(240, 320, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(5,),
                            dtype=np.float32,
                            doc='Robot state, consists of eef position, wrist' 
                                'orientation, and gripper state' 
                                '[e_x, e_y, e_z, theta, gripper].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(5,),
                        dtype=np.float32,
                        doc='Robot action, consists of [3x eef delta, '
                            '1x wrist rotation, 1x gripper width target].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 0 for all random data.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'robot': tfds.features.Text(
                        doc='Robot model used during data collection.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define data splits."""
        cam_dict = json.load(open('/scratch/sudeep/robonet/out_dict.json'))
        train = ['/scratch/sudeep/robonet/frames/' + e[:-5] for e in cam_dict.keys()]
        return dict(train=train)
