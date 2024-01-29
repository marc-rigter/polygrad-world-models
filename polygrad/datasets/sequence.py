from collections import namedtuple
import numpy as np
import torch
import pdb

from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer


Batch = namedtuple('Batch', 'trajectories actions conditions sim_states')

class OnlineSequenceDataset(torch.utils.data.Dataset):
    """ Sequence dataset for online training.
    
    Requires:
        - prefill_episodes: these are used to compute normalisation constants
    """

    def __init__(self, prefill_episodes, horizon=64,
        normalizer='LimitsNormalizer', max_path_length=1000,
        max_n_episodes=20000, termination_penalty=0, use_padding=True,
        norm_keys=['observations', 'actions', 'rewards'], update_norm_interval=None,
        preprocess_fns=[]):

        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.update_norm_interval = update_norm_interval
        self.max_n_episodes = max_n_episodes
        self.termination_penalty = termination_penalty

        self.data_buffer = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        self.indices = []
        self.initialized = False

        for episode in prefill_episodes:
            self.add_episode(episode)

        # compute and fix normalisation constants based on prefill episodes
        self.normalizer = DatasetNormalizer(self.data_buffer, normalizer)
        self.update_normalizers()

        self.observation_dim = self.data_buffer.observations.shape[-1]
        self.action_dim = self.data_buffer.actions.shape[-1]
        self.n_episodes = self.data_buffer.n_episodes
        self.path_lengths = self.data_buffer.path_lengths
        self.norm_keys = norm_keys
        
    def reset_data_buffer(self):
        self.data_buffer = ReplayBuffer(self.max_n_episodes, self.max_path_length, self.termination_penalty)

    def add_episode(self, episode):
        """ Add an episode to the dataset. """
        self.data_buffer.add_path(episode)
        new_episode_num = self.data_buffer.n_episodes - 1
        self.update_indices(new_episode_num)

    def update_normalizers(self):
        self.normalizer.update_statistics(self.data_buffer)

    def get_metrics(self):
        return self.normalizer.get_metrics()
        
    def update_indices(self, new_episode_num):
        '''
            update indices for sampling from dataset to include new episode
        '''

        path_length = self.data_buffer.path_lengths[new_episode_num]
        max_start = min(path_length - 1, self.max_path_length - self.horizon)

        max_start = min(path_length - 1, self.max_path_length - self.horizon)
        if not self.use_padding:
            max_start = min(max_start, path_length - self.horizon)

        [self.indices.append((new_episode_num, start, start + self.horizon)) for start in range(max_start)]
        return 

    def __len__(self):
        return len(self.indices)
    
    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __getitem__(self, _):
        """
        Sample a random batch
        """
        idx = np.random.randint(0, len(self.indices))
        path_ind, start, end = self.indices[idx]

        trajectory_list = []
        for key in ['observations', 'rewards', 'terminals']:
            data = self.data_buffer[key][path_ind, start:end]
            if key in self.norm_keys:
                data = self.normalizer(data, key)
            trajectory_list.append(data)

        actions = self.data_buffer['actions'][path_ind, start:end]
        if 'actions' in self.norm_keys:
            actions = self.normalizer(actions, 'actions')

        sim_states = self.data_buffer['sim_states'][path_ind, start:end]

        conditions = self.get_conditions(trajectory_list[0])
        trajectories = np.concatenate(trajectory_list, axis=-1)
        batch = Batch(trajectories, actions, conditions, sim_states)
        return batch
    
    def reset(self):
        self.indices = []

