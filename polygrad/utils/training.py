import os
import copy
import numpy as np
import torch
import einops
import pdb
import wandb
import time

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .errors import compute_traj_errors
from pathlib import Path
from os.path import join

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        env,
        renderer=None,
        ema_decay=0.995,
        train_batch_size=64,
        vis_batch_size=64,
        train_lr=2e-5,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        wandb_project="diffusion_world_models"
    ):
        super().__init__()

        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.env = env

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel
        self.batch_size = train_batch_size

        self.train_batch_size = train_batch_size
        self.vis_batch_size = vis_batch_size
        self.update_dataset(dataset)
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)
        self.scaler = torch.cuda.amp.GradScaler()

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.time = time.time()

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def update_dataset(self, dataset):
        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=self.train_batch_size, num_workers=2, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=self.vis_batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps, current_step=None):
        loss_sum = 0
        loss_count = 0
        metrics = dict()
        for step in range(n_train_steps):
            batch = next(self.dataloader)
            batch = batch_to_device(batch)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, loss_metrics = self.model.loss(batch.trajectories, batch.actions, batch.conditions)
            loss_sum += loss.item()
            loss_count += 1

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.log_freq == 0:
                loss_avg = loss_sum / loss_count
                prev = self.time
                now = time.time()
                print(f'{current_step}: {loss_avg:8.4f} | t: {(now- prev):8.4f}', flush=True)
                loss_sum = 0
                loss_count = 0
                self.time = now
                metrics.update({"loss": loss_avg})
                metrics.update(loss_metrics)

            if self.sample_freq and self.step % self.sample_freq == 0:
                if current_step is None:
                    current_step = self.step
                observations, actions, rewards, sim_states, seq = self.generate_samples(self.env)
                if self.renderer is not None:
                    savepath = join(self.logdir, f'step-{current_step}-unguided-traj.png')
                    self.renderer.composite(observations, actions, rewards, savepath=savepath)
                    for i in list(range(0, len(seq), 10)) + list(range(len(seq)-9, len(seq))):
                        savepath = join(self.logdir, f'step-{current_step}-unguided-diff-{i}.png')
                        obs = seq[i]
                        self.renderer.composite(obs, np.zeros_like(actions), np.zeros_like(rewards), savepath=savepath)
                error_metrics = compute_traj_errors(self.env, observations, actions, rewards, sim_states)
                metrics.update(error_metrics)
            self.step += 1
        return metrics

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        observations = trajectories[:, :, :self.dataset.observation_dim]
        rewards = trajectories[:, :, -2]
        actions = to_np(batch.actions)

        if "observations" in self.dataset.norm_keys:
            observations = self.dataset.normalizer.unnormalize(observations, 'observations')
        if "rewards" in self.dataset.norm_keys:
            rewards = self.dataset.normalizer.unnormalize(rewards, 'rewards')
        if "actions" in self.dataset.norm_keys:
            actions = self.dataset.normalizer.unnormalize(actions, 'actions')

        fig = self.renderer.composite(observations, actions, rewards)
        return wandb.Image(fig)
    
    def generate_samples(self, env):
        """
        Generate samples from ema diffusion models
        """
        ## get a single datapoint
        batch = self.dataloader_vis.__next__()
        conditions = to_device(batch.conditions, 'cuda:0')
        actions = to_device(batch.actions, 'cuda:0')

        if hasattr(env, "init_cond_for_viz"):
            conditions = env.init_cond_for_viz()
            conditions = self.dataset.normalizer.normalize(conditions, 'observations')
            conditions = {0: to_device(torch.tensor(conditions), "cuda:0")}

        ## [ n_samples x horizon x (action_dim + observation_dim) ]
        self.ema_model.eval()
        traj, _, seq, _ = self.ema_model(conditions, actions, return_sequence=True)
        traj = to_np(traj)
        self.ema_model.train()

        ## [ n_samples x horizon x observation_dim ]
        observations = traj[:, :, :self.dataset.observation_dim]
        rewards = traj[:, :, -2]
        actions = to_np(actions)

        ## [ n_samples x (horizon + 1) x observation_dim ]
        if "observations" in self.dataset.norm_keys:
            observations = self.dataset.normalizer.unnormalize(observations, 'observations')
            seq = [self.dataset.normalizer.unnormalize(x[:, :, :self.dataset.observation_dim], 'observations') for x in seq]
        if "rewards" in self.dataset.norm_keys:
            rewards = self.dataset.normalizer.unnormalize(rewards, 'rewards')
        if "actions" in self.dataset.norm_keys:
            actions = self.dataset.normalizer.unnormalize(actions, 'actions')

        
        return (observations, actions, rewards, batch.sim_states, seq)