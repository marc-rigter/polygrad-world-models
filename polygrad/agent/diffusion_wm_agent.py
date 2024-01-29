import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import dill as pickle 
import os
import time
import wandb

from torch import Tensor
from os.path import join

from .functions import *
from .common import *
from .utils import *
from polygrad.utils.errors import compute_traj_errors
from pathlib import Path


class DiffusionWMAgent(nn.Module):

    def __init__(self,
                 diffusion_model,
                 actor_critic,
                 dataset,
                 log_path,
                 env,
                 diffusion_method,
                 renderer=None,
                 guidance_scale=1.0,
                 log_interval=100,
                 tune_guidance=False,
                 guidance_type='grad',
                 guidance_lr=1e-3,
                 action_guidance_noise_scale=1.0,
                 update_states=False,
                 clip_std=None,
                 states_for_guidance='recon',
                 rollout_steps=None,
                 clip_state_change=1.0,
                 ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.diffusion_model = diffusion_model
        self.ac = actor_critic
        self.env = env
        self.dataset = dataset
        self.log_path = log_path
        self.renderer = renderer
        self.log_interval = log_interval
        self.guidance_type = guidance_type
        self.action_guidance_noise_scale = action_guidance_noise_scale
        self.log_guidance = torch.log(torch.tensor(guidance_scale)).to(self.device)
        self.tune_guidance = tune_guidance
        self.update_states = update_states
        self.clip_std = clip_std
        self.states_for_guidance = states_for_guidance
        self.clip_state_change = clip_state_change
        if self.tune_guidance:
            self.log_guidance.requires_grad_(True)
            self._guidance_optimizer = torch.optim.SGD([self.log_guidance], lr=guidance_lr)
        self.last_log_step = -1
        self.rollout_steps = rollout_steps
        self.diffusion_method = diffusion_method

        assert self.guidance_type in ['grad', 'sample', 'none']
        assert self.diffusion_method in ['polygrad', 'autoregressive']

    def imagine_polygrad(self, conditions):
        trajs, imag_actions, seq, sampling_metrics = self.diffusion_model(
            conditions,
            policy=self.ac.forward_actor,
            verbose=False,
            normalizer=self.dataset.normalizer,
            guidance_scale=torch.exp(self.log_guidance),
            guidance_type=self.guidance_type,
            action_noise_scale=self.action_guidance_noise_scale,
            update_states=self.update_states,
            clip_std=self.clip_std,
            states_for_guidance=self.states_for_guidance,
            clip_state_change=self.clip_state_change,
        )
         
        imag_obs = trajs[:, :, :self.dataset.observation_dim]
        imag_rewards = trajs[:, :, -2]
        imag_terminals = trajs[:, :, -1]
        imag_terminals = self.unnormalize(imag_terminals, "terminals")
        return imag_obs, imag_actions, imag_rewards, imag_terminals, sampling_metrics

    def imagine(self, conditions):
        self.diffusion_model.eval()
        metrics = dict()
        start = time.time()
        if self.diffusion_method == "polygrad":
            imag_obs, imag_actions, imag_rewards, imag_terminals, sampling_metrics = self.imagine_polygrad(conditions)
        else:
            raise NotImplementedError
        metrics[f"imagine_time/step_{self.diffusion_model.horizon}"] = time.time() - start
        self.diffusion_model.train()
    
        term_binary = torch.zeros_like(imag_terminals)
        term_binary[imag_terminals > 0.5] = 1.0
        metrics["terminal_avg"] = term_binary.mean().item()
        [metrics.update({f"sampling/{key}": sampling_metrics[key]}) for key in sampling_metrics.keys()]
        return imag_obs, imag_actions, imag_rewards, term_binary, metrics
    
    def unnormalize(self, data, key):
        if key in self.dataset.norm_keys:
            return self.dataset.normalizer.unnormalize(data, key)
        else:
            return data
        
    def normalize(self, data, key):
        if key in self.dataset.norm_keys:
            return self.dataset.normalizer.normalize(data, key)
        else:
            return data
        
    def update_guidance(self, value, target):
        loss = -(self.log_guidance * (target - value)).mean()
        self._guidance_optimizer.zero_grad()
        loss.backward()
        self._guidance_optimizer.step()
    
    def get_metrics(
            self, 
            obs_norm,
            act_norm,
            rew_norm,
            sim_states,
            device, 
            step, 
            max_log=50,
        ):
        metrics = dict()
        obs = self.unnormalize(obs_norm, "observations")
        act = self.unnormalize(act_norm, "actions")
        rew = self.unnormalize(rew_norm, "rewards")
        metrics["data/imag_obs_norm_mean"] = np.mean(obs_norm)
        metrics["data/imag_obs_norm_std"] = np.std(obs_norm)
        metrics["data/imag_act_norm_mean"] = np.mean(act_norm)
        metrics["data/imag_act_norm_std"] = np.std(act_norm)
        metrics["data/imag_rew_norm_mean"] = np.mean(rew_norm)
        metrics["data/imag_rew_norm_std"] = np.std(rew_norm)
        metrics["data/imag_obs_mean"] = np.mean(obs)
        metrics["data/imag_obs_std"] = np.std(obs)
        metrics["data/imag_act_mean"] = np.mean(act)
        metrics["data/imag_act_std"] = np.std(act)
        metrics["data/imag_rew_mean"] = np.mean(rew)
        metrics["data/imag_rew_std"] = np.std(rew)
            
        # compute imagined to real dynamics
        error_metrics = compute_traj_errors(self.env, obs[:max_log], act[:max_log], rew[:max_log], sim_states=sim_states[:max_log])
        metrics.update(error_metrics)
        return metrics

    def training_step(self, batch, step, log_only=False, max_log=50):
        obs_norm, act_norm, rew_norm, term, metrics = self.imagine(batch.conditions)
        if (step >= self.last_log_step + self.log_interval):
            metrics.update(self.get_metrics(obs_norm.cpu().detach().numpy(),
                                            act_norm.cpu().detach().numpy(),
                                            rew_norm.cpu().detach().numpy(),
                                            batch.sim_states,
                                            self.device, 
                                            step,
                                            max_log=max_log))
            self.last_log_step = step

        ac_metrics = self.ac.training_step(
                        states=obs_norm,
                        actions=act_norm, 
                        rewards=rew_norm,
                        terminals=term,
                        env_step=step,
                        log_only=log_only, 
                      )       
        metrics.update(ac_metrics)
        if self.tune_guidance and self.guidance_type =='grad':
            self.update_guidance(1.0, metrics["act_std"])
        metrics.update({"guidance_scale": torch.exp(self.log_guidance).item(),
                        "action_std_error": 1.0 - metrics["act_std"]})
        return metrics
    
    def save(self, path, step):
        """ Save the actor critic, diffusion model and current dataset. """

        ac_path = join(path, f"step-{step}-ac.pt")
        diffusion_path = join(path, f"step-{step}-diffusion.pt")
        torch.save(self.ac.state_dict(), ac_path)
        torch.save(self.diffusion_model.state_dict(), diffusion_path)

        dataset_path = join(path, f"step-{step}-dataset.pkl")
        if os.path.exists(dataset_path):
            os.remove(dataset_path)
        with open(dataset_path, 'wb') as f:
            pickle.dump(self.dataset, f)
        return
    
    def load(self, path, step, load_a2c=True, load_diffusion=True, load_dataset=True):
        """ Load the actor critic and diffusion model. """

        if load_a2c:
            ac_path = join(path, f"step-{step}-ac.pt")
            self.ac.load_state_dict(torch.load(ac_path))

        if load_diffusion:
            diffusion_path = join(path, f"step-{step}-diffusion.pt")
            self.diffusion_model.load_state_dict(torch.load(diffusion_path))

        if load_dataset:
            dataset_path = join(path, f"step-{step}-dataset.pkl")
            with open(dataset_path, 'rb') as f:
                self.dataset = pickle.load(f)
        return
       
