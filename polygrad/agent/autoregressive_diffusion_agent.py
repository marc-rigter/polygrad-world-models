import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import dill as pickle 
import time

from .functions import *
from .common import *
from .polygrad_wm_agent import *

class AutoregressiveDiffusionAgent(PolygradWMAgent):
    
    def imagine(self, conditions, return_sequence=False):
        metrics = dict()
        imag_states = torch.zeros(conditions[0].shape[0], self.rollout_steps, self.dataset.observation_dim).to(self.device)
        imag_act = torch.zeros(conditions[0].shape[0], self.rollout_steps, self.dataset.action_dim).to(self.device)
        imag_rewards = torch.zeros(conditions[0].shape[0], self.rollout_steps).to(self.device)
        imag_terminals = torch.zeros(conditions[0].shape[0], self.rollout_steps).to(self.device)

        self.diffusion_model.eval()
        start = time.time()
        for i in range(self.rollout_steps):
            current_state_normed = conditions[0]
            policy_dist = self.ac.forward_actor(current_state_normed.to(self.device), normed_input=True)
            actions = policy_dist.sample().unsqueeze(1)
            actions = self.normalize(actions, "actions")
            actions = torch.cat([actions, torch.zeros_like(actions)], dim=1) # add dummy action

            imag_states[:, i, :] = current_state_normed
            imag_act[:, i, :] = actions[:, 0, :]

            trajs, _, _, _ = self.diffusion_model(
                conditions,
                act=actions,
                policy=None,
                verbose=False,
            )
            imag_rewards[:, i] = trajs[:, 0, -2]
            imag_terminals[:, i] = trajs[:, 0, -1]

            # update next state
            conditions[0] = trajs[:, -1, :self.dataset.observation_dim]
            metrics[f"imagine_time/step_{i+1}"] = time.time() - start
        self.diffusion_model.train()

        imag_terminals = self.unnormalize(imag_terminals, "terminals")
        term_binary = torch.zeros_like(imag_terminals)
        term_binary[imag_terminals > 0.5] = 1.0
        metrics["terminal_avg"] = term_binary.mean().item()
        return imag_states, imag_act, imag_rewards, imag_terminals, metrics, None
       
