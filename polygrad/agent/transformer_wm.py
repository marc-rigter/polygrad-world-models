import math
import inspect
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class TransformerWM:
    def __init__(self, model, context_length, rollout_length):
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.context_length = context_length
        self.rollout_length = rollout_length
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, batch):
        metrics = dict()

        input_obs = batch.trajectories[:, :-1].to(self.device)
        input_act = batch.actions[:, :-1].to(self.device)
        targets = batch.trajectories[:, 1:].to(self.device)

        # forward pass
        predictions = self.model(input_obs, input_act)
        loss = self.loss_fn(predictions, targets)
        metrics["loss"] = loss.item()

        # backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return metrics

    def imagine(self, batch, policy):
        batch_size = batch.trajectories.shape[0]
        imag_states = torch.zeros(
            batch_size, self.rollout_length,batch.trajectories.shape[2] - 2,
        ).to(self.device)
        imag_act = torch.zeros(
            batch_size, self.rollout_length, batch.actions.shape[2]
        ).to(self.device)
        imag_rewards = torch.zeros(batch_size, self.rollout_length).to(self.device)
        imag_terminals = torch.zeros(batch_size, self.rollout_length).to(self.device)

        obs_context = batch.trajectories[:, 0:1].to(self.device)
        metrics = dict()
        start = time.time()
        for i in range(self.rollout_length):
            current_obs = obs_context[:, -1, :-2]
            current_act = policy(current_obs, normed_input=True).sample()
            if i == 0:
                act_context = current_act.unsqueeze(1)
            else:
                act_context = torch.cat([act_context, current_act.unsqueeze(1)], dim=1)
                act_context = act_context[:, -self.context_length :]
            rew = obs_context[:, -1, -2]
            term = obs_context[:, -1, -1]
            imag_states[:, i, :] = current_obs
            imag_act[:, i, :] = current_act
            imag_rewards[:, i] = rew
            imag_terminals[:, i] = term

            predictions = self.model(obs_context, act_context).detach()
            obs_context = torch.cat([obs_context, predictions[:, -1:]], dim=1)
            obs_context = obs_context[:, -self.context_length :]
            metrics[f"imagine_time/step_{i+1}"] = time.time() - start
        return imag_states, imag_act, imag_rewards, imag_terminals, metrics
