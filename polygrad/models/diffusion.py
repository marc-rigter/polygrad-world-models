from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb
import torch.nn.functional as F
from polygrad.sampling.functions import default_sample_fn, policy_guided_sample_fn

import polygrad.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


Sample = namedtuple('Sample', 'trajectories chains recons_after_guide recons_before_guide')

def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values

def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        noise_sched_tau=1.0, action_condition_noise_scale=1.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + 2 # obs + reward + terminals
        self.model = model
        self.action_condition_noise_scale = action_condition_noise_scale

        betas = cosine_beta_schedule(n_timesteps, tau=noise_sched_tau)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
        coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)
        self.register_buffer('posterior_mean_coef1', coef1)
        self.register_buffer('posterior_mean_coef2', coef2)

        ## initialize objective
        self.loss_fn = Losses[loss_type]()

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, act, t):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = self.model(x, act, t)
        x_recon = self.predict_start_from_noise(x, t=t, noise=prediction)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample_loop(self, shape, cond, act=None, normalizer=None, policy=None, return_sequence=False, verbose=True, return_chain=False, **sample_kwargs):
        if policy is None:
            sample_fn = default_sample_fn
        else:
            sample_fn = policy_guided_sample_fn

        x = torch.randn(shape, device=self.betas.device)
        x = apply_conditioning(x, cond, self.observation_dim)
        if sample_fn is not policy_guided_sample_fn:
            act_noisy = act
        else:
            act_noisy = torch.randn((shape[0], shape[1], self.action_dim), device=x.device)
        seq = []
        for t in reversed(range(0, self.n_timesteps)):
            x, act_noisy, metrics = sample_fn(
                    self,
                    x,
                    act_noisy,
                    cond,
                    t,
                    condition_noise_scale=self.action_condition_noise_scale,
                    q_sample=self.q_sample,
                    normalizer=normalizer, 
                    policy=policy,
                    **sample_kwargs)
            if sample_fn is not policy_guided_sample_fn and t > 0:
                act_noisy = act
            x = apply_conditioning(x, cond, self.observation_dim)
            if return_sequence:
                seq.append(x.cpu().detach().numpy())
        return x, act_noisy, seq, metrics
    
    def conditional_sample(self, cond, act=None, normalizer=None, policy=None, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, act=act, normalizer=normalizer, policy=policy, **sample_kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, act, cond, t):
        noise = torch.randn_like(x_start)

        traj_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        traj_noisy = apply_conditioning(traj_noisy, cond, self.observation_dim)

        act_noisy = self.q_sample(x_start=act, t=t, noise=self.action_condition_noise_scale * torch.randn_like(act))
        
        traj_recon = self.model(traj_noisy, act_noisy, t)
        traj_recon = apply_conditioning(traj_recon, cond, self.observation_dim)

        if self.predict_epsilon:
            target = noise
        else:
            target = x_start

        loss = self.loss_fn(traj_recon, target)
        loss_metrics = {}

        with torch.no_grad():
            loss_metrics["obs_mse_loss"] = F.mse_loss(traj_recon[:, :, :self.observation_dim], target[:, :, :self.observation_dim]).item()
            loss_metrics["reward_mse_loss"] = F.mse_loss(traj_recon[:, :, -2], target[:, :, -2]).item()
            loss_metrics["term_mse_loss"] = F.mse_loss(traj_recon[:, :, -1], target[:, :, -1]).item()
        return loss, loss_metrics

    def loss(self, x, act, cond):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, act, cond, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)


