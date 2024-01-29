import torch

from polygrad.models.helpers import (
    extract,
    apply_conditioning,
)
import numpy as np


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


@torch.no_grad()
def default_sample_fn(
    model, x, act, cond, t, q_sample, condition_noise_scale, policy, normalizer
):
    timesteps = make_timesteps(x.shape[0], t, x.device)

    # rescale actions
    act_scale = q_sample(
        act, timesteps, noise=torch.randn_like(act) * condition_noise_scale
    )
    model_mean, _, model_log_variance = model.p_mean_variance(
        x=x, act=act_scale, t=timesteps
    )
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[timesteps == 0] = 0
    return model_mean + model_std * noise, None, 0.0


def clip_change(old, new, tol=1.0):
    new = old + torch.clamp((new - old), min=-tol, max=tol)
    return new


def policy_guided_sample_fn(
    model,
    x,
    act_noisy,
    cond,
    t,
    q_sample,
    policy,
    normalizer,
    condition_noise_scale=0.0,
    guidance_scale=1.0,
    action_noise_scale=1.0,
    clip_std=None,
    states_for_guidance="recon",
    update_states=False,
    guidance_type="grad",
    clip_state_change=1.0,
):
    """Compute new sample after one step of denoising by diffusion model with policy guidance."""
    assert guidance_type in ["grad", "sample", "none"]
    timesteps = make_timesteps(x.shape[0], t, x.device)

    # compute predicted denoised trajectory
    with torch.no_grad():
        act_scale = q_sample(act_noisy, timesteps, noise=torch.zeros_like(act_noisy))
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model.model(x, act_scale, timesteps)
        x_recon = model.predict_start_from_noise(x, t=timesteps, noise=prediction)
        x_recon = apply_conditioning(x_recon, cond, model.observation_dim)

    model_mean, _, model_log_variance = model.q_posterior(
        x_start=x_recon, x_t=x, t=timesteps
    )
    model_std = torch.exp(0.5 * model_log_variance)
    noise = torch.randn_like(x)

    # clip magnitude of change near end of diffusion
    if t <= 10:
        model_mean = clip_change(x, model_mean, clip_state_change)

    if states_for_guidance == "recon":
        guide_states = x_recon[:, :, : model.observation_dim].detach()
    elif states_for_guidance == "posterior_mean":
        guide_states = model_mean[:, :, : model.observation_dim].detach()
    else:
        raise NotImplementedError

    # no guidance when t == 0
    if t == 0:
        if clip_std is not None:
            act_noisy_unnormed = normalizer.unnormalize(act_noisy, "actions")
            policy_dist = policy(guide_states, normed_input=True)
            act_noisy_unnormed = torch.clamp(
                act_noisy_unnormed,
                min=policy_dist.mean - clip_std * policy_dist.stddev,
                max=policy_dist.mean + clip_std * policy_dist.stddev,
            )
            act_noisy = normalizer.normalize(act_noisy_unnormed, "actions")
        metrics = {
            "avg_change": (model_mean - x).abs().mean().item(),
            "max_change": (model_mean - x).abs().max().item(),
            "min_change": (model_mean - x).abs().min().item(),
            "std_change": (model_mean - x).abs().std().item(),
        }
        return model_mean, act_noisy, metrics

    if guidance_type == "grad":
        # unnormalize as policy ouputs unnormalized actions
        act_noisy_unnormed = normalizer.unnormalize(act_noisy, "actions")

        # compute policy distribution at denoised states
        with torch.no_grad():
            policy_dist = policy(guide_states, normed_input=True)

        if clip_std is not None:
            act_noisy_unnormed = torch.clamp(
                act_noisy_unnormed,
                min=policy_dist.mean - clip_std * policy_dist.stddev,
                max=policy_dist.mean + clip_std * policy_dist.stddev,
            )

        # if not act_noisy_unnormed.requires_grad:
        act_noisy_unnormed.requires_grad = True

        # backprop likelihood of actions in predicted trajectory under policy
        act_logprob = policy_dist.log_prob(act_noisy_unnormed)
        loss = act_logprob.sum()
        loss.backward()

        # gradient update to actions
        act_grad = act_noisy_unnormed.grad.detach()
        act_noisy_unnormed = (act_noisy_unnormed + guidance_scale * act_grad).detach()

        # gradient update to states
        if update_states:
            guide_states.requires_grad = True
            policy_dist = policy(guide_states, normed_input=True)
            act_logprob = policy_dist.log_prob(act_noisy_unnormed)
            loss = act_logprob.sum()
            loss.backward()
            obs_grad = guide_states.grad.detach()
            obs_recon = (guide_states + guidance_scale * obs_grad).detach()
            x_recon[:, :, : model.observation_dim] = obs_recon
            x_recon = apply_conditioning(x_recon, cond, model.observation_dim)
            model_mean, _, model_log_variance = model.q_posterior(
                x_start=x_recon, x_t=x, t=timesteps
            )

        # normalize actions
        act_denoised = normalizer.normalize(act_noisy_unnormed, "actions")
        act_sample = act_denoised + action_noise_scale * model_std * torch.randn_like(
            act_denoised
        )

    elif guidance_type == "sample":
        with torch.no_grad():
            policy_dist = policy(guide_states, normed_input=True)
        act_sample_unnormed = policy_dist.sample()
        act_sample = normalizer.normalize(act_sample_unnormed, "actions")

    elif guidance_type == "none":
        act_sample = act_noisy

    return model_mean + model_std * noise, act_sample, 0.0
