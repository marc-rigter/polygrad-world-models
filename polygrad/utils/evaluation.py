import numpy as np
import torch
import torch.distributions as D
import wandb
from os.path import join


def get_standardized_stats(policy_distr, act):
    # Compute logprob with all action distributions normalized to standard normal.
    policy_mean = policy_distr.mean
    policy_std = policy_distr.stddev
    standard_normal = D.independent.Independent(
        D.normal.Normal(torch.zeros_like(policy_mean), torch.ones_like(policy_mean)), 1
    )
    normed_act = (act - policy_mean) / policy_std
    standard_logprob = standard_normal.log_prob(normed_act)

    act_stds = torch.std(normed_act, dim=[0, 1])
    act_means = torch.mean(normed_act, dim=[0, 1])
    return standard_logprob, act_stds, act_means


def evaluate_policy(
    policy,
    env,
    device,
    step,
    dataset,
    n_episodes=10,
    use_mean=False,
    renderer=None,
    savepath=None,
):
    """ """
    ep_lens = []
    rewards = []
    returns = []
    states = []
    actions = []

    for i in range(n_episodes):
        done = False
        state, _ = env.reset()

        t = 0
        ep_rewards = []
        ep_states = []
        ep_actions = []

        while not done:
            policy_dist = policy(
                torch.from_numpy(state).float().to(device), normed_input=False
            )
            if use_mean:
                act = policy_dist.mean
            else:
                act = policy_dist.sample()
            act = act.cpu().detach().numpy()
            next_state, rew, term, trunc, info = env.step(act)
            done = term or trunc

            ep_states.append(state.copy())
            ep_actions.append(act.copy())
            ep_rewards.append(rew.copy())
            t += 1
            state = next_state

        returns.append(sum(ep_rewards))
        rewards.append(np.array(ep_rewards))
        states.append(np.array(ep_states))
        actions.append(np.array(ep_actions))
        ep_lens.append(t)

    avg_return = np.mean(np.array(returns))
    min_return = np.min(np.array(returns))
    max_return = np.max(np.array(returns))
    avg_ep_len = np.mean(np.array(ep_lens))
    min_ep_len = np.min(np.array(ep_lens))
    max_ep_len = np.max(np.array(ep_lens))

    metrics = dict()
    metrics["avg_return"] = avg_return
    metrics["min_return"] = min_return
    metrics["max_return"] = max_return
    metrics["avg_ep_len"] = avg_ep_len
    metrics["min_ep_len"] = min_ep_len
    metrics["max_ep_len"] = max_ep_len

    if savepath is not None:
        savepath = join(savepath, f"step-{step}-real-policy-traj.png")

    if renderer is not None:
        fig = renderer.composite(states, actions, rewards, savepath)
        metrics.update({f"real-policy-traj": wandb.Image(fig)})
    return metrics
