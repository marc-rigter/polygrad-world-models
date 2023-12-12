import numpy as np
import torch
import torch.distributions as D
import copy

def random_exploration(n_steps, env):
    """
    """
    episodes = []

    i = 0
    while i < n_steps:
        done = False
        state, _ = env.reset()
        ep_states = []
        ep_actions = []
        ep_next_states = []
        ep_rewards = []
        ep_terminals = []
        ep_sim_states = []

        while not done:
            act = env.action_space.sample()
            next_state, rew, term, trunc, info = env.step(act) 
            done = term or trunc

            ep_states.append(state.copy())
            ep_actions.append(act.copy())
            ep_next_states.append(next_state.copy())
            ep_rewards.append(rew.copy())
            ep_terminals.append(term)
            if "sim_state" in info.keys():
                ep_sim_states.append(info["sim_state"].copy())
            else:
                ep_sim_states.append(None)

            state = next_state
            i += 1
        
        episode = {
            "observations": np.array(ep_states),
            "actions": np.array(ep_actions),
            "next_observations": np.array(ep_next_states),
            "rewards": np.array(ep_rewards),
            "terminals": np.array(ep_terminals),
            "timeouts": np.array([False] * len(ep_rewards)),
            "sim_states": np.array(ep_sim_states)
        }
        episodes.append(episode)
    return episodes



